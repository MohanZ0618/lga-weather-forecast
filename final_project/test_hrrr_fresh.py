"""Checks for the newest-run HRRR features (hrrr_fresh.py): availability, daylight saving, lead window."""

import unittest
from unittest import mock

from numcodecs import blosc
import numpy as np
import pandas as pd

import hrrr_fresh as fresh
from hrrr import GRID_POINT, TILE_SIZE
from prepare import local_cutoff

UTC = lambda text: pd.Timestamp(text, tz="UTC")


def tile(kelvin, leads):
    """A blosc chunk shaped like hrrrzarr's, constant per lead, with the LGA point set to kelvin."""
    grid = np.full((leads, TILE_SIZE, TILE_SIZE), 250.0, dtype="<f4")
    grid[:, GRID_POINT[0] % TILE_SIZE, GRID_POINT[1] % TILE_SIZE] = kelvin
    return blosc.compress(grid, b"lz4", 9, 1)


class Response:
    def __init__(self, modified, content=b"", status=200):
        self.status_code, self.content = status, content
        self.headers = {"Last-Modified": modified.strftime("%a, %d %b %Y %H:%M:%S GMT")}


class BundledTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cycles = pd.read_csv(fresh.CYCLES, parse_dates=["date"])
        cls.features = pd.read_csv(fresh.FEATURES, parse_dates=["date"])
        cls.ok = cls.cycles.loc[cls.cycles.status.eq("ok")]

    def test_no_run_archived_after_its_cutoff(self):
        archived = pd.to_datetime(self.ok.archived_utc, utc=True)
        cutoff = pd.to_datetime(self.ok.cutoff_utc, utc=True)
        run = pd.to_datetime(self.ok.run_utc, utc=True)
        self.assertGreater(len(self.ok), 5000)
        self.assertFalse((archived > cutoff).any())
        self.assertFalse((archived < run).any())
        age = (cutoff - run) / pd.Timedelta(hours=1)
        self.assertTrue(age.between(1, fresh.MAX_AGE).all())

    def test_cutoffs_are_new_york_local_times(self):
        expected = [local_cutoff(d, h).tz_convert("UTC") for d, h in zip(self.cycles.date, self.cycles.hour)]
        self.assertTrue((pd.to_datetime(self.cycles.cutoff_utc, utc=True).to_numpy() == pd.DatetimeIndex(expected).to_numpy()).all())

    def test_no_newer_qualifying_run_was_skipped(self):
        for row in self.cycles.loc[self.cycles.status.isin(["ok", "no_run"])].itertuples():
            cutoff = pd.Timestamp(row.cutoff_utc)
            tried = str(row.tried).split(";") if isinstance(row.tried, str) else []
            expected = [f"{r.hour:02}z" for r in fresh.candidate_runs(row.date, row.hour)]
            self.assertEqual([t.split(":")[0] for t in tried], expected[:len(tried)])
            if row.status == "no_run":
                self.assertEqual(len(tried), len(expected))
            for entry in tried[:-1] if row.status == "ok" else tried:
                state = entry.split(":", 2)
                self.assertIn(state[1], ("missing", "late"))
                if state[1] == "late":
                    self.assertGreater(pd.Timestamp(state[2]), cutoff)

    def test_features_match_the_bundled_point_series(self):
        self.assertEqual(len(self.features), len(self.ok))
        for row in self.ok.sample(200, random_state=4260).itertuples():
            temps = np.array([getattr(row, f"f{lead:02}") for lead in range(1, int(row.n_leads) + 1)])
            out = fresh.features(row.date, row.hour, pd.Timestamp(row.run_utc), temps)
            match = self.features.loc[self.features.date.eq(row.date) & self.features.hour.eq(row.hour)].iloc[0]
            for name, value in out.items():
                self.assertAlmostEqual(match[name], value, places=2)


class DaylightSaving(unittest.TestCase):
    def test_cutoffs_shift_on_transition_days(self):
        for day, hours in (("2024-03-09", (14, 17, 20)), ("2024-03-10", (13, 16, 19)),
                           ("2024-11-02", (13, 16, 19)), ("2024-11-03", (14, 17, 20))):
            self.assertEqual(tuple(fresh.cutoff_utc(day, h).hour for h in (9, 12, 15)), hours)

    def test_local_day_ends_at_new_york_midnight(self):
        self.assertEqual(fresh.day_hours("2024-03-10", 9)[-1], UTC("2024-03-11 03:00"))  # EDT from 02:00
        self.assertEqual(fresh.day_hours("2024-11-03", 9)[-1], UTC("2024-11-04 04:00"))  # EST from 02:00
        self.assertEqual(len(fresh.day_hours("2024-07-01", 15)), 9)  # 15:00 through 23:00

    def test_candidates_need_leads_to_local_midnight(self):
        # An 18-lead 09 UTC run ends at 03 UTC: 23:00 in summer, 22:00 in winter.
        hours = lambda day: [r.hour for r in fresh.candidate_runs(day, 9)]
        self.assertEqual(hours("2024-03-10"), [12, 11, 10, 9])
        self.assertEqual(hours("2024-03-09"), [13, 12, 11, 10])
        self.assertEqual([r.hour for r in fresh.candidate_runs("2024-01-15", 15)], [19, 18, 17, 16, 15, 14])


class LeadWindow(unittest.TestCase):
    def test_rest_of_day_excludes_the_past_and_after_midnight(self):
        init = UTC("2024-07-01 17:00")  # 15:00 EDT cutoff is 19 UTC; midnight EDT is 04 UTC
        temps = np.arange(18, dtype=float)
        temps[0] = 99  # 18 UTC, before the cutoff
        temps[11] = 98  # 05 UTC next day, after midnight
        out = fresh.features(pd.Timestamp("2024-07-01"), 15, init, temps)
        self.assertEqual(out["hrrr_fresh_now"], 1)  # lead 2 is valid at 19 UTC
        self.assertEqual(out["hrrr_fresh_remaining_max"], 9)  # lead 10 is valid at 03 UTC = 23:00 EDT
        self.assertEqual(out["hrrr_fresh_warming"], 8)
        self.assertEqual(out["hrrr_fresh_age_h"], 2)

    def test_synoptic_run_uses_48_leads(self):
        init = UTC("2024-01-15 12:00")
        temps = np.zeros(48)
        temps[15] = 50  # valid 04 UTC = 23:00 EST, still the same local day
        temps[16] = 60  # valid 05 UTC = midnight
        out = fresh.features(pd.Timestamp("2024-01-15"), 9, init, temps)
        self.assertEqual((out["hrrr_fresh_remaining_max"], out["hrrr_fresh_age_h"]), (50, 2))

    def test_run_that_stops_before_midnight_is_rejected(self):
        with self.assertRaises(ValueError):
            fresh.features(pd.Timestamp("2024-01-15"), 9, UTC("2024-01-15 09:00"), np.zeros(18))


class Availability(unittest.TestCase):
    def test_rule_includes_the_cutoff_second(self):
        init, cutoff = UTC("2024-07-01 11:00"), UTC("2024-07-01 13:00")
        self.assertTrue(fresh.available(init, cutoff, cutoff))
        self.assertFalse(fresh.available(init, cutoff + pd.Timedelta(seconds=1), cutoff))
        self.assertFalse(fresh.available(init, init - pd.Timedelta(seconds=1), cutoff))

    def test_fetch_keeps_the_newest_run_archived_by_the_cutoff(self):
        # 09:00 EDT on 2024-07-01 is 13 UTC. 12z was archived late, 11z in time.
        modified = {12: UTC("2024-07-01 14:10"), 11: UTC("2024-07-01 12:56")}
        content = tile(np.arange(18) + 280.0, 18)

        def request(method, url):
            hour = int(url.split("_")[1][:2])
            if hour not in modified:
                return Response(UTC("2024-07-01 00:00"), status=404)
            return Response(modified[hour], content)

        with mock.patch.object(fresh, "request", request):
            record = fresh.fetch_one(("2024-07-01", 9))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(pd.Timestamp(record["run_utc"]).hour, 11)
        self.assertTrue(record["tried"].startswith("12z:late:"))
        np.testing.assert_allclose(record["temps"], (np.arange(18) + 280.0 - 273.15) * 1.8 + 32, atol=1e-3)

    def test_fetch_rechecks_last_modified_on_the_download(self):
        heads = iter([UTC("2024-07-01 12:56"), UTC("2024-07-01 15:00"), UTC("2024-07-01 11:58"),
                      UTC("2024-07-01 11:58")])
        content = tile(np.full(18, 290.0), 18)
        with mock.patch.object(fresh, "request", lambda method, url: Response(next(heads), content)):
            record = fresh.fetch_one(("2024-07-01", 9))
        # 12z: HEAD in time but GET later than the cutoff, so it is rejected; 11z then qualifies.
        self.assertEqual(pd.Timestamp(record["run_utc"]).hour, 11)
        self.assertTrue(record["tried"].startswith("12z:late:2024-07-01T15:00"))


if __name__ == "__main__":
    unittest.main()
