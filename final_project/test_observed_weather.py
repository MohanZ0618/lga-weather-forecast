"""Checks for the IEM ASOS features (asos.py) and the observed-weather experiment."""

import unittest

import numpy as np
import pandas as pd

import asos
from asos import ALLOWANCES, FEATURES, STATIONS, build_rows, clean, latest, wind_components

ROOT = asos.DATA.parent


def synthetic_raw(start="2024-03-08", end="2024-03-12", seed=0, minute=51, specials=True):
    """Routine reports at :minute every hour plus random specials, with every field the builder reads."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{start} 00:{minute:02}", f"{end} 23:{minute:02}", freq="h", tz="UTC")
    if specials:
        times = times.append(times[rng.random(len(times)) < 0.3] + pd.Timedelta(minutes=17)).sort_values()
    n = len(times)
    covers = rng.choice(["CLR", "FEW", "SCT", "BKN", "OVC"], size=(n, 3))
    raw = pd.DataFrame({
        "station": "LGA", "valid": times.tz_convert(None).strftime("%Y-%m-%d %H:%M"),
        "tmpf": (50 + rng.normal(0, 5, n)).round(), "drct": rng.integers(0, 36, n) * 10,
        "sknt": rng.integers(0, 20, n), "gust": np.where(rng.random(n) < 0.2, 25, np.nan),
        "alti": 30 + rng.normal(0, 0.1, n), "mslp": np.nan, "skyc1": covers[:, 0], "skyc2": covers[:, 1],
        "skyc3": covers[:, 2], "skyl1": 2000.0, "skyl2": 5000.0, "skyl3": 9000.0,
        "p01i": np.where(rng.random(n) < 0.3, 0.05, 0.0), "vsby": 10.0, "wxcodes": ""})
    raw["dwpf"] = raw.tmpf - rng.integers(0, 15, n)
    return raw.astype(str).replace("nan", "")


class WindConvention(unittest.TestCase):
    def test_from_west_blows_east(self):
        u, v = wind_components([270], [10])
        self.assertAlmostEqual(u[0], 10.0)
        self.assertAlmostEqual(v[0], 0.0, places=9)

    def test_from_north_and_south(self):
        u, v = wind_components([360, 180, 90], [10, 10, 10])
        np.testing.assert_allclose(v[:2], [-10, 10], atol=1e-9)
        self.assertLess(u[2], 0)  # FROM the east blows toward the west

    def test_calm_and_variable_have_no_vector(self):
        u, v = wind_components([0, np.nan], [0, 4])
        np.testing.assert_array_equal(np.r_[u, v], np.zeros(4))


class SkyAndCeiling(unittest.TestCase):
    def test_cover_fraction_and_ceiling(self):
        raw = synthetic_raw(specials=False).head(4).copy()
        raw[["skyc1", "skyc2", "skyc3"]] = [["FEW", "SCT", "BKN"], ["CLR", "", ""], ["VV", "", ""], ["SCT", "OVC", ""]]
        raw[["skyl1", "skyl2", "skyl3"]] = [["1500", "4000", "8000"], ["", "", ""], ["300", "", ""], ["2000", "15000", ""]]
        frame = clean(raw)
        np.testing.assert_allclose(frame.sky, [0.75, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(frame.ceiling_kft, [8.0, 12.0, 0.3, 12.0])


class Causality(unittest.TestCase):
    def setUp(self):
        frame = clean(synthetic_raw())
        self.reports = {s: frame for s in STATIONS}
        self.dates = pd.date_range("2024-03-09", "2024-03-11")

    def test_latest_is_strictly_before_cutoff_minus_allowance(self):
        times = pd.DatetimeIndex(["2024-01-02 13:49", "2024-01-02 13:50", "2024-01-02 13:51"], tz="UTC")
        frame = pd.DataFrame({"tmpf": [1.0, 2.0, 3.0]}, index=times)
        cutoff = pd.DatetimeIndex(["2024-01-02 14:00"], tz="UTC")
        value, when = latest(frame, "tmpf", cutoff, pd.Timedelta(minutes=10))
        self.assertEqual(value[0], 1.0)  # 13:50 is exactly cutoff - 10 min and is excluded
        self.assertEqual(when[0], times[0])
        value, _ = latest(frame, "tmpf", cutoff, pd.Timedelta(minutes=5))
        self.assertEqual(value[0], 3.0)

    def test_stale_reports_are_dropped(self):
        times = pd.DatetimeIndex(["2024-01-02 12:29", "2024-01-02 12:30"], tz="UTC")
        cutoff = pd.DatetimeIndex(["2024-01-02 14:00"], tz="UTC")
        value, _ = latest(pd.DataFrame({"tmpf": [1.0, np.nan]}, index=times), "tmpf", cutoff, pd.Timedelta(minutes=10))
        self.assertTrue(np.isnan(value[0]))  # 91 minutes old
        value, _ = latest(pd.DataFrame({"tmpf": [np.nan, 2.0]}, index=times), "tmpf", cutoff, pd.Timedelta(minutes=10))
        self.assertEqual(value[0], 2.0)  # exactly 90 minutes old

    def test_reports_at_or_after_the_limit_cannot_change_features(self):
        for hour in (9, 12, 15):
            for allowance in ALLOWANCES:
                full = build_rows(self.reports, self.dates, hour, allowance)
                for i, day in enumerate(self.dates):
                    limit = full.cutoff_utc.iloc[i] - pd.Timedelta(minutes=allowance)
                    corrupted = {}
                    for s, frame in self.reports.items():
                        later = frame.index >= limit
                        frame = frame.copy()
                        numeric = frame.select_dtypes("number").columns
                        frame.loc[later, numeric] = frame.loc[later, numeric] + 1000.0
                        corrupted[s] = frame
                    again = build_rows(corrupted, [day], hour, allowance)
                    pd.testing.assert_frame_equal(full.iloc[[i]].reset_index(drop=True), again, check_dtype=False)

    def test_precipitation_window(self):
        raw = synthetic_raw(specials=False)
        raw["p01i"] = "0.01"
        frame = clean(raw)
        cutoff = pd.DatetimeIndex(["2024-03-10 14:00"], tz="UTC")
        total = asos.precipitation(frame, cutoff, pd.Timedelta(minutes=10))
        self.assertAlmostEqual(total[0], 0.03)  # routine reports at 10:51, 11:51, 12:51 UTC


class DaylightSaving(unittest.TestCase):
    def test_cutoffs_follow_new_york_clock(self):
        frame = clean(synthetic_raw("2024-03-08", "2024-11-04", specials=False))
        reports = {s: frame for s in STATIONS}
        days = pd.DatetimeIndex(["2024-03-09", "2024-03-10", "2024-11-02", "2024-11-03"])
        rows = build_rows(reports, days, 9, 10)
        expected = pd.DatetimeIndex(["2024-03-09 14:00", "2024-03-10 13:00", "2024-11-02 13:00", "2024-11-03 14:00"],
                                    tz="UTC")
        self.assertTrue((pd.DatetimeIndex(rows.cutoff_utc) == expected).all())
        # The 08:51 local routine report is excluded at 10 minutes and used at 5, on both sides of each change.
        used10 = pd.DatetimeIndex(rows.LGA_report_utc)
        used5 = pd.DatetimeIndex(build_rows(reports, days, 9, 5).LGA_report_utc)
        np.testing.assert_array_equal((expected - used10).total_seconds() / 60, 69)
        np.testing.assert_array_equal((expected - used5).total_seconds() / 60, 9)


@unittest.skipUnless(FEATURES.exists(), "run asos.py build first")
class StoredFeatures(unittest.TestCase):
    def test_every_stored_lga_report_respects_the_timing_rule(self):
        table = pd.read_csv(FEATURES)
        cutoff = pd.to_datetime(table.cutoff_utc)
        used = pd.to_datetime(table.LGA_report_utc)
        have = used.notna()
        limit = cutoff - pd.to_timedelta(table.allowance_min, unit="min")
        self.assertTrue((used[have] < limit[have]).all())
        self.assertTrue((used[have] >= cutoff[have] - pd.Timedelta(minutes=90)).all())
        self.assertGreater(have.mean(), 0.97)
        local = pd.to_datetime(table.cutoff_utc).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        self.assertTrue((local.dt.hour == table.hour).all() and (local.dt.minute == 0).all())
        self.assertTrue((local.dt.strftime("%Y-%m-%d") == table.date).all())


@unittest.skipUnless((ROOT / "results" / "observed_weather" / "test_scores.csv").exists(), "run observed_weather.py")
class ExperimentOutputs(unittest.TestCase):
    def test_same_test_days_and_current_reproduced(self):
        scores = pd.read_csv(ROOT / "results" / "observed_weather" / "test_scores.csv")
        self.assertTrue((scores.n == 359).all())
        history = pd.read_csv(ROOT / "results" / "history" / "test_scores.csv")
        reference = history.loc[history.system.eq("Censored Gaussian, 2021-2023")].set_index("hour").log_loss
        current = scores.loc[scores.system.eq("CURRENT")].set_index("hour").log_loss
        np.testing.assert_allclose(current.sort_index(), reference.sort_index(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
