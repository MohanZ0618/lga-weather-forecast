"""Checks for the long-record MOS features (mos.py) and the two-stage experiment (mos_history.py)."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import mos
import mos_history
from prepare import DATA, local_cutoff

UTC = "UTC"


def fake_run(run, temp, first=6, steps=12, max_value=None):
    """3-hourly projections from `first` hours after initialization, n_x at every 00 UTC projection."""
    run = pd.Timestamp(run, tz=UTC)
    ftime = pd.date_range(run + pd.Timedelta(hours=first), periods=steps, freq="3h")
    temps = np.full(steps, float(temp)) if np.isscalar(temp) else np.asarray(temp, float)
    n_x = np.where(ftime.hour == 0, np.nan if max_value is None else max_value, np.nan)
    return run, {"ftime": ftime.tz_localize(None).to_numpy("datetime64[ns]"), "tmp": temps, "n_x": n_x,
                 "wdr": np.full(steps, 270.0), "wsp": np.full(steps, 10.0), "cld": np.full(steps, 0.44)}


def product(*runs):
    table = dict(runs)
    return {"GFS": (table, pd.DatetimeIndex(sorted(table)))}


def day_runs(day, with_noon_max=False):
    """00, 06 and 12 UTC runs of one day with distinct temperatures (50, 60, 70) and daytime maxima.

    Like the real MAV, the 12 UTC run no longer carries today's daytime max (unless with_noon_max).
    """
    return product(fake_run(f"{day} 00:00", 50, max_value=81.0),
                   fake_run(f"{day} 06:00", 60, max_value=82.0),
                   fake_run(f"{day} 12:00", 70, max_value=83.0 if with_noon_max else None, first=6))


class Release(unittest.TestCase):
    def test_run_counts_only_from_its_assumed_release(self):
        starts = pd.DatetimeIndex([pd.Timestamp("2024-07-01 06:00", tz=UTC), pd.Timestamp("2024-07-01 12:00", tz=UTC)])
        oldest = pd.Timestamp("2024-06-30", tz=UTC)
        newest = lambda cutoff: next(mos.released(starts, pd.Timestamp(cutoff, tz=UTC), mos.DELAY["GFS"], oldest))
        self.assertEqual(newest("2024-07-01 16:59"), starts[0])  # the 12 UTC run is released at 17:00 UTC
        self.assertEqual(newest("2024-07-01 17:00"), starts[1])
        mex = lambda cutoff: next(mos.released(starts, pd.Timestamp(cutoff, tz=UTC), mos.DELAY["MEX"], oldest))
        self.assertEqual(mex("2024-07-01 17:30"), starts[0])  # MEX waits 6 h
        self.assertEqual(mex("2024-07-01 18:00"), starts[1])

    def test_noon_cutoff_follows_daylight_saving(self):
        # 12:00 New York is 17:00 UTC in standard time (the 12 UTC run is out) and 16:00 UTC in daylight time.
        winter = mos.day_features(day_runs("2024-03-08"), "2024-03-08", 12)
        summer = mos.day_features(day_runs("2024-03-11"), "2024-03-11", 12)
        self.assertEqual(winter["gfs_rest_run"], pd.Timestamp("2024-03-08 12:00"))
        self.assertEqual(winter["gfs_remaining_max"], 70)
        self.assertEqual(summer["gfs_rest_run"], pd.Timestamp("2024-03-11 06:00"))
        self.assertEqual(summer["gfs_remaining_max"], 60)
        # The 12 UTC run has no projection at or before 17:00 UTC, so `now` comes from the 06 UTC run.
        self.assertEqual(winter["gfs_now_run"], pd.Timestamp("2024-03-08 06:00"))
        self.assertEqual(winter["gfs_now"], 60)

    def test_daytime_max_falls_back_to_the_newest_run_that_still_forecasts_it(self):
        row = mos.day_features(day_runs("2024-07-01"), "2024-07-01", 15)
        self.assertEqual(row["gfs_max_run"], pd.Timestamp("2024-07-01 06:00"))
        self.assertEqual(row["gfs_max"], 82.0)
        row = mos.day_features(day_runs("2024-07-01", with_noon_max=True), "2024-07-01", 15)
        self.assertEqual((row["gfs_max_run"], row["gfs_max"]), (pd.Timestamp("2024-07-01 12:00"), 83.0))
        early = mos.day_features(day_runs("2024-07-01"), "2024-07-01", 9)  # 13:00 UTC: 06 UTC run is newest
        self.assertEqual(early["gfs_max_run"], pd.Timestamp("2024-07-01 06:00"))

    def test_remaining_max_stops_at_local_midnight_and_starts_at_the_cutoff(self):
        # 06 UTC run, projections 12 UTC to 21 UTC next day. Cutoff 15:00 EDT = 19 UTC, local midnight 04 UTC.
        temps = [90, 95, 70, 71, 72, 73, 99, 99, 99, 99, 99, 99]  # 12Z 15Z | 18Z 21Z 00Z 03Z | 06Z ...
        runs = product(fake_run("2024-07-01 06:00", temps))
        row = mos.day_features(runs, "2024-07-01", 15)
        self.assertEqual(row["gfs_remaining_max"], 73)  # 15Z (before) and 06Z (after midnight) are excluded
        self.assertAlmostEqual(row["gfs_now"], 70 + (71 - 70) / 3)  # 19Z, between 18Z and 21Z
        # Standard time: midnight is 05 UTC, the same projections qualify.
        winter = mos.day_features(product(fake_run("2024-01-15 06:00", temps)), "2024-01-15", 15)
        self.assertEqual(winter["gfs_remaining_max"], 73)
        self.assertAlmostEqual(winter["gfs_now"], 70 + 2 * (71 - 70) / 3)  # 20Z


class BundledTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runs = [c for p in ("gfs", "nam", "mex") for c in (f"{p}_max_run", f"{p}_rest_run", f"{p}_now_run")]
        table = pd.read_csv(DATA / "mos_features.csv.gz", parse_dates=["date"])
        cls.runs = [c for c in runs if c in table]
        for column in cls.runs:
            table[column] = pd.to_datetime(table[column], format="ISO8601").dt.tz_localize(UTC)
        cutoffs = {(d, h): local_cutoff(d, h).tz_convert(UTC) for d, h in set(zip(table.date, table.hour))}
        table["cutoff"] = [cutoffs[key] for key in zip(table.date, table.hour)]
        cls.table = table

    def test_no_feature_uses_a_run_before_its_assumed_release(self):
        for column in self.runs:
            delay = mos.DELAY[column.split("_")[0].upper()]
            used = self.table[column].notna()
            self.assertGreater(used.sum(), 0, column)
            self.assertFalse((self.table.loc[used, column] + delay > self.table.loc[used, "cutoff"]).any(), column)

    def test_noon_run_choice_follows_daylight_saving(self):
        noon = self.table.loc[self.table.hour.eq(12) & self.table.gfs_rest_run.notna()]
        daylight = noon.cutoff.dt.hour.eq(16)
        self.assertFalse(noon.loc[daylight, "gfs_rest_run"].dt.hour.eq(12).any())
        self.assertGreater(noon.loc[~daylight, "gfs_rest_run"].dt.hour.eq(12).mean(), 0.9)

    def test_daytime_max_is_today(self):
        # A daytime max may come from yesterday's runs but never from a run after the cutoff.
        used = self.table.gfs_max_run.notna()
        age = (self.table.cutoff - self.table.gfs_max_run)[used] / pd.Timedelta(hours=1)
        self.assertTrue(age.between(5, 36).all())


class FakeProblem:
    """Records the training years of every fit; mu identifies which fit produced a value."""

    def __init__(self, years):
        self.samples = pd.DataFrame(index=pd.to_datetime([f"{y}-06-01" for y in years]))
        self.calls = []

    def fit(self, columns, train, grid):
        self.calls.append(set(self.samples.index.year[train]))
        n = len(self.samples)
        return {"mu": np.full(n, float(len(self.calls) - 1)), "log_sigma": np.zeros(n)}


class TwoStage(unittest.TestCase):
    def test_out_of_fold_stage_one_never_saw_the_row_year(self):
        years = np.array([2010, 2015, 2021, 2021, 2022, 2023, 2024, 2025])
        problem = FakeProblem(years)
        window = years <= 2023
        stage2 = (years >= 2021) & (years <= 2023)
        mu, _, held = mos_history.out_of_fold(problem, [], window, stage2, 1.0)
        self.assertEqual(held, [2021, 2022, 2023])
        for row, year in enumerate(years):
            source = problem.calls[int(mu[row])]
            if stage2[row]:
                self.assertNotIn(year, source)
            else:
                self.assertEqual(int(mu[row]), 0)  # validation and test rows use the full-window fit
            self.assertFalse(source & {2024, 2025})

    def test_relative_mos_features(self):
        samples = pd.DataFrame({"LGA_max_so_far": [60.0], "LGA_temp": [58.0]}, index=pd.to_datetime(["2024-07-01"]))
        mos_table = pd.DataFrame({"date": pd.to_datetime(["2024-07-01"]), "hour": [12], "gfs_max": [70.0],
                                  "gfs_remaining_max": [68.0], "gfs_now": [57.0], "nam_max": [71.0],
                                  "nam_remaining_max": [69.0], "nam_now": [59.0], "mex_max": [72.0],
                                  "gfs_rest_run": ["2024-07-01 06:00"]})
        out = mos_history.add_mos(samples, mos_table, 12).iloc[0]
        self.assertEqual((out.rel_gfs, out.rel_gfs_rest, out.gfs_now_error), (10.0, 8.0, 1.0))
        self.assertEqual((out.rel_nam, out.nam_now_error, out.rel_mex), (11.0, -1.0, 12.0))
        self.assertNotIn("gfs_rest_run", out.index)


class Results(unittest.TestCase):
    def test_every_system_is_scored_on_the_same_359_days(self):
        path = Path(__file__).resolve().parent / "results" / "mos_history" / "test_scores.csv"
        if not path.exists():
            self.skipTest("run mos_history.py first")
        scores = pd.read_csv(path)
        self.assertTrue(scores.n.eq(359).all())
        selection = pd.read_csv(path.with_name("selection.csv"))
        self.assertLess(selection.current_refit_max_abs_prob_diff_vs_saved.max(), 0.01)


if __name__ == "__main__":
    unittest.main()
