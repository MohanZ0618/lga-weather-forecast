"""Checks for the station-guidance features and the distributional models."""

import unittest

import numpy as np
import pandas as pd

import distribution
import guidance
from prepare import DATA, local_cutoff


def fake_run(run, temps, cloud="SC", direction=270, speed=10):
    times = pd.date_range(run + pd.Timedelta(hours=1), periods=len(temps), freq="h")
    return pd.DataFrame({"runtime": run, "ftime": times, "tmp": temps, "cld": cloud,
                         "wdr": direction, "wsp": speed})


class GuidanceTiming(unittest.TestCase):
    def test_new_york_cutoffs_follow_daylight_saving(self):
        self.assertEqual(local_cutoff(pd.Timestamp("2024-07-01"), 9).tz_convert("UTC").hour, 13)
        self.assertEqual(local_cutoff(pd.Timestamp("2024-01-15"), 9).tz_convert("UTC").hour, 14)

    def test_run_is_not_used_before_its_assumed_release(self):
        first, second = pd.Timestamp("2024-07-01 06:00", tz="UTC"), pd.Timestamp("2024-07-01 12:00", tz="UTC")
        runs = {first: fake_run(first, [60] * 30), second: fake_run(second, [70] * 30)}
        starts = pd.DatetimeIndex(sorted(runs))
        midnight, evening = pd.Timestamp("2024-07-02 04:00", tz="UTC"), pd.Timestamp("2024-07-01 22:00", tz="UTC")
        oldest = pd.Timestamp("2024-07-01 00:00", tz="UTC")
        newest = lambda cutoff: next(guidance.newest_run(runs, starts, "LAV", cutoff, oldest))[0]
        self.assertEqual(newest(pd.Timestamp("2024-07-01 12:54", tz="UTC")), first)  # 12 UTC run counts from 12:55
        self.assertEqual(newest(pd.Timestamp("2024-07-01 12:55", tz="UTC")), second)
        early = guidance.lamp_features(runs, starts, pd.Timestamp("2024-07-01 12:00", tz="UTC"), midnight, evening)
        late = guidance.lamp_features(runs, starts, pd.Timestamp("2024-07-01 13:00", tz="UTC"), midnight, evening)
        self.assertEqual((early["lamp_run"], early["lamp_remaining_max"]), (first, 60))
        self.assertEqual((late["lamp_run"], late["lamp_remaining_max"]), (second, 70))

    def test_wind_components_use_the_from_direction(self):
        run = pd.Timestamp("2024-07-01 12:00", tz="UTC")
        cutoff, evening = pd.Timestamp("2024-07-01 13:00", tz="UTC"), pd.Timestamp("2024-07-01 22:00", tz="UTC")
        midnight = pd.Timestamp("2024-07-02 04:00", tz="UTC")
        for direction, u, v in ((270, 10, 0), (180, 0, 10), (90, -10, 0)):
            runs = {run: fake_run(run, [70] * 30, direction=direction)}
            row = guidance.lamp_features(runs, pd.DatetimeIndex([run]), cutoff, midnight, evening)
            self.assertAlmostEqual(row["lamp_u"], u, places=6)
            self.assertAlmostEqual(row["lamp_v"], v, places=6)

    def test_bundled_features_respect_release_delays(self):
        table = pd.read_csv(DATA / "guidance_features.csv.gz", parse_dates=["date", "lamp_run", "nbm_run"])
        cutoff = pd.Series([local_cutoff(d, h).tz_convert("UTC") for d, h in zip(table.date, table.hour)])
        self.assertFalse((table.lamp_run + pd.Timedelta(minutes=55) > cutoff).any())
        self.assertFalse((table.nbm_run + pd.Timedelta(minutes=66) > cutoff).any())


class DistributionMath(unittest.TestCase):
    def test_quantile_point_mass_at_zero(self):
        q = np.zeros((1, 9))
        cdf = distribution.quantile_cdf(q, np.array([[0.0, 0.5]]))
        np.testing.assert_allclose(cdf, [[0.0, 1.0]])

    def test_band_probabilities_sum_to_one_and_respect_floor(self):
        q = np.array([[0, 0, 1, 2, 3, 4, 6, 7, 9]], dtype=float)
        samples = pd.DataFrame({"LGA_max_so_far": [56.0]})
        bands = distribution.to_bands(distribution.quantile_cdf(q, distribution.band_cdf_edges(samples)))
        self.assertAlmostEqual(bands.sum(), 1.0)
        self.assertEqual(bands[0, :2].sum(), 0.0)  # bands below the observed 56 F are impossible

    def test_guidance_columns_only_in_guidance_scopes(self):
        frame = pd.DataFrame(columns=["season_sin", "season_cos", "LGA_temp", "LGA_trend_1h", "rel_now",
                                      "rel_hrrr", "hrrr_remaining_max", "hrrr_warming", *distribution.GUIDANCE])
        plain = distribution.relative_columns(frame, "Four stations + trends + HRRR")
        lamp = distribution.relative_columns(frame, "Four stations + trends + LAMP/NBM")
        self.assertFalse(set(distribution.GUIDANCE) & set(plain))
        self.assertTrue(set(distribution.GUIDANCE) <= set(lamp))
        self.assertNotIn("hrrr_remaining_max", lamp)
        self.assertIn("hrrr_remaining_max", plain)


if __name__ == "__main__":
    unittest.main()
