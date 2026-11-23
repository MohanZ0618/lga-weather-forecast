"""Regression tests for source timestamps, features and experiment boundaries."""

import unittest

import numpy as np
import pandas as pd

from experiment import BOUNDS, split_dates, stabilize
from hrrr import forecast_features
from prepare import lcd_local_time, local_cutoff, station_features


class ForecastTests(unittest.TestCase):
    def test_lcd_year_round_standard_time(self):
        converted = lcd_local_time(pd.Series(["2026-01-01T00:51:00", "2026-08-01T00:51:00"]))
        self.assertEqual(converted.dt.tz_convert("UTC").dt.hour.tolist(), [5, 5])
        self.assertEqual(converted.dt.hour.tolist(), [0, 1])

    def test_boundary_values(self):
        np.testing.assert_array_equal(
            np.searchsorted(BOUNDS, [39.99, 40, 55, 70, 85], side="right"), np.arange(5)
        )

    def test_daylight_saving_and_future_valid_times(self):
        for day, expected_utc_hour, expected_max in (("2026-03-08", 13, 16), ("2025-11-02", 14, 17)):
            day = pd.Timestamp(day)
            self.assertEqual(local_cutoff(day, 9).tz_convert("UTC").hour, expected_utc_hour)
            cutoff = local_cutoff(day, 9).tz_convert("UTC")
            temperatures = np.arange(18, dtype=float)
            maximum, warming = forecast_features(day, 9, cutoff, temperatures)
            self.assertEqual(maximum, expected_max)
            self.assertEqual(warming, expected_max - (expected_utc_hour - 11))
            with self.assertRaises(ValueError):
                forecast_features(day, 9, cutoff + pd.Timedelta(seconds=1), temperatures)

    def test_after_cutoff_readings_do_not_leak(self):
        day = pd.Timestamp("2026-05-01")
        timestamps = pd.to_datetime(["2026-05-01 11:51Z", "2026-05-01 12:51Z",
                                     "2026-05-01 13:00Z", "2026-05-01 13:51Z"], utc=True)
        table = pd.DataFrame({"temp": [50, 55, 56, 99], "rh": 50, "wind": 2},
                             index=timestamps.tz_convert("America/New_York"))
        maxima = {"LGA": {}}
        full = station_features({"LGA": table}, maxima, day, 9)
        limited = station_features({"LGA": table.iloc[:-1]}, maxima, day, 9)
        self.assertEqual(full["LGA_temp"], 56)
        self.assertEqual(full["LGA_max_so_far"], 55)
        self.assertEqual(full["LGA_temp"], limited["LGA_temp"])
        self.assertAlmostEqual(full["LGA_trend_1h"], limited["LGA_trend_1h"])
        self.assertAlmostEqual(full["LGA_trend_1h"], 6 / 1.15)

    def test_stabilization_normalizes_and_respects_observed_lower_bound(self):
        prior = np.full((3, 5), 0.2)
        result = stabilize(np.eye(5)[[0, 1, 2]], prior, np.array([70.0, 85.0, np.nan]))
        np.testing.assert_allclose(result.sum(axis=1), 1)
        np.testing.assert_allclose(result[0, :3], 0)
        np.testing.assert_allclose(result[1], [0, 0, 0, 0, 1])
        self.assertTrue((result[2] > 0).all())

    def test_splits_are_chronological_and_disjoint(self):
        index = pd.to_datetime(["2023-12-31", "2024-01-01", "2025-01-01", "2025-08-31",
                               "2025-09-01", "2026-08-31", "2026-09-01"])
        splits = split_dates(index)
        self.assertEqual([int(mask.sum()) for mask in splits.values()], [1, 1, 2, 2])
        np.testing.assert_array_equal(np.sum(list(splits.values()), axis=0), [1, 1, 1, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
