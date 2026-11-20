import unittest
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from weather import (
    HOURS, TZ, daily_labels, five_probabilities, hourly_rows, local_cutoff,
    make_features, previous_day_maxima, read_noaa, validate_noaa_csv,
)


def observations(times, temperatures):
    index = pd.DatetimeIndex(times, tz=TZ, name="timestamp")
    return pd.DataFrame({"temp": temperatures, "rh": 50.0, "wind": 2.0}, index=index)


def lga_features(table, day="2025-01-03", hour=9):
    return make_features({"LGA": table}, {"LGA": previous_day_maxima(table)}, ("LGA",), day, hour)


class WeatherTests(unittest.TestCase):
    def test_future_reports_do_not_change_features(self):
        table = observations(
            ["2025-01-02 23:55", "2025-01-03 05:55", "2025-01-03 06:03",
             "2025-01-03 08:55", "2025-01-03 09:00"],
            [40.0, 43.0, 65.0, 50.0, 55.0],
        )
        future = observations(["2025-01-03 09:02", "2025-01-03 12:51"], [99.0, 110.0])
        features, times = lga_features(table)
        self.assertEqual(lga_features(pd.concat([table, future])), (features, times))
        self.assertEqual(times["LGA"], local_cutoff("2025-01-03", 9))
        self.assertEqual(features["LGA_temp"], 55.0)
        self.assertEqual(features["LGA_max_so_far"], 65.0)
        self.assertEqual(features["LGA_change_3h"], 12.0)
        self.assertEqual(features["LGA_previous_max"], 40.0)

    def test_cutoffs_keep_wall_clock_time_on_dst_transitions(self):
        for day, utc_offset in [("2025-03-09", -4), ("2025-11-02", -5)]:
            for hour in HOURS:
                with self.subTest(day=day, hour=hour):
                    cutoff = local_cutoff(day, hour)
                    self.assertEqual(cutoff.strftime("%Y-%m-%d %H:%M"), f"{day} {hour:02d}:00")
                    self.assertEqual(cutoff.utcoffset(), pd.Timedelta(hours=utc_offset))

    def test_labels_require_coverage_and_use_the_specified_boundaries(self):
        hours = pd.date_range("2025-01-01", periods=24 * 7, freq="h", tz=TZ)
        temperatures = np.repeat([39.9, 40, 55, 70, 85, 60, 65], 24).astype(float)
        temperatures[0:4] = np.nan     # 20 valid hours with every period represented.
        temperatures[120:125] = np.nan  # 19 valid hours.
        temperatures[-6:] = np.nan    # No valid observations from 18:00 through 23:59.
        table = pd.DataFrame({"temp": temperatures}, index=hours)
        labels = daily_labels(table)
        self.assertEqual(labels.index.strftime("%Y-%m-%d").tolist(), [
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05",
        ])
        self.assertEqual(labels["target_category"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(labels["valid_hours"].iloc[0], 20)

    def test_stale_observations_leave_latest_inputs_missing(self):
        table = observations(["2025-01-03 07:59"], [42.0])
        features, times = lga_features(table)
        self.assertIsNone(times["LGA"])
        for name in ("temp", "rh", "wind", "change_3h", "previous_max"):
            self.assertTrue(np.isnan(features[f"LGA_{name}"]), name)
        self.assertEqual(features["LGA_max_so_far"], 42.0)
        boundary = observations(["2025-01-03 08:00"], [42.0])
        self.assertEqual(lga_features(boundary)[0]["LGA_temp"], 42.0)

    def test_probabilities_expand_to_five_categories(self):
        x = pd.DataFrame({"month": [1, 2, 3, 4]})
        model = DummyClassifier(strategy="prior").fit(x, [1, 1, 1, 4])
        probabilities = five_probabilities(model, x)
        self.assertEqual(probabilities.shape, (4, 5))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[0], [0, 0.75, 0, 0, 0.25])

    def test_noaa_quality_codes_units_and_hourly_timestamps(self):
        raw = '''DATE,TMP,DEW,WND,REPORT_TYPE
2025-01-03T13:05:00,"+0080,1","+0050,1","090,1,N,0015,1",FM-15
2025-01-03T13:55:00,"+0100,1","+0100,4","270,1,N,0020,5", FM-15
2025-01-03T13:59:00,"+0200,1","+0100,1","270,1,N,0030,1",FM-12
2025-01-03T14:02:00,"-9999,1","+0100,1","270,1,N,0020,2",FM-16
'''
        routine, count = read_noaa(StringIO(raw))
        hourly = hourly_rows(routine)
        self.assertEqual(count, 3)
        self.assertEqual(hourly.index.strftime("%H:%M").tolist(), ["08:55", "09:02"])
        np.testing.assert_allclose(hourly.iloc[0].to_numpy(), [50.0, 100.0, 2.0])
        self.assertTrue(hourly.iloc[1].isna().all())
        with self.assertRaises(ValueError):
            validate_noaa_csv(b"<html>redirect</html>", "test")


if __name__ == "__main__":
    unittest.main()
