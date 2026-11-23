import gzip
import unittest

import numpy as np
import pandas as pd

from buoy_study import asof_features, clean_archive, crossing_to_bands


class BuoyStudyTests(unittest.TestCase):
    def test_ordered_bands_and_recorded_high(self):
        crossing = np.array([[0.8, 0.4, 0.5, 0.2], [0.5, 0.4, 0.3, 0.2]])
        prob = crossing_to_bands(crossing, np.array([55., np.nan]), np.full((2, 5), 0.2))
        np.testing.assert_allclose(prob.sum(axis=1), 1)
        np.testing.assert_array_equal(prob[0, :2], 0)
        self.assertTrue(np.all(prob >= 0))

    def test_ndbc_units_missing_codes_and_wind_direction(self):
        header = "#YY MM DD hh mm WDIR WSPD ATMP WTMP\n#yr mo dy hr mn degT m/s degC degC\n"
        rows = ("2024 01 02 13 50 90 4.0 0.0 10.0\n"
                "2024 01 02 14 00 999 99 999.0 999.0\n"
                "2024 01 02 14 10 999 0 MM MM\n")
        obs = clean_archive(gzip.compress((header + rows).encode()))
        self.assertEqual(str(obs.index.tz), "UTC")
        np.testing.assert_allclose(obs.iloc[0], [50, 32, -4, 0], atol=1e-12)
        self.assertTrue(obs.iloc[1].isna().all())
        self.assertEqual(obs.iloc[2].east_wind_ms, 0)

    def test_availability_age_and_dst(self):
        days = pd.DatetimeIndex(["2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12"])
        times = pd.to_datetime(["2024-03-09 13:30", "2024-03-09 13:31", "2024-03-10 12:30",
                                "2024-03-10 13:01", "2024-03-11 10:59", "2024-03-12 11:00"], utc=True)
        obs = pd.DataFrame({"sea_f": range(6)}, index=times.rename("observed_utc"))
        result = asof_features(days, 9, obs)
        np.testing.assert_allclose(result.sea_f, [0, 2, np.nan, 5], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
