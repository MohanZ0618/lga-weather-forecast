from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/"rolling"))
from market_data import bounds, metar_tenths, partition, regime, whole_f
from market_study import DEGREES, degree_mass, market_days, membership


class MarketTests(unittest.TestCase):
    def test_integer_rounding_including_negative_tie(self):
        np.testing.assert_array_equal(whole_f([69.98,70.,69.49,69.50,-8.5,-8.51]),[70,70,69,70,-8,-9])

    def test_remark_precedes_body_and_dewpoint_never_sets_temperature(self):
        self.assertEqual(metar_tenths("KLGA 012251Z 18/12 RMK AO2 T01780120"),178)
        self.assertEqual(metar_tenths("METAR KLGA 012251Z M11/M17 RMK AO2"),-110)
        self.assertEqual(metar_tenths("KLGA 012251Z M11/M17 RMK T10171022"),-17)
        self.assertTrue(np.isnan(metar_tenths("KLGA 012251Z ///// RMK AO2")))

    def test_rule_priority_with_alternate_source(self):
        self.assertEqual(regime("KLGA hourly temperature remark routine SPECI; Weather Underground fallback"),"routine_metar")
        self.assertEqual(regime("KLGA recorded by NOAA weather.gov/wrh/timeseries; Wunderground fallback"),"noaa_timeseries")

    def test_partition_handles_open_single_and_pair_bins(self):
        titles = ["68°F or below","69°F","70–71°F","72°F or higher"]
        items = [dict(low=bounds(t)[0],high=bounds(t)[1]) for t in titles]
        self.assertEqual(partition(items[::-1]),items)
        with self.assertRaises(ValueError):
            partition(items[:1]+items[2:])
        frame = pd.DataFrame(items)
        mask = membership(frame.low.to_numpy(),frame.high.to_numpy())
        p = degree_mass(np.array([70.]),1.,np.array([69.]),True)[0]
        bins = mask @ p
        self.assertAlmostEqual(bins[2],p[(DEGREES==70)|(DEGREES==71)].sum())
        self.assertAlmostEqual(bins.sum(),1.)

    def test_unlisted_bracket_and_known_winner_are_excluded(self):
        days = pd.date_range("2025-10-01",periods=3)
        rows = []
        for n,day in enumerate(days):
            at = (day+pd.Timedelta(hours=9)).tz_localize("America/New_York")
            for i,title in enumerate(["69°F or below","70°F or higher"]):
                lo,hi = bounds(title)
                rows.append(dict(date=day,low=lo,high=hi,bracket=title,winner=i==0,slug="x",regime="wunderground",
                    listed=at+pd.Timedelta(minutes=1 if n==1 and i==1 else -5),
                    closed=at+pd.Timedelta(hours=-1 if n==2 else 24)))
        valid = market_days(pd.DataFrame(rows),9,pd.DataFrame(index=days))
        self.assertEqual(valid.index.tolist(),[days[0]])


if __name__=="__main__":
    unittest.main()
