from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/"bracket_study"))
from estimators import BASE, features
from panels import FREEZE, TZ
from market_study import DEGREES
from morning_data import market_layouts
from old09 import DEFAULT, Adjustment, adjust, digit_weights, validation_truth
from time_study import TimedDistribution, common_dates, summarize


class OldModelTests(unittest.TestCase):
    def test_blending_bias_and_available_floor(self):
        n = len(DEGREES)
        ix = int(60-DEGREES[0])
        first,second = np.zeros((1,n)),np.zeros((1,n))
        first[0,ix] = 1
        second[0,ix+2] = 1
        inputs = (first,second,np.array([59.]),np.array([62.]),np.ones((1,n)))
        np.testing.assert_allclose(adjust(inputs,DEFAULT),(first+second)/2)
        shifted = adjust(inputs,Adjustment(.75,-.5))[0]
        np.testing.assert_allclose(shifted[ix-1:ix+3],[.375,.375,.125,.125])
        np.testing.assert_allclose(adjust(inputs,Adjustment(.5,0.,0.,True)),second)
        np.testing.assert_equal(first[0,ix],1.)

    def test_pre_freeze_digit_weights_ignore_later_readings(self):
        ix = pd.date_range("2024-01-01",FREEZE)
        y = pd.DataFrame(dict(hours=24,blocks=4,all_f=60),index=ix)
        expected = digit_weights(y,"all_f")
        y.loc[ix>=pd.Timestamp(FREEZE)-pd.Timedelta(days=7),"all_f"] = 20
        np.testing.assert_array_equal(expected,digit_weights(y,"all_f"))

    def test_parameters_need_settlement_and_reporting_buffer(self):
        ix = pd.date_range("2026-02-20",periods=3)
        g = pd.DataFrame(dict(date=ix,winner=True,market_id=list("abc"),closed=pd.to_datetime(
            ["2026-02-24T12:00Z","2026-02-27T05:00Z","2026-03-03T12:00Z"])))
        self.assertEqual(validation_truth(g,"2026-03-01").index.tolist(),[ix[0]])

    def test_later_listing_only_available_for_later_forecast(self):
        day = pd.Timestamp("2026-03-10")
        noon = (day+pd.Timedelta(hours=12)).tz_localize(TZ)
        g = pd.DataFrame(dict(date=day,market_id=["a","b","c"],bracket=["low","middle","high"],
            low=[np.nan,60,62],high=[59,61,np.nan],regime="test",listed=noon,closed=noon+pd.Timedelta(days=1)))
        self.assertFalse(market_layouts(g,9))
        self.assertEqual(market_layouts(g,12)[day]["ids"],["a","b","c"])

    def test_hour_indicators_do_not_change_weather_or_floor(self):
        g = pd.DataFrame(1.,index=range(3),columns=BASE).assign(floor_f=60.,issue_hour=[9,12,15])
        timed = TimedDistribution("threshold","station",{})
        x,anchor,floor = timed.design(g)
        base,a,f = features(g,"station")
        np.testing.assert_array_equal(x[:,:-2],base)
        np.testing.assert_array_equal(x[:,-2:],[[0,0],[1,0],[0,1]])
        np.testing.assert_array_equal(anchor,a)
        np.testing.assert_array_equal(floor,f)

    def test_overall_counts_only_matched_dates_with_equal_hour_weights(self):
        days = pd.date_range("2026-03-01",periods=2)
        df = pd.DataFrame(dict(date=[days[0]]*3+[days[1]]*2,hour=[9,12,15,9,15],
                               correct=[0,0,1,1,1],log_loss=1.,model="a"))
        common = common_dates(df)
        self.assertEqual(common.tolist(),[days[0]])
        metrics = summarize(df.loc[df.date.isin(common)]).set_index("scope")
        self.assertEqual(metrics.loc["overall","dates"],1)
        self.assertEqual(metrics.loc["overall","forecasts"],3)
        self.assertAlmostEqual(metrics.loc["overall","accuracy"],1/3)
