from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/"rolling"))
from data import issue_index, station_inputs
from rolling_models import BASE, Candidate, available, key, predict_route
from study import choose, spec, splits


class RollingTests(unittest.TestCase):
    def test_issue_clock_stays_local_across_dst(self):
        days = pd.DatetimeIndex(["2024-03-09","2024-03-10","2024-11-02","2024-11-03"])
        self.assertEqual(list(issue_index(days,9).tz_convert("UTC").hour),[14,13,13,14])

    def test_delay_applies_to_temperature_trend_and_floor(self):
        days = pd.DatetimeIndex(["2024-05-06"])
        times = pd.DatetimeIndex(["2024-05-06 07:50","2024-05-06 08:51","2024-05-06 09:01"],
                                 tz="America/New_York")
        values = pd.DataFrame({c:[50.,65.,90.] for c in
                              ["temp","rh","wind","spread","east","north","sky","vis"]},index=times)
        timely = station_inputs(values,days,9,0).iloc[0]
        delayed = station_inputs(values,days,9,15).iloc[0]
        self.assertEqual(timely.temp,65)
        self.assertEqual(delayed.temp,50)
        self.assertEqual(delayed.max_so_far,50)
        self.assertEqual(timely.max_so_far,65)

    def test_missing_latest_temperature_preserves_an_observed_trend(self):
        days = pd.DatetimeIndex(["2024-03-12"])
        times = pd.DatetimeIndex(["2024-03-12 07:45","2024-03-12 08:45","2024-03-12 08:59"],
                                 tz="America/New_York")
        obs = pd.DataFrame({c:[50.,53.,np.nan] for c in
                           ["temp","rh","wind","spread","east","north","sky","vis"]},index=times)
        row = station_inputs(obs,days,9,0).iloc[0]
        self.assertTrue(pd.isna(row.temp))
        self.assertEqual(row.trend_1h,3)

    def test_fold_boundaries_and_selection_tie(self):
        data = pd.DataFrame(index=pd.date_range("2022-01-01","2026-01-01"))
        for year,train,val in splits(data):
            self.assertLess(train.index.max(),val.index.min())
            self.assertEqual(val.index[0].year,int(year))
            self.assertLess(val.index.max(),pd.Timestamp("2025-09-01"))
        simple = {**spec("base","station",2018,["single"]),"loss":.201}
        rich = {**spec("base","sat",2021,["blend","second"]),"loss":.2001}
        self.assertEqual(choose([rich,simple]),simple)
        self.assertEqual(choose([rich,{**simple,"loss":.202}]),rich)

    def test_missing_training_column_and_full_source_fallback(self):
        dates = pd.date_range("2021-04-01",periods=35)
        df = pd.DataFrame(0.,columns=BASE,index=dates)
        df["target_category"] = np.arange(len(df))%5
        df["target_max"] = 30+15*df.target_category
        df["LGA_temp"] = df.target_max-5
        df["LGA_max_so_far"] = df.target_max-8
        df["JFK_rh"] = np.nan
        fallback = Candidate(key("threshold",C=1),"base","station").fit(df)
        self.assertNotIn("JFK_rh",fallback.used)
        df["latest_high"],df["latest_warm"] = np.nan,np.nan
        self.assertFalse(available(df,"latest").any())
        p,routed = predict_route(df,dict(spec={"tier":"latest"},members=[],fallback=[fallback]))
        self.assertFalse(routed.any())
        np.testing.assert_allclose(p.sum(axis=1),1)
        self.assertTrue((p>=0).all())
        np.testing.assert_array_equal(p[df.LGA_max_so_far>=70,:3],0)

if __name__=="__main__":
    unittest.main()
