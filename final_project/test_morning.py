from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/"bracket_study"))
from morning import BASE, forecast, month_inputs, stage_scores
from morning_data import market_layouts, observation_features
from panels import TZ
from threadpoolctl import threadpool_limits


def simple_contracts(date):
    day = pd.Timestamp(date)
    issue = (day+pd.Timedelta(hours=9)).tz_localize(TZ)
    return pd.DataFrame(dict(date=day,market_id=["a","b","c"],bracket=["<60","60–61",">61"],
        low=[np.nan,60,62],high=[59,61,np.nan],regime="test",listed=issue-pd.Timedelta(hours=1),
        closed=issue+pd.Timedelta(days=1)))


class MorningTests(unittest.TestCase):
    def test_layout_uses_only_listed_partition_and_ignores_winner(self):
        g = simple_contracts("2025-11-02")
        layout = market_layouts(g)
        day = g.date.iloc[0]
        self.assertEqual(layout[day]["center"],60.5)
        for winner in ([True,False,False],[False,False,True]):
            self.assertEqual(market_layouts(g.assign(winner=winner))[day]["ids"],layout[day]["ids"])
        g.loc[1,"listed"] = g.closed.iloc[0]
        self.assertFalse(market_layouts(g))

    def test_0856_and_later_observations_cannot_change_features(self):
        times = pd.DatetimeIndex(["2025-06-09 08:50","2025-06-09 08:56","2025-06-09 13:10"],tz=TZ)
        obs = pd.DataFrame({c:[50.,60.,70.] for c in ["temp","wind","rh","east","north","sky","spread","vis"]},
                           index=times)
        days = pd.DatetimeIndex(["2025-06-09"])
        before = observation_features(obs,days)
        obs.loc[times[1:],"temp"] = 123.
        pd.testing.assert_frame_equal(before,observation_features(obs,days))
        self.assertEqual(before.early_high.item(),50.)

    def test_unresolved_outcome_cannot_select_parameters(self):
        dates = pd.to_datetime(["2025-04-01","2025-04-02"])
        layouts = {d:market_layouts(simple_contracts(d))[d] for d in dates}
        states = {d:dict(ridge=(60.,2.),digit_weight=np.ones(9),floor=45.) for d in dates}
        truth = pd.DataFrame(dict(market_id=["b","a"],closed=pd.to_datetime(
            ["2025-04-04T10:00Z","2025-05-02T10:00Z"])),index=dates)
        def scores():
            return stage_scores("family",states,layouts,truth,[("ridge",BASE)],"2025-05-01")
        before = scores()
        truth.loc[dates[-1],"market_id"]="c"
        pd.testing.assert_frame_equal(before,scores())
        self.assertEqual(before.days.item(),1)

    def test_future_weather_labels_and_later_month_features_cannot_change_forecast(self):
        from morning import COLS
        rng = np.random.default_rng(111)
        index = pd.date_range("2021-01-01","2024-03-31")
        x = pd.DataFrame(rng.normal(size=(len(index),len(COLS))),columns=COLS,index=index)
        for col in ("latest_high","nbm_high","LGA_temp","LGA_early_high","LGA_previous_max"):
            x[col] = 55+rng.normal(0,3,len(x))
        obs = pd.DataFrame(dict(all_f=(x.nbm_high+rng.normal(0,2,len(x))).round(),hours=24,blocks=4))
        early,late = pd.to_datetime(["2024-03-01","2024-03-20"])
        def get():
            state = month_inputs(x,obs,[early,late],early,"all_f")[early]
            self.assertLess(state["train_end"],early-pd.Timedelta(days=7))
            self.assertEqual(state["cal_end"],state["train_end"])
            layout = market_layouts(simple_contracts(early))[early]
            return forecast(state,layout,"blend",**BASE)
        with threadpool_limits(2):
            before = get()
            obs.loc[early-pd.Timedelta(days=7):,"all_f"]=130.
            x.loc[early+pd.Timedelta(days=1):,:] = -90.
            np.testing.assert_array_equal(before,get())
        self.assertAlmostEqual(before.sum(),1.)


if __name__=="__main__":
    unittest.main()
