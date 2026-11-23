import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from features import TZ
from temperature import (FINE_BOUNDS, day_features, day_probabilities, fine_probabilities, issue_time, live_day, next_day_high,
                         observations, predict_points, target_cycle)
from runtime import metar_tables


def report(t, temp, delay=0):
    return {"icaoId":"KLGA","obsTime":int(t.timestamp()),"receiptTime":(t+pd.Timedelta(minutes=delay)).isoformat(),
            "temp":temp,"wspd":5,"wdir":90}


class HorizonTests(unittest.TestCase):
    def test_dst_forecast_uses_every_hour_of_the_target_day(self):
        for day, hours in [(date(2026,3,8),23),(date(2026,11,1),25)]:
            run = target_cycle(day)
            start = pd.Timestamp(day).tz_localize(TZ)
            first = int((start-run)/pd.Timedelta(hours=1))
            v = np.zeros(48)
            v[first-2] = v[first+hours-1] = 99
            v[first+hours-2] = 77
            cycle = {"run":run.isoformat(),"available":issue_time(day).isoformat(),"values":v}
            self.assertEqual(next_day_high(cycle,day),77)
            self.assertTrue(np.isnan(next_day_high({**cycle,"values":v[:first+hours-2]},day)))
            late = issue_time(day)+pd.Timedelta(seconds=1)
            self.assertTrue(np.isnan(next_day_high({**cycle,"available":late.isoformat()},day)))

    def test_previous_noon_excludes_later_observations_and_receipts(self):
        day = date(2026,11,5)
        t = issue_time(day)
        reports = [report(t-pd.Timedelta(hours=2),12),report(t-pd.Timedelta(minutes=10),18)]
        row = day_features(metar_tables(reports),day,None)
        reports += [report(t-pd.Timedelta(minutes=2),40,delay=10),report(t+pd.Timedelta(days=1),43)]
        test = day_features(metar_tables(reports),day,None)
        self.assertEqual(row["LGA_temp"],test["LGA_temp"])
        self.assertEqual(row["high_so_far"],test["high_so_far"])
        with patch("temperature.recent_metars", side_effect=AssertionError):
            self.assertIsNone(live_day(day,t-pd.Timedelta(seconds=1)))

    def test_probability_mass_and_exact_boundary(self):
        probs = day_probabilities(np.array([70.,55.]),np.zeros(100))
        np.testing.assert_allclose(probs.sum(axis=1),1)
        np.testing.assert_array_equal(probs.argmax(axis=1),[3,2])
        self.assertTrue((probs>0).all())

    def test_fine_ranges_preserve_broad_probabilities_and_observed_floor(self):
        points = np.array([67., 95., 10.])
        floors = np.array([60., 95., -np.inf])
        broad = np.array([[0,0,.7,.2,.1],[0,0,0,0,1],[.98,.01,.006,.003,.001]])
        fine = fine_probabilities(points,floors,broad,np.arange(-4.,5.))
        parents = np.searchsorted([40,55,70,85],np.r_[-np.inf,FINE_BOUNDS],side="right")
        for i in range(5):
            np.testing.assert_allclose(fine[:,parents==i].sum(axis=1),broad[:,i])
        self.assertTrue((fine[0,:np.searchsorted(FINE_BOUNDS,60,side="right")]==0).all())
        self.assertEqual(fine[1,-1],1)

    def test_final_observation_and_signed_difference(self):
        day = date(2026,11,5)
        start = pd.Timestamp(day).tz_localize(TZ)
        records = [report(start+pd.Timedelta(hours=h,minutes=50),10+h*.5) for h in range(24)]
        obs, pending = observations(records,day,start+pd.Timedelta(hours=14))
        self.assertEqual(len(obs),14)
        self.assertIsNone(pending)
        _, actual = observations(records,day,start+pd.Timedelta(hours=25))
        self.assertAlmostEqual(actual,70.7)
        frame = pd.DataFrame({"high_so_far":[70.],"nwp_high":[65.],"expected_rise":[2.]})
        point = predict_points(frame,"rolling",{"estimator":None})[0]
        self.assertAlmostEqual(point-actual,-0.7)
        self.assertIsNone(observations(records[:-1],day,start+pd.Timedelta(hours=25))[1])
