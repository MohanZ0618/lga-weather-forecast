import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from features import TZ, ground, fusion, numerical
from models import physical, choose, fit_level
from runtime import latest_slot, metar_tables, select_tier
from sources import eligible_cycle, goes_time


class TimingTests(unittest.TestCase):
    def test_training_window_follows_calibration(self):
        for start, end in [("2024-09", "2025-09"), ("2025-09", "2025-09")]:
            with self.assertRaisesRegex(ValueError, "calibration"):
                fit_level(pd.DataFrame(), "station", start, end)

    def test_service_hours_dst_and_after_close(self):
        for instant, desired in [("2026-03-08 10:29Z","2026-03-08 06:15"),
                                 ("2026-11-01 11:14Z","2026-11-01 06:00"),
                                 ("2026-07-10 05:15Z","2026-07-09 20:00"),
                                 ("2026-07-10 00:53Z","2026-07-09 20:00")]:
            self.assertEqual(latest_slot(pd.Timestamp(instant)),pd.Timestamp(desired,tz=TZ))

    def test_cloud_scan_uses_end_and_issue_times(self):
        at = pd.Timestamp("2026-05-01 10:15",tz=TZ)
        base = {"direction":90., "LGA_temp":65., "sun":.6}
        def scan(minutes):
            return {"end":(at-pd.Timedelta(minutes=minutes+5)).isoformat(),
                    "available":(at-pd.Timedelta(minutes=minutes+2)).isoformat(),
                    "fraction":[.3,.4,.5,.6,.7,.8,.9,1.,.2,.3]}
        snapshots = {"scans":[scan(0),scan(30),scan(60)]}
        result = fusion(at,base,snapshots)
        self.assertAlmostEqual(result["upwind"],.7)
        self.assertEqual(result["delta60"],0)
        snapshots["scans"][0]["available"]=(at+pd.Timedelta(seconds=1)).isoformat()
        self.assertTrue(np.isnan(fusion(at,base,snapshots)["sat10"]))
        snapshots["scans"][0]=scan(31)
        self.assertTrue(np.isnan(fusion(at,base,snapshots)["sat10"]))
        self.assertEqual(goes_time("e20261651503563"),pd.Timestamp("2026-06-14 15:03:56.3Z"))

    def test_receipts_and_future_reports(self):
        at = pd.Timestamp("2026-05-01 10:00",tz=TZ)
        def metar(obs,received,temp):
            return {"icaoId":"KLGA","obsTime":obs.timestamp(),"receiptTime":received.isoformat(),
                    "temp":temp,"dewp":8,"wspd":5,"wdir":270,"cover":"SCT"}
        records = [metar(at-pd.Timedelta(hours=2),at-pd.Timedelta(minutes=113),12),
                   metar(at-pd.Timedelta(minutes=30),at-pd.Timedelta(minutes=28),20),
                   metar(at-pd.Timedelta(minutes=1),at+pd.Timedelta(minutes=2),29),
                   metar(at+pd.Timedelta(minutes=30),at+pd.Timedelta(minutes=33),36)]
        base,stamps=ground(metar_tables(records),at)
        self.assertEqual(base["LGA_temp"],68)
        self.assertAlmostEqual(base["high_so_far"],53.6)
        self.assertEqual(pd.Timestamp(stamps["LGA"]),at-pd.Timedelta(minutes=30))
        self.assertIsNone(ground(metar_tables(records[:1]),at)[0])

    def test_cycle_expiry_and_day_boundary(self):
        at = pd.Timestamp("2026-05-01 10:15",tz=TZ)
        cycle={"run":"2026-05-01T10:00:00Z","available":"2026-05-01T12:00:00Z","values":list(range(18))}
        self.assertTrue(eligible_cycle(cycle,at))
        values=numerical(at,{"LGA_temp":10},cycle,None)
        self.assertEqual(values["nwp_high"],16)
        self.assertEqual(values["run_age"],4.25)
        cycle["values"]=[1,2]
        self.assertFalse(eligible_cycle(cycle,at))
        cycle["available"]="2026-05-01T15:00:00Z"
        self.assertTrue(np.isnan(numerical(at,{"LGA_temp":10},cycle,None)["nwp_high"]))

    def test_fallback_and_probability_support(self):
        policy={"routing":{"satellite":"upwind_cloud","nwp":"latest"}}
        for nwp,cloud,expected in [(np.nan,.4,"station"),(72,np.nan,"latest"),(72,.4,"upwind_cloud")]:
            self.assertEqual(select_tier({"nwp_high":nwp,"sat10":cloud},policy),expected)
        p=physical(np.array([[1,0,0,0,0],[0,0,0,0,1]]),np.array([70,85]))
        np.testing.assert_allclose(p.sum(axis=1),1)
        self.assertTrue((p[0,:3]==0).all())
        self.assertEqual(p[1,-1],1)
        options={"a":{"accuracy":.8,"log_loss":.50},"b":{"accuracy":.85,"log_loss":.51}}
        self.assertEqual(choose(options,.5),"a")

