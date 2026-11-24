from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
from datetime import datetime,timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/"bracket_study"))
from estimators import band_probs,Distribution,FAMILIES,Routed,defaults,BASE,present
from nbm_archive import parse_card,read_card
from panels import Brackets,events,history,SCREEN,TUNE,FREEZE
from train import trials


class BracketTests(unittest.TestCase):
    def test_actual_brackets_rounding_open_tails_and_padding(self):
        ev = events(9)
        different = ev.loc[ev.brackets.str.len().isin([7,11])].groupby(ev.brackets.str.len()).head(1)
        b = Brackets(different)
        for day in [39.49,39.5,69.49,69.5,100.01,-100.]:
            probs = b.unpack(band_probs((day>=b.cuts).astype(float)))
            for event,p in zip(different.itertuples(),probs):
                from market_study import DEGREES
                self.assertEqual(p.argmax(),event.bins[:,int(np.floor(day+.5))-DEGREES[0]].argmax())
                self.assertAlmostEqual(p.sum(),1.)

    def test_history_and_freeze(self):
        frame = pd.DataFrame(dict(label_f=[1,2,3],hours=24,blocks=4),
                             index=pd.to_datetime(["2024-12-31","2025-01-01","2025-09-01"]))
        self.assertEqual(len(history(frame,"2025-01-01","all_f")),1)
        self.assertEqual(SCREEN[-1][1],TUNE[0][0])
        self.assertEqual(TUNE[-1][1],FREEZE)
        self.assertEqual(len(list(trials("monotone_xgb"))),65)

    def test_nbm_card_issue_and_midnight(self):
        ny = ZoneInfo("America/New_York")
        for month in (1,7):
            at = datetime(2024,month,10,9,tzinfo=ny)
            run = at.astimezone(timezone.utc)-pd.Timedelta(hours=1)
            times = [(run+pd.Timedelta(hours=i+1)).hour for i in range(25)]
            header = f" KLGA   NBM V4.2 NBH GUIDANCE   {run:%m/%d/%Y}  {run:%H%M} UTC"
            fields = {"UTC":times,"TMP":[75]*24+[120],"TSD":[2]*25,"SKY":[50]*25}
            text = "\n".join([header]+[f" {k:3} "+"".join(f"{n:3}" for n in v) for k,v in fields.items()])
            row = parse_card(text.encode(),run,at)
            self.assertEqual(row["nbm_high"],75)
            self.assertEqual(row["nbm_spread"],2)
            with self.assertRaises(ValueError):
                parse_card(text.encode(),run+pd.Timedelta(hours=1),at)

    def test_model_families_obey_closed_hour_floor(self):
        rng = np.random.default_rng(4260)
        n = 120
        x = pd.DataFrame(rng.normal(size=(n,len(BASE))),columns=BASE)
        x["floor_f"] = rng.integers(35,78,size=n)
        x["label_f"] = x.floor_f + np.maximum(0,rng.integers(-5,10,size=n))
        x["LGA_temp"] = x.floor_f
        cuts = np.tile([40.,55.,70.,85.],(n,1))
        for family in FAMILIES:
            p = defaults(family)
            p["trees"] = 8
            pred = Distribution(family,"station",p).fit(x).survival(x,cuts)
            self.assertTrue((np.diff(pred,axis=1)<=1e-10).all())
            np.testing.assert_array_equal(pred[x.floor_f.to_numpy()[:,None]>=cuts],1.)
            np.testing.assert_allclose(band_probs(pred).sum(axis=1),1.)
        x["latest_high"],x["latest_warm"],x["nbm_high"] = 70.,2.,np.nan
        self.assertTrue(present(x,"hrrr").all())
        self.assertFalse(present(x,"nbm").any())

    def test_delayed_archive_publication_is_unavailable(self):
        at = datetime(2024,6,1,13,tzinfo=timezone.utc)
        response = Mock(status_code=206,headers={"Last-Modified":"Sat, 01 Jun 2024 13:00:01 GMT"})
        with patch("nbm_archive.client") as http:
            http.return_value.get.return_value = response
            self.assertIsNone(read_card("https://example.test/forecast",at))

    def test_nested_source_fallback_preserves_date_order(self):
        panel = pd.DataFrame(dict(latest_high=[60.,np.nan,60.],latest_warm=[1.,np.nan,1.],
                                  nbm_high=[np.nan,60.,60.]))
        model = Routed("threshold","nbm",defaults("threshold"))
        model.models = {}
        for name,p in (("station",.2),("hrrr",.4),("nbm",.7)):
            pred = Mock()
            pred.survival.return_value = np.array([[p]])
            model.models[name] = pred
        probs,route = model.predict(panel,np.full((3,1),65.))
        self.assertEqual(route.tolist(),["hrrr","station","nbm"])
        np.testing.assert_allclose(probs[:,1],[.4,.2,.7])


if __name__=="__main__":
    unittest.main()
