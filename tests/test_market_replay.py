from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from market_replay import degree_terms


class MarketReplayTests(unittest.TestCase):
    def test_degree_sums_include_both_edges_and_open_tail(self):
        row = pd.Series({"p_61F":.1,"p_62F":.3,"p_63F":.2,"p_64F":.4})
        for low,high,expected in [(62,63,.5),(None,62,.4),(64,None,.4)]:
            self.assertAlmostEqual(degree_terms(row,pd.Series({"low":low,"high":high})).sum(),expected)

    def test_replay_date_cutoff_and_settlement(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value,"2026-11-21")
        self.assertFalse(app.toggle[0].value)
        self.assertFalse(app.info)
        self.assertEqual(app.dataframe[0].value["纽约时间"].tolist(),["09:00"])
        hour = next(r for r in app.radio if "预测时点" in r.label)
        hour.set_value(15).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value["纽约时间"].tolist(),["09:00","12:00","15:00"])
        app.toggle[0].set_value(True).run()
        self.assertIn("62-63°F",app.info[0].value)
        app.selectbox[0].select("2026-11-20").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.toggle[0].value)
        self.assertFalse(app.info)


if __name__=="__main__":
    unittest.main()
