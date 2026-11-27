"""Checks for the weighted censored Gaussian, the walk-forward label rule and the drift diagnostics."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nonstationarity as ns
import shash


def synthetic(n=400, k=4, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    angle = rng.uniform(0, 2 * np.pi, n)
    z = np.column_stack([np.ones(n), np.sin(angle), np.cos(angle)])
    latent = 2 + x @ np.array([1.5, -1.0, 0.5, 0.0]) + np.exp(0.3 + 0.2 * z[:, 1]) * rng.normal(size=n)
    return x, z, np.maximum(latent, 0.0)


class WeightedFit(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        x, z, rise = synthetic()
        self.assertGreater((rise == 0).mean(), 0.05)  # both censored and uncensored rows are exercised
        rng = np.random.default_rng(2)
        w = rng.uniform(0.1, 3.0, len(rise))
        w /= w.mean()
        theta = rng.normal(scale=0.3, size=1 + x.shape[1] + z.shape[1])
        _, grad = ns.weighted_loss(theta, x, z, rise, 5.0, w)
        step, numeric = 1e-6, np.zeros_like(theta)
        for i in range(len(theta)):
            up, down = theta.copy(), theta.copy()
            up[i] += step
            down[i] -= step
            numeric[i] = (ns.weighted_loss(up, x, z, rise, 5.0, w)[0] - ns.weighted_loss(down, x, z, rise, 5.0, w)[0]) / (2 * step)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)

    def test_uniform_weights_reproduce_shash_censored_gaussian(self):
        x, z, rise = synthetic()
        ours = ns.fit_weighted(x, z, rise, 1.0)
        theirs = shash.fit(x, z, rise, 1.0, shape=False)
        np.testing.assert_allclose(ours, theirs, atol=2e-4)

    def test_integer_weights_equal_duplicated_rows(self):
        x, z, rise = synthetic(n=200)
        counts = np.random.default_rng(3).integers(1, 4, len(rise))
        weighted = ns.fit_weighted(x, z, rise, 0.0, counts.astype(float))
        repeat = np.repeat(np.arange(len(rise)), counts)
        duplicated = ns.fit_weighted(x[repeat], z[repeat], rise[repeat], 0.0)
        np.testing.assert_allclose(weighted, duplicated, atol=2e-4)

    def test_recency_weights_halve_every_half_life(self):
        dates = pd.to_datetime(["2023-12-31", "2022-01-01", "2020-01-01", "2024-03-01"])
        w = ns.recency(dates, "2024-01-01", 2.0)
        age = (pd.Timestamp("2024-01-01") - dates[:3]).days / 365.25
        np.testing.assert_allclose(w[:3], 0.5 ** (age / 2.0))
        self.assertEqual(w[3], 1.0)  # a day after the reference is never up-weighted
        np.testing.assert_array_equal(ns.recency(dates, "2024-01-01", None), np.ones(4))


class FakeProblem:
    """fit() records the newest training date; predictions carry it so each row's source can be checked."""

    def __init__(self, dates):
        self.samples = pd.DataFrame(index=pd.DatetimeIndex(dates))

    def fit(self, columns, train, weights, grid):
        newest = self.samples.index[train].max()
        return {"p": np.full((len(self.samples), 5), float(newest.toordinal()))}


class WalkForward(unittest.TestCase):
    def test_training_labels_are_settled_two_days_before_the_month(self):
        dates = pd.date_range("2026-04-01", "2026-07-31")
        train = ns.walk_forward_train(dates, np.ones(len(dates), bool), pd.Timestamp("2026-06-01"))
        self.assertEqual(dates[train].max(), pd.Timestamp("2026-05-30"))

    def test_each_month_is_predicted_only_from_earlier_labels(self):
        dates = pd.date_range("2025-06-01", "2026-08-31")
        months = list(pd.date_range("2025-09-01", "2026-08-01", freq="MS"))
        p = ns.walk_forward(FakeProblem(dates), [], np.ones(len(dates), bool), months, l2=1.0)
        for i, day in enumerate(dates):
            if day < months[0]:
                self.assertTrue(np.isnan(p[i, 0]))
                continue
            newest = pd.Timestamp.fromordinal(int(p[i, 0]))
            self.assertLessEqual(newest, day.replace(day=1) - pd.Timedelta(days=2))


class Diagnostics(unittest.TestCase):
    def test_max_t_matches_a_direct_search(self):
        values = np.random.default_rng(4).normal(size=30)
        k, t, shift = ns._max_t(values, 5)
        best = None
        for split in range(5, 26):
            a, b = values[:split], values[split:]
            pooled = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / (len(values) - 2)
            stat = (b.mean() - a.mean()) / np.sqrt(pooled * (1 / len(a) + 1 / len(b)))
            if best is None or abs(stat) > abs(best[1]):
                best = (split, stat, b.mean() - a.mean())
        self.assertEqual(k[0], best[0])
        self.assertAlmostEqual(t[0], best[1])
        self.assertAlmostEqual(shift[0], best[2])

    def test_level_shift_finds_a_planted_step(self):
        rng = np.random.default_rng(5)
        values = rng.normal(scale=0.5, size=120) + np.where(np.arange(120) >= 70, 1.0, 0.0)
        anomaly = pd.Series(values, index=pd.period_range("2012-01", periods=120, freq="M"))
        found = ns.level_shift(anomaly, draws=500)
        self.assertEqual(found["break_month"], str(anomaly.index[70]))
        self.assertLess(found["p"], 0.01)
        self.assertAlmostEqual(found["shift"], 1.0, delta=0.25)
        flat = ns.level_shift(pd.Series(rng.normal(size=120), index=anomaly.index), draws=500)
        self.assertGreater(flat["p"], 0.05)

    def test_era_indicators_mark_only_the_older_era(self):
        samples = pd.DataFrame(index=pd.to_datetime(["2014-12-31", "2015-01-01", "2024-06-01", "2026-06-01"]))
        out = ns.era_features(samples, {"GFS MOS daytime max": pd.Timestamp("2015-01-01")})
        np.testing.assert_array_equal(out.era_gfs_max, [1, 0, 0, 0])  # validation and test are in the latest era
        np.testing.assert_array_equal(out.period_2014_2017, [1, 1, 0, 0])


class Results(unittest.TestCase):
    def test_scored_on_the_same_days_and_walk_forward_reported_separately(self):
        out = Path(__file__).resolve().parent / "results" / "nonstationarity"
        if not (out / "test_scores.csv").exists():
            self.skipTest("run nonstationarity.py first")
        scores = pd.read_csv(out / "test_scores.csv")
        self.assertTrue(scores.n.eq(359).all())
        walk = scores.system.str.startswith("Walk-forward")
        self.assertTrue(scores.loc[walk, "part"].eq("c walk-forward").all())
        selection = pd.read_csv(out / "selection.csv")
        fixed = selection.loc[selection.pool.str.startswith("fixed"), "selected"]
        self.assertFalse(fixed.str.startswith("Walk-forward").any())
        self.assertLess(selection.current_refit_max_abs_prob_diff_vs_saved.max(), 0.01)
        shifts = pd.read_csv(out / "level_shifts.csv")
        training = shifts.loc[shifts.data.str.contains("training")]
        self.assertTrue(training.last_month.le("2023-12").all())  # era features never see 2024 or later


if __name__ == "__main__":
    unittest.main()
