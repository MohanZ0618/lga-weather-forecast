"""Checks for the censored SHASH model and the 2010-2017 history extension."""

import unittest

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

import shash
from prepare import DATA


def simulated(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    day = rng.uniform(0, 2 * np.pi, n)
    z = np.column_stack([np.ones(n), np.sin(day), np.cos(day)])
    theta = np.r_[2.0, rng.normal(size=4), np.log(2.0), 0.2, -0.1, 0.5, 0.2, 0.0, np.log(0.7)]
    mu, log_sigma, eps, log_delta = shash.parameters(theta, x, z)
    draw = mu + np.exp(log_sigma) * np.sinh((np.arcsinh(rng.normal(size=n)) + eps) / np.exp(log_delta))
    return x, z, np.maximum(draw, 0.0), theta


class CensoredShash(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        x, z, rise, theta = simulated()
        point = theta + np.random.default_rng(1).normal(size=theta.size) * 0.3
        analytic = shash.loss_and_grad(point, x, z, rise, 1.0)[1]
        numeric = approx_fprime(point, lambda t: shash.loss_and_grad(t, x, z, rise, 1.0)[0], 1e-7)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-4)

    def test_gaussian_is_the_fixed_shape_special_case(self):
        x, z, rise, _ = simulated()
        theta = shash.fit(x, z, rise, 1.0, shape=False)
        np.testing.assert_array_equal(theta[-4:], 0.0)  # eps = 0 and log delta = 0

    def test_fit_recovers_skew_and_tails(self):
        x, z, rise, truth = simulated(n=4000)
        theta = shash.fit(x, z, rise, 0.01)
        self.assertAlmostEqual(theta[-4], truth[-4], delta=0.15)                # eps intercept
        self.assertAlmostEqual(np.exp(theta[-1]), np.exp(truth[-1]), delta=0.1)  # delta

    def test_quantile_inverts_cdf_and_mass_below_zero_is_zero(self):
        x, z, rise, theta = simulated()
        q = shash.quantile(theta, x, z, 0.7)
        positive = q > 0
        np.testing.assert_allclose(shash.cdf(theta, x, z, q[:, None])[positive, 0], 0.7, atol=1e-9)
        np.testing.assert_array_equal(shash.cdf(theta, x, z, np.zeros((len(x), 1))), 0.0)


class HistoryExtension(unittest.TestCase):
    def test_rebuilt_2018_rows_equal_the_bundled_samples(self):
        check = pd.read_csv(DATA / "history" / "overlap_check_2018.csv")
        self.assertEqual(len(check), 3)
        self.assertTrue(check.same_dates.all())
        self.assertTrue((check.cells_equal == 1.0).all())

    def test_history_ends_before_the_bundled_samples(self):
        for hour in (9, 12, 15):
            early = pd.read_csv(DATA / "history" / f"samples_{hour:02}_2010_2017.csv.gz", parse_dates=["date"])
            self.assertLess(early.date.max(), pd.Timestamp("2018-01-01"))
            self.assertTrue(early.date.is_unique)


if __name__ == "__main__":
    unittest.main()
