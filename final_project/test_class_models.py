"""Checks for class_models.py: hand-written backpropagation, the censored likelihood, bagging and pooling."""

import unittest

import numpy as np
import pandas as pd

import class_models as cm
import shash


def censored_data(n=300, k=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    latent = 1.0 + x @ rng.normal(size=k) + np.exp(0.3 * x[:, 0]) * rng.normal(size=n)
    return x, np.maximum(latent, 0.0)


class Backpropagation(unittest.TestCase):
    def check(self, linear_path):
        x, rise = censored_data(n=60)
        self.assertTrue(0.1 < (rise == 0).mean() < 0.9)  # both likelihood branches are exercised
        params = cm.mlp_init(x.shape[1], 3, np.random.default_rng(1), rise.mean(), linear_path)
        rng = np.random.default_rng(2)
        for key in params:  # move away from the initialization so every term is active
            params[key] = params[key] + 0.3 * rng.normal(size=params[key].shape)
        decay = 0.05
        _, grads = cm.mlp_loss_and_grads(params, x, rise, decay)
        self.assertEqual(set(grads), set(params))
        for key, value in params.items():
            numeric = np.zeros_like(value)
            for index in np.ndindex(value.shape):
                original = value[index]
                value[index] = original + 1e-6
                up = cm.mlp_loss_and_grads(params, x, rise, decay)[0]
                value[index] = original - 1e-6
                down = cm.mlp_loss_and_grads(params, x, rise, decay)[0]
                value[index] = original
                numeric[index] = (up - down) / 2e-6
            np.testing.assert_allclose(grads[key], numeric, rtol=1e-5, atol=1e-8, err_msg=key)

    def test_plain_network_gradient_matches_finite_differences(self):
        self.check(linear_path=False)

    def test_linear_path_gradient_matches_finite_differences(self):
        self.check(linear_path=True)


class Likelihood(unittest.TestCase):
    def test_matches_the_censored_gaussian_in_shash(self):
        x, rise = censored_data()
        rng = np.random.default_rng(3)
        z = np.column_stack([np.ones(len(x)), rng.uniform(-1, 1, (len(x), 2))])
        k, m = x.shape[1], z.shape[1]
        theta = np.r_[0.5, rng.normal(size=k) * 0.5, 0.2, 0.1, -0.1, np.zeros(m), 0.0]  # eps = 0, delta = 1
        mu, log_sigma, _, _ = shash.parameters(theta, x, z)
        nll, d_mu, d_log_sigma = cm.censored_terms(mu, log_sigma, rise)
        loss, grad = shash.loss_and_grad(theta, x, z, rise, 0.0)
        self.assertAlmostEqual(nll.mean(), loss, places=10)
        np.testing.assert_allclose(grad[0], d_mu.mean(), rtol=1e-8)
        np.testing.assert_allclose(grad[k + 1:k + 1 + m], z.T @ d_log_sigma / len(x), rtol=1e-8)

    def test_bands_are_distributions_with_no_mass_below_zero_rise(self):
        mu, sigma = np.array([2.0, -1.0, 8.0]), np.array([3.0, 1.0, 5.0])
        edges = np.array([[-5.0, 1.0, 6.0, 20.0], [-20.0, -5.0, 3.0, 18.0], [-30.0, -15.0, 0.0, 15.0]])
        bands = cm.gaussian_bands(mu, sigma, edges)
        np.testing.assert_allclose(bands.sum(axis=1), 1.0)
        self.assertTrue((bands[0, 0] == 0) and (bands[1, :2] == 0).all() and (bands[2, :3] == 0).all())


class Training(unittest.TestCase):
    def test_training_improves_on_the_constant_start_and_keeps_the_best_epoch(self):
        x, rise = censored_data(n=900, seed=4)
        train, val = slice(0, 600), slice(600, None)
        predict, best_epoch, curve = cm.mlp_train(x[train], rise[train], x[val], rise[val], 4, 1e-3, seed=0,
                                                  max_epochs=120, patience=20)
        mu, sigma = predict(x[val])
        scale = rise[train].std()
        start = cm.censored_terms(np.full(300, rise[train].mean()), np.full(300, np.log(scale)), rise[val])[0].mean()
        achieved = cm.censored_terms(mu, np.log(sigma), rise[val])[0].mean()
        self.assertLess(achieved, start - 0.2)
        self.assertTrue((sigma > 0).all())
        # Early stopping returns the best validation epoch (curve is in rescaled units: shift by log scale).
        self.assertEqual(best_epoch, int(np.argmin(curve)) + 1)
        uncensored = (rise[val] > 1e-9).mean()
        self.assertAlmostEqual(achieved, min(curve) + uncensored * np.log(scale), places=6)


class BaggingAndPooling(unittest.TestCase):
    def test_week_resample_draws_whole_blocks(self):
        dates = pd.date_range("2021-01-01", periods=70, freq="D").delete([3, 17, 40])
        blocks = cm.week_blocks(dates)
        self.assertEqual(len(blocks), 10)
        self.assertEqual(sum(len(b) for b in blocks), len(dates))
        rows = cm.block_resample(blocks, np.random.default_rng(0))
        counts = np.bincount(rows, minlength=len(dates))
        for block in blocks:
            self.assertEqual(len(set(counts[block])), 1)  # every day of a week is drawn equally often

    def test_simplex_grid_and_weight_choice(self):
        self.assertEqual(len(cm.simplex(2)), 21)
        self.assertEqual(len(cm.simplex(3)), 231)
        self.assertTrue(all(abs(w.sum() - 1) < 1e-12 for w in cm.simplex(3)))
        rng = np.random.default_rng(5)
        target = rng.integers(0, 5, 200)
        good = np.full((200, 5), 0.05)
        good[np.arange(200), target] = 0.8
        flat = np.full((200, 5), 0.2)
        mask = np.ones(200, dtype=bool)
        weights, loss = cm.best_weights([good, flat], target, mask, cm.simplex(2))
        np.testing.assert_allclose(weights, [1.0, 0.0])
        folds = np.arange(200) % 2
        self.assertGreaterEqual(cm.cross_fitted_loss([good, flat], target, mask, folds, cm.simplex(2)), loss - 1e-12)


if __name__ == "__main__":
    unittest.main()
