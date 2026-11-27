"""Censored sinh-arcsinh (SHASH) regression for the remaining warming R = final max - high so far >= 0.

Jones and Pewsey (2009): X = mu + sigma * sinh((asinh(Z) + eps) / delta) with Z standard normal, so
P(X < x) = Phi(S(x)), S(x) = sinh(delta * asinh((x - mu) / sigma) - eps). eps sets the skew and delta the
tails (delta < 1 heavier than normal). eps = 0, delta = 1 is the Gaussian, so the censored Gaussian is the
same code with those two fixed. R = max(0, X): days whose high is already in contribute log P(X <= 0).

mu is linear in the features, log sigma and eps linear in [1, sin, cos] of the day of year, log delta is one
constant. Fit by penalized maximum likelihood (L-BFGS-B, analytic gradient).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

LOG_DELTA = (np.log(0.25), np.log(4.0))
HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)


def _log_cosh(w):
    a = np.abs(w)
    return a + np.log1p(np.exp(-2 * a)) - np.log(2)


def unpack(theta, k, m):
    b0, beta = theta[0], theta[1:k + 1]
    gamma, eta, log_delta = theta[k + 1:k + 1 + m], theta[k + 1 + m:k + 1 + 2 * m], theta[-1]
    return b0, beta, gamma, eta, log_delta


def parameters(theta, x, z):
    b0, beta, gamma, eta, log_delta = unpack(theta, x.shape[1], z.shape[1])
    return b0 + x @ beta, z @ gamma, z @ eta, log_delta


def loss_and_grad(theta, x, z, rise, l2):
    n, k = x.shape
    m = z.shape[1]
    mu, log_sigma, eps, log_delta = parameters(theta, x, z)
    sigma, delta = np.exp(log_sigma), np.exp(log_delta)
    zero = rise <= 1e-9
    t = np.where(zero, -mu / sigma, (rise - mu) / sigma)
    a = np.arcsinh(t)
    w = np.clip(delta * a - eps, -30, 30)
    s = np.sinh(w)
    root = np.sqrt(1 + t * t)

    # Uncensored: log density. Censored: log P(X <= 0) = log Phi(S(0)).
    log_cdf = norm.logcdf(s)
    ll = np.where(zero, log_cdf,
                  log_delta + _log_cosh(w) - log_sigma - 0.5 * np.log1p(t * t) - 0.5 * s * s - HALF_LOG_2PI)
    d_w = np.where(zero, np.exp(norm.logpdf(s) - log_cdf) * np.cosh(w), np.tanh(w) - s * np.cosh(w))
    d_t = d_w * delta / root - np.where(zero, 0.0, t / (1 + t * t))
    d_mu = -d_t / sigma
    d_log_sigma = -d_t * t - np.where(zero, 0.0, 1.0)
    d_eps = -d_w
    d_log_delta = d_w * delta * a + np.where(zero, 0.0, 1.0)

    beta = theta[1:k + 1]
    loss = -ll.mean() + l2 * (beta @ beta) / n
    grad = -np.concatenate([[d_mu.sum()], x.T @ d_mu, z.T @ d_log_sigma, z.T @ d_eps, [d_log_delta.sum()]]) / n
    grad[1:k + 1] += 2 * l2 * beta / n
    return loss, grad


def fit(x, z, rise, l2, shape=True, start=None):
    """shape=False fixes eps = 0 and delta = 1 (censored Gaussian)."""
    k, m = x.shape[1], z.shape[1]
    if start is None:
        start = np.zeros(k + 2 * m + 2)
        start[0] = rise.mean()
        start[k + 1] = np.log(rise.std() + 1e-3)
    free = (None, None)
    bounds = [free] * (k + 1 + m)
    bounds += ([free] * m + [LOG_DELTA]) if shape else ([(0.0, 0.0)] * m + [(0.0, 0.0)])
    if not shape:
        start = start.copy()
        start[k + 1 + m:] = 0.0
    result = minimize(loss_and_grad, start, args=(x, z, rise, l2), jac=True, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": 2000})
    return result.x


def cdf(theta, x, z, points):
    """P(R < points) for points of shape (n, j); zero at or below 0 because R is never negative."""
    mu, log_sigma, eps, log_delta = parameters(theta, x, z)
    t = (points - mu[:, None]) / np.exp(log_sigma)[:, None]
    s = np.sinh(np.clip(np.exp(log_delta) * np.arcsinh(t) - eps[:, None], -30, 30))
    return np.where(points > 0, norm.cdf(s), 0.0)


def quantile(theta, x, z, level):
    """level-quantile of R (0 where the point mass at zero covers it)."""
    mu, log_sigma, eps, log_delta = parameters(theta, x, z)
    value = mu + np.exp(log_sigma) * np.sinh((np.arcsinh(norm.ppf(level)) + eps) / np.exp(log_delta))
    return np.maximum(value, 0.0)
