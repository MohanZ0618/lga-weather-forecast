"""Forecast the remaining warming as a full distribution, then integrate it onto the five bands.

The five-class models in experiment.py learn each band separately. Here two models predict
R = (final hourly-observation max) - (max of the closed hours so far), which is never negative and
is exactly zero once the high is already in (12% of days at 09:00, 63% at 15:00). Any set of
bins (the five course bands, or 2 F market brackets) is then a calculation, not a new model:
P(band) = P(floor + R in band).

- Censored Gaussian regression (Tobit, EMOS style; Gneiting et al. 2005). R = max(0, R*) with
  R* ~ Normal(mu, sigma), mu linear in the features, log sigma linear in season. Fit by
  minimizing the negative log-likelihood with an L2 penalty.
- Quantile gradient boosting. One XGBoost model predicts nine quantiles of R (pinball loss),
  turned into a distribution by linear interpolation between quantiles. A spread factor around
  the median is chosen on 2024 validation, as for every other setting.

Same splits, cohorts, 1% climatology mix and floor rule as experiment.py. Mohan's classifiers are
refit with the same two added floor-relative features so the comparison isolates the formulation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

import experiment
from experiment import (BOUNDS, FORECAST_FEATURES, HOURS, PROBS, feature_columns, fit_scope,
                        month_prior, season_features, split_dates, stabilize)
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
TAUS = np.array([.01, .05, .10, .25, .50, .75, .90, .95, .99])
SCALES = np.round(np.arange(0.8, 1.61, 0.1), 2)


def add_relative(samples):
    """Floor-relative inputs: how far the latest reading and HRRR sit above the high so far."""
    out = samples.assign(rel_now=samples.LGA_temp - samples.LGA_max_so_far)
    if "hrrr_remaining_max" in out:
        out["rel_hrrr"] = out.hrrr_remaining_max - out.LGA_max_so_far
    if "lamp_remaining_max" in out:  # station guidance from guidance.py
        out["lamp_now_error"] = out.LGA_temp - out.lamp_now
        out["rel_lamp"] = out.lamp_remaining_max - out.LGA_max_so_far
        out["rel_nbm"] = out.nbm_max - out.LGA_max_so_far
    return out


GUIDANCE = ["lamp_remaining_max", "rel_lamp", "lamp_now_error", "lamp_cloud", "lamp_u", "lamp_v",
            "nbm_max", "rel_nbm", "nbm_max_sd"]


def relative_columns(samples, scope):
    base = scope.replace(" + LAMP/NBM", "")
    extra = [c for c in ("rel_now", "rel_hrrr") if c in samples and (c != "rel_hrrr" or "HRRR" in base)]
    guidance = GUIDANCE if scope.endswith(" + LAMP/NBM") else []
    return original_columns(samples, base) + extra + guidance


original_columns = feature_columns


def design(samples, columns, train):
    x = samples[columns].astype(float)
    median = x[train].median()
    x = x.fillna(median)
    mean, sd = x[train].mean(), x[train].std().replace(0, 1)
    return ((x - mean) / sd).to_numpy()


def band_cdf_edges(samples):
    """Rise needed to reach each band edge from today's floor: shape (n, 4)."""
    return BOUNDS[None, :] - samples.LGA_max_so_far.to_numpy()[:, None]


def to_bands(cdf_at_edges):
    cdf = np.column_stack([np.zeros(len(cdf_at_edges)), cdf_at_edges, np.ones(len(cdf_at_edges))])
    return np.clip(np.diff(cdf, axis=1), 0, None)


# Censored Gaussian (Tobit) ------------------------------------------------------------------

def tobit_fit(x, z, rise, l2):
    zero = rise <= 1e-9
    k, m = x.shape[1], z.shape[1]

    def loss(theta):
        b0, beta, gamma = theta[0], theta[1:k + 1], theta[k + 1:]
        mu = b0 + x @ beta
        sigma = np.exp(z @ gamma)
        nll = np.where(zero, -norm.logcdf(-mu / sigma),
                       -norm.logpdf((rise - mu) / sigma) + np.log(sigma))
        return nll.mean() + l2 * (beta @ beta) / len(rise)

    start = np.r_[rise.mean(), np.zeros(k), np.log(rise.std() + 1e-3), np.zeros(m - 1)]
    return minimize(loss, start, method="L-BFGS-B").x, k


def tobit_bands(theta, k, x, z, edges):
    mu = theta[0] + x @ theta[1:k + 1]
    sigma = np.exp(z @ theta[k + 1:])
    # P(R < e) = Phi((e - mu) / sigma) for e > 0, else 0 (R is never negative).
    cdf = np.where(edges > 0, norm.cdf((edges - mu[:, None]) / sigma[:, None]), 0.0)
    return to_bands(cdf), mu, sigma


# Quantile boosting --------------------------------------------------------------------------

def quantile_cdf(q, x):
    """P(R < x) for a piecewise-linear quantile function with knots at TAUS (left limit).

    Tails are extended to probability 0 and 1 by the slope of the outer pair. Zero-width pieces
    (several quantiles at exactly 0) keep their probability as a point mass at that value.
    """
    q = np.maximum(np.sort(q, axis=1), 0)
    lo = np.maximum(q[:, 0] - (q[:, 1] - q[:, 0]) * .01 / .04, 0)
    hi = q[:, -1] + (q[:, -1] - q[:, -2]) * .01 / .04
    knots = np.column_stack([lo, q, hi])
    weights = np.diff(np.r_[0., TAUS, 1.])
    cdf = np.zeros(x.shape)
    for i, w in enumerate(weights):
        a, b = knots[:, i, None], knots[:, i + 1, None]
        width = b - a
        frac = np.where(width > 0, np.clip((x - a) / np.where(width > 0, width, 1), 0, 1), x > a)
        cdf += w * frac
    return cdf


def scaled(q, s):
    middle = q[:, [4]]
    return middle + s * (q - middle)


# Experiment ---------------------------------------------------------------------------------

def run_cohort(hour, samples, cohort, scope, log):
    samples = add_relative(samples)
    splits = split_dates(samples.index)
    train, validation = splits["train"], splits["validation"]
    prior = month_prior(samples.loc[train], samples)
    floor = samples.LGA_max_so_far.to_numpy()
    target = samples.target_category.to_numpy()
    rise = (samples.target_max - samples.LGA_max_so_far).to_numpy()
    edges = band_cdf_edges(samples)
    columns = relative_columns(samples, scope)
    x = design(samples, columns, train)
    z = np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos])
    out = {}

    def val_loss(p):
        return daily_scores(p[validation], target[validation])["log_loss"].mean()

    # Mohan's classifier families on the same augmented features.
    experiment.feature_columns = relative_columns
    try:
        tuning = []
        predictions, selected = fit_scope(samples, splits, scope, prior, tuning)
    finally:
        experiment.feature_columns = original_columns
    out["Classifiers + relative features"] = predictions[selected]

    best = None
    for l2 in (0.1, 1.0, 10.0, 30.0, 100.0, 300.0):
        theta, k = tobit_fit(x[train], z[train], rise[train], l2)
        p = stabilize(tobit_bands(theta, k, x, z, edges)[0], prior, floor)
        loss = val_loss(p)
        log.append({"hour": hour, "cohort": cohort, "model": "Censored Gaussian", "setting": f"l2={l2}",
                    "validation_log_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, p, theta, k)
    out["Censored Gaussian"] = best[1]
    _, mu, sigma = tobit_bands(best[2], best[3], x, z, edges)

    best = None
    for depth in (2, 3, 4):
        model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=TAUS, n_estimators=300,
                             learning_rate=0.05, max_depth=depth, min_child_weight=20, subsample=0.9,
                             colsample_bytree=0.9, random_state=4260, n_jobs=2)
        model.fit(x[train], rise[train])
        q = model.predict(x)
        for s in SCALES:
            p = stabilize(to_bands(quantile_cdf(scaled(q, s), edges)), prior, floor)
            loss = val_loss(p)
            log.append({"hour": hour, "cohort": cohort, "model": "Quantile boosting",
                        "setting": f"depth={depth}, scale={s}", "validation_log_loss": loss})
            if best is None or loss < best[0]:
                best = (loss, p, scaled(q, s))
    out["Quantile boosting"] = best[1]
    qbest = np.maximum(np.sort(best[2], axis=1), 0)

    # Continuous checks on the final max itself (degrees F), test split only.
    test = splits["test"]
    y = samples.target_max.to_numpy()
    lo80 = floor + qbest[:, 2]  # 10th percentile
    hi80 = floor + qbest[:, 6]      # 90th percentile
    continuous = {
        "hour": hour, "cohort": cohort, "n": int(test.sum()),
        "quantile_median_mae_f": float(np.abs(floor + qbest[:, 4] - y)[test].mean()),
        "quantile_80_coverage": float(((y >= lo80) & (y <= hi80))[test].mean()),
        "quantile_80_width_f": float((hi80 - lo80)[test].mean()),
        "gaussian_mean_mae_f": float(np.abs(floor + np.maximum(mu, 0) - y)[test].mean()),
    }
    return out, splits, continuous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "distribution")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(ROOT / "results" / "daily_predictions.csv", parse_dates=["date"])
    forecast = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    log, rows, paired, continuous, daily = [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples = season_features(pd.read_csv(args.data / f"samples_{hour:02}.csv.gz",
                                                  parse_dates=["date"], index_col="date").sort_index())
            matched = samples.join(forecast.loc[forecast.hour == hour].set_index("date")[FORECAST_FEATURES],
                                   how="inner", validate="one_to_one")
            for cohort, data, scope in (("Observations", samples, "Four stations + trends"),
                                        ("Matched HRRR", matched, "Four stations + trends + HRRR")):
                models, splits, cont = run_cohort(hour, data, cohort, scope, log)
                continuous.append(cont)
                test = splits["test"]
                dates = data.index[test]
                target = data.target_category.to_numpy()[test]
                ref = baseline.loc[baseline.cohort.eq(cohort) & baseline.hour.eq(hour) & baseline.scope.eq(scope)]
                ref = ref.set_index("date").loc[dates]
                systems = {"Selected classifier (paper)": ref[PROBS].to_numpy(),
                           **{name: p[test] for name, p in models.items()}}
                scores = {name: daily_scores(p, target) for name, p in systems.items()}
                for name, p in systems.items():
                    rows.append({"hour": hour, "cohort": cohort, "model": name, "n": len(target),
                                 **scores[name].mean().to_dict()})
                    frame = pd.DataFrame(p, columns=PROBS, index=dates).reset_index()
                    daily.append(frame.assign(hour=hour, cohort=cohort, model=name, target_category=target))
                for name in models:
                    for metric in ("log_loss", "rps", "correct"):
                        diff = (scores[name][metric] - scores["Selected classifier (paper)"][metric]).to_numpy()
                        mean, low, high = block_bootstrap(dates, diff)
                        paired.append({"hour": hour, "cohort": cohort, "candidate": name, "metric": metric,
                                       "delta_vs_paper": mean, "ci95_low": low, "ci95_high": high,
                                       "excludes_zero": low > 0 or high < 0})
                print(f"{hour:02}:00 {cohort} complete", flush=True)
    pd.DataFrame(log).to_csv(args.out / "validation_grid.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out / "test_scores.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out / "paired_vs_paper.csv", index=False)
    pd.DataFrame(continuous).to_csv(args.out / "continuous_checks.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(args.out / "test_probabilities.csv.gz", index=False)
    with pd.option_context("display.width", 180, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(paired).to_string(index=False))
        print(pd.DataFrame(continuous).to_string(index=False))


if __name__ == "__main__":
    main()
