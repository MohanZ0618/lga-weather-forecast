"""Does non-stationarity explain the mixed long-history results, and does handling it help? Run after mos.py build.

Forecast models and sensors change over time. mos_history.py found that with GFS/NAM MOS in every year, the
test year preferred training on 2010-2023 while 2024 validation preferred 2021-2023 at 09:00 and 12:00.

1. Drift diagnostics (results/nonstationarity/): forecast error of each guidance source (forecast minus the
   final hourly max, and max(high so far, forecast) minus it) by year and quarter, mean and SD
   (drift_table.csv, drift.png). Version boundaries are drawn only where documented in the project notes:
   NBM text v3.2 until 2020-09-28, v4.0 to v4.3 from 2020-09-29, v5.0 from 2026-05-04 06z (inside the
   test year; nbm_v5.csv checks bias and spread after it). Other model upgrades are not asserted: level
   shifts are detected from the data (level_shifts.csv: the single break in the deseasonalized monthly mean,
   or log SD, with the largest t statistic, p value from shuffling 3-month blocks). Sensor drift: LGA minus
   JFK, EWR and TEB at the same cutoffs, monthly, 2010-2026.
2. Handling drift, every choice on 2024 validation, censored Gaussian throughout (weighted fit below):
   (a) exponential recency weights on training days, half-life 0.5, 1, 2, 4, 8 years or none, for CURRENT's
       inputs (2021-2023) and for the long MOS model (stations + trends + GFS/NAM MOS, 2010-2023), plus the
       two-stage model of mos_history.py with an unweighted and a recency-weighted stage 1;
   (b) era indicators that shift the mean (and optionally log sigma): breaks detected on training years only,
       or calendar periods (4-year blocks for the long model, years for CURRENT's inputs);
   (c) OPERATIONAL, reported separately: walk-forward monthly refits of CURRENT's model through 2024 and the
       test year, using only labels at least 2 days old at the start of each month (so test-year labels of
       earlier months are used), with each half-life and with an NBM v5 indicator.
Scores use the same 359 test days per cutoff as every other result; the paired baseline is the saved CURRENT.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from threadpoolctl import threadpool_limits

import shash
from distribution import band_cdf_edges, design, relative_columns, to_bands
from experiment import HOURS, PROBS, month_prior, split_dates, stabilize
from history_study import GROUPS, SCOPE, load
from mos_history import OBS, add_mos, mos_columns, mos_present
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
L2 = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)
HALF_LIVES = (0.5, 1.0, 2.0, 4.0, 8.0, None)
CURRENT = "Censored Gaussian, HRRR + LAMP/NBM"
LONG = ("gfs", "nam")
SOURCES = {"GFS MOS daytime max": "gfs_max", "NAM MOS daytime max": "nam_max", "NBM daytime max": "nbm_max",
           "LAMP remaining max": "lamp_remaining_max", "HRRR remaining max": "hrrr_remaining_max"}
CURRENT_SOURCES = ("NBM daytime max", "LAMP remaining max", "HRRR remaining max")
LONG_SOURCES = ("GFS MOS daytime max", "NAM MOS daytime max")
NEIGHBORS = ("JFK", "EWR", "TEB")
NEIGHBOUR_MEAN = "LGA minus mean of JFK, EWR, TEB, cutoff temperature"
# Documented in the project notes; no other version dates are asserted.
NBM_V4 = pd.Timestamp("2020-09-29")
NBM_V5 = pd.Timestamp("2026-05-04 06:00")
TRAIN_END = pd.Timestamp("2024-01-01")
LAG = pd.Timedelta(days=2)
HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)
COLORS = {"raw": "#257de0", "floored": "#f47230", "JFK": "#257de0", "EWR": "#f47230", "TEB": "#00b894"}
INK, MUTED, SPAN = "#191919", "#666666", "#ececec"


# Weighted censored Gaussian ---------------------------------------------------------------------------------

def weighted_loss(theta, x, z, rise, l2, w):
    """Mean weighted negative log likelihood of R = max(0, R*), R* ~ N(b0 + x beta, exp(z gamma)), plus L2.

    w has mean 1, so with w = 1 this is exactly the loss shash.fit minimizes with shape=False.
    """
    n, k = x.shape
    mu = theta[0] + x @ theta[1:k + 1]
    log_sigma = z @ theta[k + 1:]
    sigma = np.exp(log_sigma)
    zero = rise <= 1e-9
    t = np.where(zero, -mu / sigma, (rise - mu) / sigma)
    log_cdf = norm.logcdf(t)
    ll = np.where(zero, log_cdf, -0.5 * t * t - log_sigma - HALF_LOG_2PI)
    ratio = np.exp(norm.logpdf(t) - log_cdf)  # phi / Phi for the censored rows
    d_mu = np.where(zero, -ratio / sigma, t / sigma)
    d_log_sigma = np.where(zero, -ratio * t, t * t - 1.0)
    beta = theta[1:k + 1]
    loss = -(w @ ll) / n + l2 * (beta @ beta) / n
    grad = -np.concatenate([[w @ d_mu], x.T @ (w * d_mu), z.T @ (w * d_log_sigma)]) / n
    grad[1:k + 1] += 2 * l2 * beta / n
    return loss, grad


def fit_weighted(x, z, rise, l2, weights=None):
    """Returns theta in shash.py's layout (skew 0, log delta 0), so shash.cdf and shash.parameters apply."""
    n, k = x.shape
    m = z.shape[1]
    w = np.ones(n) if weights is None else np.asarray(weights, float) / np.mean(weights)
    start = np.zeros(1 + k + m)
    start[0] = np.average(rise, weights=w)
    start[k + 1] = np.log(np.sqrt(np.average((rise - start[0]) ** 2, weights=w)) + 1e-3)
    result = minimize(weighted_loss, start, args=(x, z, rise, l2, w), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000})
    return np.r_[result.x, np.zeros(m), 0.0]


def recency(dates, reference, half_life):
    """Weight 0.5 ** (age in years / half-life) for days before `reference`; ones when half_life is None."""
    if half_life is None:
        return np.ones(len(dates))
    age = (pd.Timestamp(reference) - pd.DatetimeIndex(dates)).days.to_numpy() / 365.25
    return 0.5 ** (np.maximum(age, 0) / half_life)


def label(half_life):
    return "no decay" if half_life is None else f"half-life {half_life:g} y"


class Problem:
    def __init__(self, samples, evaluate):
        self.samples = samples
        self.z = np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos])
        self.floor = samples.LGA_max_so_far.to_numpy()
        self.rise = (samples.target_max - samples.LGA_max_so_far).to_numpy()
        self.edges = band_cdf_edges(samples)
        self.target = samples.target_category.to_numpy()
        self.evaluate = evaluate

    def loss(self, p, split="validation"):
        mask = self.evaluate[split]
        return float(daily_scores(p[mask], self.target[mask])["log_loss"].mean())

    def fit(self, columns, train, weights=None, sigma_columns=(), grid=L2, log=None, tag=None):
        """Censored Gaussian over the L2 grid, kept at the lowest 2024 validation log loss."""
        x = design(self.samples, columns, train)
        z = self.z if not sigma_columns else np.column_stack([self.z, design(self.samples, list(sigma_columns), train)])
        prior = month_prior(self.samples.loc[train], self.samples)
        best = None
        for l2 in grid:
            theta = fit_weighted(x[train], z[train], self.rise[train], l2, None if weights is None else weights[train])
            p = stabilize(to_bands(shash.cdf(theta, x, z, self.edges)), prior, self.floor)
            loss = self.loss(p)
            if log is not None:
                log.append({**tag, "l2": l2, "validation_log_loss": loss})
            if best is None or loss < best["validation_log_loss"]:
                mu, log_sigma, _, _ = shash.parameters(theta, x, z)
                best = {"p": p, "l2": l2, "validation_log_loss": loss, "mu": mu, "log_sigma": log_sigma}
        return best


def out_of_fold(problem, columns, window_train, stage2_train, l2, weights):
    """Stage-1 mean and log sigma; rows of each stage-2 training year come from a fit without that year."""
    full = problem.fit(columns, window_train, weights, grid=(l2,))
    mu, log_sigma = full["mu"].copy(), full["log_sigma"].copy()
    years = problem.samples.index.year.to_numpy()
    for year in sorted(set(years[stage2_train])):
        fold = problem.fit(columns, window_train & (years != year), weights, grid=(l2,))
        rows = years == year
        mu[rows], log_sigma[rows] = fold["mu"][rows], fold["log_sigma"][rows]
    return mu, log_sigma


def walk_forward_train(dates, eligible, month_start, lag=LAG):
    """Rows whose label is settled (at least `lag` old) at the start of the forecast month."""
    return eligible & (pd.DatetimeIndex(dates) <= pd.Timestamp(month_start) - lag)


def walk_forward(problem, columns, eligible, months, l2, half_life=None):
    """Refit at the start of every month with the labels available then; predict that month only."""
    dates = problem.samples.index
    p = np.full((len(dates), 5), np.nan)
    for start in months:
        train = walk_forward_train(dates, eligible, start)
        fit = problem.fit(columns, train, recency(dates, start, half_life), grid=(l2,))
        rows = (dates >= start) & (dates < start + pd.offsets.MonthBegin(1))
        p[rows] = fit["p"][rows]
    return p


# Drift diagnostics -----------------------------------------------------------------------------------------

def error_frame(hours):
    frames = []
    for hour, (samples, _) in hours.items():
        floor, target = samples.LGA_max_so_far, samples.target_max
        for name, column in SOURCES.items():
            forecast = samples[column]
            ok = (forecast.notna() & target.notna()).to_numpy()
            frames.append(pd.DataFrame({"date": samples.index[ok], "hour": hour, "source": name,
                                        "error": (forecast - target)[ok].to_numpy(),
                                        "floored": (np.fmax(floor, forecast) - target)[ok].to_numpy()}))
        # An LGA-only change moves LGA against all three neighbours; a neighbour's own change averages out.
        difference = samples.LGA_temp - samples[[f"{s}_temp" for s in NEIGHBORS]].mean(axis=1, skipna=False)
        ok = difference.notna().to_numpy()
        frames.append(pd.DataFrame({"date": samples.index[ok], "hour": hour, "source": NEIGHBOUR_MEAN,
                                    "error": difference[ok].to_numpy(), "floored": np.nan}))
        for station in NEIGHBORS:
            for kind, a, b in (("cutoff temperature", "LGA_temp", f"{station}_temp"),
                               ("previous-day max", "LGA_previous_max", f"{station}_previous_max")):
                difference = samples[a] - samples[b]
                ok = difference.notna().to_numpy()
                frames.append(pd.DataFrame({"date": samples.index[ok], "hour": hour,
                                            "source": f"LGA minus {station}, {kind}",
                                            "error": difference[ok].to_numpy(), "floored": np.nan}))
    return pd.concat(frames, ignore_index=True)


def drift_table(errors):
    rows = []
    for kind, key in (("year", errors.date.dt.year.astype(str)), ("quarter", errors.date.dt.to_period("Q").astype(str))):
        grouped = errors.groupby(["source", "hour", key])
        table = grouped.agg(n=("error", "size"), mean_error=("error", "mean"), sd_error=("error", "std"),
                            mean_floored=("floored", "mean"), sd_floored=("floored", "std")).reset_index()
        rows.append(table.rename(columns={"date": "period"}).assign(period_type=kind))
    table = pd.concat(rows, ignore_index=True)
    return table[["source", "hour", "period_type", "period", "n", "mean_error", "sd_error", "mean_floored", "sd_floored"]]


def monthly_anomaly(frame, statistic="mean", column="error", minimum=20):
    """Monthly mean (or log SD) of daily errors pooled over the cutoffs, minus its calendar-month average."""
    grouped = frame.groupby(frame.date.dt.to_period("M"))[column]
    enough = grouped.count() >= minimum
    value = grouped.mean()[enough] if statistic == "mean" else np.log(grouped.std()[enough])
    return value - value.groupby(value.index.month).transform("mean")


def _max_t(series, min_segment):
    """Largest |t| of a single mean shift, for each row of `series` (draws x months)."""
    series = np.atleast_2d(series)
    length = series.shape[1]
    k = np.arange(min_segment, length - min_segment + 1)
    s1, q1 = np.cumsum(series, axis=1)[:, k - 1], np.cumsum(series ** 2, axis=1)[:, k - 1]
    s, q = series.sum(axis=1, keepdims=True), (series ** 2).sum(axis=1, keepdims=True)
    m1, m2 = s1 / k, (s - s1) / (length - k)
    sse = (q1 - k * m1 ** 2) + (q - q1 - (length - k) * m2 ** 2)
    t = (m2 - m1) / np.sqrt(np.maximum(sse / (length - 2), 1e-12) * (1 / k + 1 / (length - k)))
    best = np.abs(t).argmax(axis=1)
    return k[best], t[np.arange(len(series)), best], m2[np.arange(len(series)), best] - m1[np.arange(len(series)), best]


def level_shift(anomaly, min_segment=6, block=3, draws=2000, seed=4260):
    """Single break with the largest |t|; p value from shuffling whole 3-month blocks (keeps short memory)."""
    values = anomaly.to_numpy(float)
    k, t, shift = _max_t(values, min_segment)
    usable = len(values) // block * block
    blocks = values[:usable].reshape(-1, block)
    rng = np.random.default_rng(seed)
    null = np.stack([blocks[rng.permutation(len(blocks))].ravel() for _ in range(draws)])
    _, null_t, _ = _max_t(null, min_segment)
    p = (1 + np.sum(np.abs(null_t) >= abs(t[0]))) / (1 + draws)
    return {"break_month": str(anomaly.index[k[0]]), "shift": float(shift[0]), "t": float(t[0]), "p": float(p),
            "months": len(values), "first_month": str(anomaly.index[0]), "last_month": str(anomaly.index[-1])}


def detect(errors, sources, first, last, data_label):
    rows = []
    span = errors.loc[(errors.date >= first) & (errors.date < last)]
    for source in sources:
        frame = span.loc[span.source.eq(source)]
        for statistic in ("mean", "log SD"):
            anomaly = monthly_anomaly(frame, "mean" if statistic == "mean" else "sd")
            if len(anomaly) >= 18:
                rows.append({"series": source, "statistic": statistic, "data": data_label, **level_shift(anomaly)})
    return rows


def two_sample_block(a_dates, a, b_dates, b, block_days=7, draws=5000, seed=4260):
    """95% intervals for mean(a) - mean(b) and sd(a) - sd(b), resampling 7-day blocks within each sample."""
    rng = np.random.default_rng(seed)

    def blocks(dates, values):
        ids = np.asarray((pd.DatetimeIndex(dates) - pd.DatetimeIndex(dates).min()).days // block_days)
        return [values[ids == i] for i in np.unique(ids)]

    def draw(groups):
        picks = rng.integers(0, len(groups), size=len(groups))
        return np.concatenate([groups[i] for i in picks])

    ga, gb = blocks(a_dates, a), blocks(b_dates, b)
    means, sds = [], []
    for _ in range(draws):
        x, y = draw(ga), draw(gb)
        means.append(x.mean() - y.mean())
        sds.append(x.std(ddof=1) - y.std(ddof=1))
    return np.quantile(means, [0.025, 0.975]), np.quantile(sds, [0.025, 0.975])


def nbm_v5_check(hours, errors):
    """NBM error after the v5.0 switch versus the same calendar window (May 4 to Aug 31) in 2021-2025."""
    rows = []
    for hour, (samples, _) in hours.items():
        version5 = samples.nbm_run.ge(NBM_V5)
        for source in ("NBM daytime max", "GFS MOS daytime max", "LAMP remaining max"):
            frame = errors.loc[errors.source.eq(source) & errors.hour.eq(hour)].set_index("date")
            window = (frame.index.month.isin([5, 6, 7, 8])) & ~((frame.index.month == 5) & (frame.index.day < 4))
            after = window & (frame.index >= "2026-05-04")
            if source == "NBM daytime max":
                after &= version5.reindex(frame.index).fillna(False).astype(bool).to_numpy()
            before = window & (frame.index.year >= 2021) & (frame.index.year <= 2025)
            a, b = frame.loc[after, "error"], frame.loc[before, "error"]
            (mlo, mhi), (slo, shi) = two_sample_block(a.index, a.to_numpy(), b.index, b.to_numpy())
            rows.append({"hour": hour, "source": source, "n_2026_after_v5": len(a), "n_2021_2025_same_window": len(b),
                         "mean_2026": a.mean(), "mean_2021_2025": b.mean(), "mean_shift": a.mean() - b.mean(),
                         "mean_shift_ci95_low": mlo, "mean_shift_ci95_high": mhi, "sd_2026": a.std(),
                         "sd_2021_2025": b.std(), "sd_shift": a.std() - b.std(), "sd_shift_ci95_low": slo,
                         "sd_shift_ci95_high": shi})
    return rows


def plot_drift(errors, shifts, path, hour=9):
    names = list(SOURCES) + ["LGA minus neighbours, cutoff temperature"]
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 2.0 * len(names)), sharex=True)
    detected = {(r["series"], r["statistic"]): r for r in shifts if r["data"].startswith("2010-2026") and r["p"] < 0.05}
    for row, name in enumerate(names):
        for col, statistic in enumerate(("mean", "SD")):
            ax = axes[row, col]
            ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), color=SPAN, zorder=0, lw=0)
            ax.axvspan(pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-01"), color=SPAN, zorder=0, lw=0)
            if name.startswith("LGA minus"):
                series = [(f"LGA minus {s}", COLORS[s], errors.source.eq(f"LGA minus {s}, cutoff temperature"), "error")
                          for s in NEIGHBORS] + [("LGA minus their mean", INK, errors.source.eq(NEIGHBOUR_MEAN), "error")]
                frame_hour = errors
            else:
                series = [("forecast minus final max", COLORS["raw"], errors.source.eq(name), "error"),
                          ("max(high so far, forecast) minus final max", COLORS["floored"], errors.source.eq(name), "floored")]
                frame_hour = errors.loc[errors.hour.eq(hour)]
            for text, color, mask, column in series:
                frame = frame_hour.loc[mask.reindex(frame_hour.index)]
                for freq, width, alpha in (("Q", 1.4, 1.0), ("Y", 2.4, 0.35)):
                    grouped = frame.groupby(frame.date.dt.to_period(freq))[column]
                    value = grouped.mean() if statistic == "mean" else grouped.std()
                    value = value[grouped.count() >= 20]
                    if freq == "Q":
                        ax.plot(value.index.to_timestamp(how="start") + pd.Timedelta(days=45), value.to_numpy(),
                                color=color, lw=width, marker="o", ms=2.5, label=text if row in (0, len(names) - 1) else None)
                    else:
                        for period, v in value.items():
                            ax.hlines(v, period.start_time, period.end_time, color=color, lw=width, alpha=alpha)
            if name == "NBM daytime max":
                for stamp, text in ((NBM_V4, "v4.0"), (NBM_V5, "v5.0")):
                    ax.axvline(stamp, color=INK, lw=1.0)
                    ax.text(stamp, 1.0, f" {text}", transform=ax.get_xaxis_transform(), va="top", fontsize=7, color=INK)
            keys = [NEIGHBOUR_MEAN] if name.startswith("LGA minus") else [name]
            for key in keys:
                hit = detected.get((key, "mean" if statistic == "mean" else "log SD"))
                if hit:
                    stamp = pd.Period(hit["break_month"]).start_time
                    ax.axvline(stamp, color=MUTED, lw=1.0, ls="--")
                    ax.text(stamp, 0.02, " data-detected", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED)
            ax.axhline(0, color=MUTED, lw=0.6) if statistic == "mean" else None
            ax.grid(axis="y", color="#e2e2e2", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.set_title(f"{name}: {statistic} (F)" + ("" if name.startswith("LGA minus") else f", {hour:02}:00 cutoff"),
                         fontsize=8.5, color=INK, loc="left")
    for ax, anchor in ((axes[0, 0], (0.01, 0.995)), (axes[-1, 0], (0.5, 0.995))):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=anchor,
                   ncol=2, handlelength=1.6)
    fig.text(0.01, 0.003, "Quarterly (lines) and yearly (bands) values. Grey: 2024 validation and the test year. "
             "Solid lines: documented NBM versions; dashed: data-detected level shift (p < 0.05).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.012, 1, 0.968))
    fig.savefig(path, dpi=170)
    plt.close(fig)


# Experiments -----------------------------------------------------------------------------------------------

def era_features(samples, breaks):
    """Indicator = 1 before a training-years break (older era); validation and test are in the latest era."""
    dates, years = samples.index, samples.index.year
    out = {}
    for source, start in breaks.items():
        out[f"era_{SOURCES[source]}"] = (dates < start).astype(float)
    periods = {"period_2010_2013": (years <= 2013), "period_2014_2017": (years >= 2014) & (years <= 2017),
               "period_2018_2020": (years >= 2018) & (years <= 2020), "year_2021": years == 2021, "year_2022": years == 2022}
    out.update({k: v.astype(float) for k, v in periods.items()})
    return samples.assign(**out)


def run_hour(hour, samples, complete, saved, breaks, log):
    splits = split_dates(samples.index)
    evaluate = {"validation": splits["validation"] & complete, "test": splits["test"] & complete}
    samples = era_features(samples, {**breaks["long"], **breaks["current"]})
    samples["nbm_v5"] = samples.nbm_run.ge(NBM_V5).astype(float).to_numpy()
    problem = Problem(samples, evaluate)
    dates, years = samples.index, samples.index.year.to_numpy()
    test = evaluate["test"]
    current_columns = relative_columns(samples, SCOPE) + list(GROUPS)
    long_columns = relative_columns(samples, OBS) + mos_columns(LONG)
    short = splits["train"] & complete & (years >= 2021)
    long_train = splits["train"] & (years >= 2010) & mos_present(samples, LONG)
    systems = {}

    def add(name, part, fit, train, note=""):
        systems[name] = {"p": fit["p"], "part": part, "l2": fit.get("l2"), "train_rows": int(train.sum()),
                         "validation_log_loss": fit["validation_log_loss"], "note": note}

    def tag(name):
        return {"hour": hour, "system": name}

    current_fits, long_fits = {}, {}
    for h in HALF_LIVES:
        weights = recency(dates, TRAIN_END, h)
        name = f"CURRENT inputs 2021-2023, {label(h)}" + (" (CURRENT refit)" if h is None else "")
        current_fits[h] = problem.fit(current_columns, short, weights, log=log, tag=tag(name))
        add(name, "a recency", current_fits[h], short)
        name = f"Long MOS model 2010-2023, {label(h)}"
        long_fits[h] = problem.fit(long_columns, long_train, weights, log=log, tag=tag(name))
        add(name, "a recency", long_fits[h], long_train)
    best_long = min(HALF_LIVES, key=lambda h: long_fits[h]["validation_log_loss"])
    for h in dict.fromkeys([None, best_long]):
        mu, log_sigma = out_of_fold(problem, long_columns, long_train, short, long_fits[h]["l2"],
                                    recency(dates, TRAIN_END, h))
        staged = Problem(samples.assign(stage1_mu=mu, stage1_log_sigma=log_sigma), evaluate)
        name = f"Two-stage, stage 1 long MOS model with {label(h)}"
        add(name, "a recency", staged.fit(current_columns + ["stage1_mu", "stage1_log_sigma"], short, log=log,
                                          tag=tag(name)), short, f"stage-1 l2={long_fits[h]['l2']}")

    families = (("Long MOS model 2010-2023", long_columns, long_train,
                 {"data-detected eras": [f"era_{SOURCES[s]}" for s in breaks["long"]],
                  "4-year periods": ["period_2010_2013", "period_2014_2017", "period_2018_2020"]}),
                ("CURRENT inputs 2021-2023", current_columns, short,
                 {"data-detected eras": [f"era_{SOURCES[s]}" for s in breaks["current"]],
                  "year indicators": ["year_2021", "year_2022"]}))
    for base, columns, train, options in families:
        for option, flags in options.items():
            if not flags:
                continue
            for where, sigma in (("mean", ()), ("mean and log sigma", flags)):
                name = f"{base} + {option} ({where})"
                add(name, "b eras", problem.fit(columns + flags, train, sigma_columns=sigma, log=log, tag=tag(name)),
                    train, ", ".join(flags))

    # (c) Operational walk-forward: refit monthly with settled labels, including earlier test-year months.
    eligible = complete & (years >= 2021)
    months = list(pd.date_range("2024-01-01", "2024-12-01", freq="MS")) + list(
        pd.date_range("2025-09-01", "2026-08-01", freq="MS"))
    for h in HALF_LIVES:
        p = walk_forward(problem, current_columns, eligible, months, current_fits[h]["l2"], h)
        add(f"Walk-forward CURRENT, {label(h)}", "c walk-forward",
            {"p": p, "l2": current_fits[h]["l2"], "validation_log_loss": problem.loss(p)}, eligible)
    p = walk_forward(problem, current_columns + ["nbm_v5"], eligible, months, current_fits[None]["l2"])
    add("Walk-forward CURRENT + NBM v5 indicator, no decay", "c walk-forward",
        {"p": p, "l2": current_fits[None]["l2"], "validation_log_loss": problem.loss(p)}, eligible,
        "indicator is 0 in every 2024 refit, so validation cannot tell it from the plain walk-forward")

    target = problem.target[test]
    test_dates = dates[test]
    base = saved.loc[saved.hour.eq(hour)].set_index("date").loc[test_dates, PROBS].to_numpy()
    scores = {"CURRENT (saved)": daily_scores(base, target)}
    rows = [{"hour": hour, "system": "CURRENT (saved)", "part": "baseline", "n": len(target),
             "validation_log_loss": current_fits[None]["validation_log_loss"], **scores["CURRENT (saved)"].mean().to_dict()}]
    paired, frames = [], []
    for name, s in systems.items():
        scores[name] = daily_scores(s["p"][test], target)
        rows.append({"hour": hour, "system": name, "part": s["part"], "n": len(target),
                     "validation_log_loss": s["validation_log_loss"], **scores[name].mean().to_dict(),
                     "l2": s["l2"], "train_rows": s["train_rows"], "note": s["note"]})
        frames.append(pd.DataFrame(s["p"][test], columns=PROBS).assign(system=name, hour=hour, date=test_dates,
                                                                       target_category=target))
        for metric in ("log_loss", "rps"):
            diff = (scores[name][metric] - scores["CURRENT (saved)"][metric]).to_numpy()
            mean, low, high = block_bootstrap(test_dates, diff)
            paired.append({"hour": hour, "candidate": name, "metric": metric, "delta_vs_current": mean,
                           "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
    fixed = [n for n, s in systems.items() if not s["part"].startswith("c")]
    walking = [n for n, s in systems.items() if s["part"].startswith("c")]
    selections = []
    for pool, kind in ((fixed, "fixed fits (protocol)"), (walking + ["CURRENT inputs 2021-2023, no decay (CURRENT refit)"],
                                                         "walk-forward allowed (operational)")):
        chosen = min(pool, key=lambda n: systems[n]["validation_log_loss"])
        delta = next(r for r in paired if r["candidate"] == chosen and r["metric"] == "log_loss")
        selections.append({"hour": hour, "pool": kind, "selected": chosen,
                           "validation_log_loss": systems[chosen]["validation_log_loss"],
                           "test_log_loss": float(scores[chosen]["log_loss"].mean()),
                           "delta_vs_current": delta["delta_vs_current"], "ci95_low": delta["ci95_low"],
                           "ci95_high": delta["ci95_high"], "best_long_half_life": label(best_long)})
    reproduction = float(np.abs(systems["CURRENT inputs 2021-2023, no decay (CURRENT refit)"]["p"][test] - base).max())
    for s in selections:
        s["current_refit_max_abs_prob_diff_vs_saved"] = reproduction
    return rows, paired, selections, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "nonstationarity")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    mos = pd.read_csv(args.data / "mos_features.csv.gz", parse_dates=["date"])
    saved = pd.read_csv(ROOT / "results" / "guidance" / "test_probabilities.csv.gz", parse_dates=["date"])
    saved = saved.loc[saved.system.eq(CURRENT)]
    guidance["nbm_run"] = pd.to_datetime(guidance.nbm_run, utc=True).dt.tz_localize(None)
    hours = {}
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples, complete = load(args.data, hour, hrrr, guidance)
            samples = add_mos(samples, mos, hour)
            samples["nbm_run"] = guidance.loc[guidance.hour.eq(hour)].set_index("date").nbm_run.reindex(samples.index)
            hours[hour] = (samples, complete)

        errors = error_frame(hours)
        drift_table(errors).to_csv(args.out / "drift_table.csv", index=False, float_format="%.4f")
        sensors = [NEIGHBOUR_MEAN] + [f"LGA minus {s}, {kind}" for s in NEIGHBORS
                                      for kind in ("cutoff temperature", "previous-day max")]
        shifts = detect(errors, list(SOURCES) + sensors, "2010-01-01", "2026-09-01", "2010-2026 (descriptive)")
        training = detect(errors, LONG_SOURCES, "2010-01-01", "2024-01-01", "2010-2023 (training, long model eras)")
        training += detect(errors, CURRENT_SOURCES, "2021-01-01", "2024-01-01", "2021-2023 (training, CURRENT eras)")
        pd.DataFrame(shifts + training).to_csv(args.out / "level_shifts.csv", index=False, float_format="%.4f")
        pd.DataFrame(nbm_v5_check(hours, errors)).to_csv(args.out / "nbm_v5.csv", index=False, float_format="%.4f")
        plot_drift(errors, shifts, args.out / "drift.png")
        # Era features use only breaks found in training years, in the mean, at p < 0.05.
        breaks = {"long": {}, "current": {}}
        for r in training:
            if r["statistic"] == "mean" and r["p"] < 0.05:
                breaks["long" if r["series"] in LONG_SOURCES else "current"][r["series"]] = pd.Period(r["break_month"]).start_time
        print(f"diagnostics done ({time.time() - started:.0f} s); training-year breaks {breaks}", flush=True)

        log, rows, paired, selections, frames = [], [], [], [], []
        for hour in HOURS:
            out = run_hour(hour, *hours[hour], saved, breaks, log)
            rows += out[0]
            paired += out[1]
            selections += out[2]
            frames += out[3]
            print(f"{hour:02}:00 done ({time.time() - started:.0f} s)", flush=True)
    for name, table in (("validation_grid", log), ("test_scores", rows), ("paired_vs_current", paired),
                        ("selection", selections)):
        pd.DataFrame(table).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(args.out / "test_probabilities.csv.gz", index=False,
                                                compression={"method": "gzip", "mtime": 0})
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 400,
                           "display.max_colwidth", 70):
        print(pd.DataFrame(rows).drop(columns=["note"]).to_string(index=False))
        print(pd.DataFrame(selections).to_string(index=False))
    print(f"total {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
