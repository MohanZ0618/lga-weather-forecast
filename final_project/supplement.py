"""Supplementary checks for the CS4260 paper: ordinal scoring, paired intervals, raw HRRR baseline.

Run after experiment.py. Reads results/daily_predictions.csv and the bundled samples, writes
results/supplement_scores.csv, results/supplement_paired.csv and results/supplement_hrrr.csv.

1. Ranked probability score (RPS). The five bands are ordered. Log loss and accuracy treat a
   one-band miss the same as a four-band miss; RPS compares cumulative probabilities, so mass
   placed far from the observed band costs more (Epstein 1969; Gneiting and Raftery 2007).
2. Paired 7-day block bootstrap. Daily scores are autocorrelated (weather regimes last days), so
   the paired differences in paired_comparisons.csv get 95% intervals that resample whole weeks.
3. Raw HRRR baseline. Does the learned model beat the forecast it consumes? The baseline takes
   max(observed high so far, HRRR remaining max + bias) with a Gaussian error whose bias and spread
   are fit on training days only, then truncates below the observed floor.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiment import BOUNDS, CLASSES, HOURS, PROBS, split_dates

ROOT = Path(__file__).resolve().parent
COMPARISONS = (
    ("Observations", "LGA", "Four stations"),
    ("Observations", "Four stations", "Four stations + trends"),
    ("Observations", "Remaining warming", "Four stations + trends"),
    ("Matched HRRR", "Four stations + trends", "Four stations + trends + HRRR"),
    ("Matched HRRR", "Raw HRRR", "Four stations + trends + HRRR"),
)


def rps(probabilities, target):
    """Sum over the four thresholds of (forecast CDF - observed CDF) squared."""
    forecast = np.cumsum(probabilities, axis=1)[:, :-1]
    observed = (CLASSES[None, :-1] >= np.asarray(target)[:, None]).astype(float)
    return ((forecast - observed) ** 2).sum(axis=1)


def daily_scores(probabilities, target):
    target = np.asarray(target, dtype=int)
    chosen = probabilities[np.arange(len(target)), target]
    return pd.DataFrame({
        "log_loss": -np.log(np.clip(chosen, 1e-15, None)),
        "rps": rps(probabilities, target),
        "correct": (probabilities.argmax(axis=1) == target).astype(float),
    })


def block_bootstrap(dates, difference, block_days=7, draws=5000, seed=4260):
    """95% interval for the mean paired difference, resampling 7-calendar-day blocks."""
    dates = pd.DatetimeIndex(dates)
    blocks = np.asarray((dates - dates.min()).days // block_days)
    ids, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ids), size=(draws, len(ids)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def band_probabilities(center, sigma, floor):
    """Gaussian bands, with mass below the observed high so far removed and renormalized."""
    edges = np.concatenate([[-np.inf], BOUNDS, [np.inf]])
    lower = np.maximum(edges[None, :-1], floor[:, None])
    upper = np.maximum(edges[None, 1:], floor[:, None])
    z = lambda x: norm.cdf((x - center[:, None]) / sigma[:, None])
    probabilities = np.clip(z(upper) - z(lower), 0, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def raw_hrrr(data):
    """Fit bias and spread on 2021-2023 per hour and season, score validation and test."""
    forecast = pd.read_csv(data / "hrrr_features.csv.gz", parse_dates=["date"])
    rows, daily = [], []
    for hour in HOURS:
        samples = pd.read_csv(data / f"samples_{hour:02}.csv.gz", parse_dates=["date"], index_col="date")
        samples = samples.join(forecast.loc[forecast.hour == hour].set_index("date")[["hrrr_remaining_max"]],
                               how="inner", validate="one_to_one").sort_index()
        splits = split_dates(samples.index)
        floor = samples["LGA_max_so_far"].to_numpy()
        guess = np.fmax(samples["hrrr_remaining_max"].to_numpy(), np.nan_to_num(floor, nan=-np.inf))
        error = samples["target_max"].to_numpy() - guess
        season = (samples.index.month % 12) // 3  # DJF, MAM, JJA, SON
        center, sigma = np.empty(len(samples)), np.empty(len(samples))
        for s in range(4):
            fit = splits["train"] & (season == s)
            center[season == s] = guess[season == s] + error[fit].mean()
            sigma[season == s] = max(error[fit].std(ddof=1), 0.5)
        probabilities = band_probabilities(center, sigma, np.nan_to_num(floor, nan=-np.inf))
        target = samples["target_category"].to_numpy()
        for split in ("validation", "test"):
            mask = splits[split]
            scores = daily_scores(probabilities[mask], target[mask]).mean()
            rows.append({"hour": hour, "split": split, "n": int(mask.sum()), **scores.to_dict()})
        mask = splits["test"]
        frame = samples.loc[mask, ["target_max", "target_category", "LGA_max_so_far"]].reset_index()
        frame[PROBS] = probabilities[mask]
        daily.append(frame.assign(hour=hour, cohort="Matched HRRR", scope="Raw HRRR", model="Raw HRRR"))
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    daily = pd.read_csv(args.out / "daily_predictions.csv", parse_dates=["date"])
    hrrr_table, hrrr_daily = raw_hrrr(args.data)
    hrrr_table.to_csv(args.out / "supplement_hrrr.csv", index=False)
    daily = pd.concat([daily, hrrr_daily], ignore_index=True)
    scored = daily_scores(daily[PROBS].to_numpy(), daily["target_category"])
    daily = pd.concat([daily.reset_index(drop=True), scored.add_prefix("s_")], axis=1)

    keys = ["hour", "cohort", "scope", "model"]
    summary = daily.groupby(keys, sort=False).agg(
        n=("s_rps", "size"), log_loss=("s_log_loss", "mean"), rps=("s_rps", "mean"),
        accuracy=("s_correct", "mean")).reset_index()
    summary.sort_values(["hour", "cohort"], kind="stable").to_csv(args.out / "supplement_scores.csv", index=False)

    paired = []
    for cohort, baseline, candidate in COMPARISONS:
        for hour in HOURS:
            match = daily.loc[daily.cohort.eq(cohort) & daily.hour.eq(hour)]
            for metric in ("s_log_loss", "s_rps", "s_correct"):
                table = match.pivot(index="date", columns="scope", values=metric)
                if baseline not in table or candidate not in table:
                    continue
                both = table[[baseline, candidate]].dropna()
                mean, low, high = block_bootstrap(both.index, (both[candidate] - both[baseline]).to_numpy())
                paired.append({"cohort": cohort, "hour": hour, "baseline": baseline, "candidate": candidate,
                               "metric": metric[2:], "n": len(both), "delta": mean,
                               "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
    paired = pd.DataFrame(paired)
    paired.to_csv(args.out / "supplement_paired.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
