"""Longer training history and a skewed, heavy-tailed shape for the guidance model.

Inputs are the selected guidance scope (four stations + trends + HRRR + LAMP/NBM). Three training windows,
all ending 2023-12-31:
  2021-2023  complete rows only (the setup of guidance_study.py)
  2018-2023  every bundled date; a source that did not exist yet is imputed and flagged as missing
             (NBM from November 2018, LAMP from July 2020, HRRR cache from 2021)
  2010-2023  the same plus 2010-2017 rows built by history.py with the paper's own pipeline
Models per window: censored Gaussian, censored SHASH (shash.py), quantile boosting, and the paper's rule of
averaging the two best by validation. One candidate per cutoff is chosen on 2024 validation log loss.
Validation and test use only dates with every source, so all windows are scored on the same 359 test days
as the paper's selected HRRR system. Writes results/history/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

import shash
from distribution import SCALES, TAUS, add_relative, band_cdf_edges, design, quantile_cdf, relative_columns, scaled, to_bands
from experiment import FORECAST_FEATURES, HOURS, PROBS, month_prior, season_features, split_dates, stabilize
from guidance_study import PAPER, REQUIRED
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
SCOPE = "Four stations + trends + HRRR + LAMP/NBM"
WINDOWS = {"2021-2023": "2021-01-01", "2018-2023": "2018-01-01", "2010-2023": "2010-01-01"}
# A source counts as missing when its raw values are absent; its derived columns are then blanked too.
GROUPS = {"hrrr_missing": (["hrrr_remaining_max", "hrrr_warming"], ["rel_hrrr"]),
          "lamp_missing": (["lamp_remaining_max", "lamp_now", "lamp_cloud", "lamp_u", "lamp_v"],
                           ["rel_lamp", "lamp_now_error"]),
          "nbm_missing": (["nbm_max", "nbm_max_sd"], ["rel_nbm"])}
OBSERVATIONS = "Four stations + trends"
L2 = (1.0, 10.0, 30.0)


def load(data, hour, hrrr, guidance):
    samples = pd.read_csv(data / f"samples_{hour:02}.csv.gz", parse_dates=["date"], index_col="date")
    early = data / "history" / f"samples_{hour:02}_2010_2017.csv.gz"
    if early.exists():
        early = pd.read_csv(early, parse_dates=["date"], index_col="date")
        assert early.index.max() < samples.index.min()
        samples = pd.concat([early[samples.columns], samples])
    samples = season_features(samples.sort_index())
    samples = samples.join(hrrr.loc[hrrr.hour == hour].set_index("date")[FORECAST_FEATURES], how="left")
    samples = add_relative(samples.join(guidance.loc[guidance.hour == hour].set_index("date")[REQUIRED], how="left"))
    for flag, (raw, derived) in GROUPS.items():
        missing = samples[raw].isna().any(axis=1)
        samples.loc[missing, raw + derived] = np.nan
        samples[flag] = missing.astype(float)
    complete = samples[list(GROUPS)].sum(axis=1).eq(0).to_numpy()
    return samples, complete


def fit_window(samples, train, evaluate, log, hour, window, scope=SCOPE):
    """Return {model: band probabilities for every row} and {model: 80% interval of the final max}."""
    columns = relative_columns(samples, scope) + (list(GROUPS) if scope == SCOPE else [])
    x = design(samples, columns, train)
    z = np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos])
    prior = month_prior(samples.loc[train], samples)
    floor = samples.LGA_max_so_far.to_numpy()
    rise = (samples.target_max - samples.LGA_max_so_far).to_numpy()
    edges = band_cdf_edges(samples)
    target = samples.target_category.to_numpy()
    validation = evaluate["validation"]

    def val_loss(p):
        return daily_scores(p[validation], target[validation])["log_loss"].mean()

    best, intervals = {}, {}
    for l2 in L2:
        gauss = shash.fit(x[train], z[train], rise[train], l2, shape=False)
        skewed = shash.fit(x[train], z[train], rise[train], l2, shape=True, start=gauss)
        for name, theta in (("Censored Gaussian", gauss), ("Censored SHASH", skewed)):
            p = stabilize(to_bands(shash.cdf(theta, x, z, edges)), prior, floor)
            loss = val_loss(p)
            log.append({"hour": hour, "window": window, "model": name, "setting": f"l2={l2}", "validation_log_loss": loss})
            if name not in best or loss < best[name][0]:
                best[name] = (loss, p)
                intervals[name] = (floor + shash.quantile(theta, x, z, 0.1), floor + shash.quantile(theta, x, z, 0.9))
                if name == "Censored SHASH":
                    eps = (z @ theta[x.shape[1] + 4:x.shape[1] + 7])[evaluate["test"]]
                    log[-1].update(eps_mean=float(eps.mean()), delta=float(np.exp(theta[-1])))
    for depth in (2, 3, 4):
        model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=TAUS, n_estimators=300, learning_rate=0.05,
                             max_depth=depth, min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
                             random_state=4260, n_jobs=2)
        model.fit(x[train], rise[train])
        q = model.predict(x)
        for s in SCALES:
            p = stabilize(to_bands(quantile_cdf(scaled(q, s), edges)), prior, floor)
            loss = val_loss(p)
            log.append({"hour": hour, "window": window, "model": "Quantile boosting",
                        "setting": f"depth={depth}, scale={s}", "validation_log_loss": loss})
            if "Quantile boosting" not in best or loss < best["Quantile boosting"][0]:
                best["Quantile boosting"] = (loss, p)
                qs = np.maximum(np.sort(scaled(q, s), axis=1), 0)
                intervals["Quantile boosting"] = (floor + qs[:, 2], floor + qs[:, 6])
    top = sorted(best, key=lambda name: best[name][0])[:2]
    blend = np.mean([best[name][1] for name in top], axis=0)
    log.append({"hour": hour, "window": window, "model": "Two-model average", "setting": " + ".join(top),
                "validation_log_loss": val_loss(blend)})
    out = {name: value[1] for name, value in best.items()}
    out["Two-model average"] = blend
    return out, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "history")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paper = pd.read_csv(ROOT / "results" / "daily_predictions.csv", parse_dates=["date"])
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    log, rows, paired, selections, coverage, calibration = [], [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples, complete = load(args.data, hour, hrrr, guidance)
            splits = split_dates(samples.index)
            evaluate = {"validation": splits["validation"] & complete, "test": splits["test"] & complete}
            test = evaluate["test"]
            dates = samples.index[test]
            target = samples.target_category.to_numpy()
            y = samples.target_max.to_numpy()
            reference = paper.loc[paper.cohort.eq("Matched HRRR") & paper.hour.eq(hour) & paper.scope.eq(PAPER)]
            systems = {"Paper: selected HRRR system": reference.set_index("date").loc[dates, PROBS].to_numpy()}
            val_loss = {}
            for window, start in WINDOWS.items():
                train = splits["train"] & (samples.index >= start)
                if window == "2021-2023":
                    train = train & complete
                coverage.append({"hour": hour, "window": window, "train_dates": int(train.sum()),
                                 **{f"train_{flag}": int(samples.loc[train, flag].sum()) for flag in GROUPS},
                                 "validation_dates": int(evaluate["validation"].sum()), "test_dates": int(test.sum())})
                models, intervals = fit_window(samples, train, evaluate, log, hour, window)
                for name, p in models.items():
                    key = f"{name}, {window}"
                    systems[key] = p[test]
                    val_loss[key] = daily_scores(p[evaluate["validation"]], target[evaluate["validation"]])["log_loss"].mean()
                    if name in intervals:
                        lo, hi = intervals[name]
                        calibration.append({"hour": hour, "system": key,
                                            "test_80_coverage": float(((y >= lo) & (y <= hi))[test].mean()),
                                            "test_80_width_f": float((hi - lo)[test].mean())})
            # Clean history test: observation inputs exist for every year, so nothing is imputed.
            for window in ("2018-2023", "2010-2023"):
                train = splits["train"] & (samples.index >= WINDOWS[window])
                models, _ = fit_window(samples, train, evaluate, log, hour, f"{window} observations only", OBSERVATIONS)
                for name, p in models.items():
                    key = f"{name}, {window} observations only"
                    systems[key] = p[test]
                    val_loss[key] = daily_scores(p[evaluate["validation"]], target[evaluate["validation"]])["log_loss"].mean()
            chosen = min(val_loss, key=val_loss.get)
            selections.append({"hour": hour, "selected": chosen, "validation_log_loss": val_loss[chosen]})
            scores = {name: daily_scores(p, target[test]) for name, p in systems.items()}
            for name in systems:
                rows.append({"hour": hour, "system": name, "validation_log_loss": val_loss.get(name, np.nan),
                             "n": int(test.sum()), **scores[name].mean().to_dict()})
            for baseline, candidate in (("Paper: selected HRRR system", chosen),
                                        ("Censored Gaussian, 2021-2023", chosen),
                                        ("Censored Gaussian, 2021-2023", "Censored Gaussian, 2010-2023"),
                                        ("Censored Gaussian, 2021-2023", "Censored SHASH, 2021-2023"),
                                        ("Censored Gaussian, 2018-2023 observations only",
                                         "Censored Gaussian, 2010-2023 observations only"),
                                        ("Quantile boosting, 2018-2023 observations only",
                                         "Quantile boosting, 2010-2023 observations only")):
                if baseline == candidate:
                    continue
                for metric in ("log_loss", "rps", "correct"):
                    diff = (scores[candidate][metric] - scores[baseline][metric]).to_numpy()
                    mean, low, high = block_bootstrap(dates, diff)
                    paired.append({"hour": hour, "baseline": baseline, "candidate": candidate, "metric": metric,
                                   "delta": mean, "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
            print(f"{hour:02}:00 complete, selected {chosen}", flush=True)
    for name, table in (("coverage", coverage), ("validation_grid", log), ("test_scores", rows), ("paired", paired),
                        ("selection", selections), ("interval_calibration", calibration)):
        pd.DataFrame(table).to_csv(args.out / f"{name}.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format, "display.max_rows", 200):
        print(pd.DataFrame(coverage).to_string(index=False))
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
