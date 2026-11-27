"""Model evaluation in the vocabulary of Lecture 12: learning curves, rolling-origin cross-validation,
the L2 regularization path and train / validation / test gaps.

CURRENT is the censored Gaussian of history_study.py: scope SCOPE (four stations + trends + HRRR +
LAMP/NBM), trained on the 2021-2023 complete rows, L2 chosen from (1, 10, 30) on 2024 validation.
Every choice below still uses validation years only. The test year (2025-09-01 to 2026-08-31, the 359
complete dates per cutoff) is scored after a choice is made and never picks anything.

1. Learning curves. Train on the most recent N training days with hyperparameters fixed at their
   full-data choice; report training-set and 2024 validation log loss. Two scopes: the guidance scope
   (2021-2023, at most 1040 days) and observations only (2010-2023, up to 5108 days), each for the
   censored Gaussian (one linear mean) and quantile boosting (trees, the flexible model).
2. Rolling-origin cross-validation. Folds train on every complete year before a held-out year:
   2021 -> 2022, 2021-2022 -> 2023, 2021-2023 -> 2024 (the last fold is the single 2024 split).
   The Gaussian's L2 and the model (Gaussian, SHASH, quantile boosting and the paper's five classifier
   families on the same inputs) are chosen by the mean fold log loss, then refit on 2021-2023 and
   scored on test. Classifiers get the same median-imputed, standardized design matrix as the
   distributional models.
3. Regularization path: training, 2024 validation, cross-validation and test log loss against L2.
4. Overfitting check: train vs validation vs test for CURRENT, the other families on its inputs, and
   the paper's classifier families refit exactly as experiment.py does on its matched-HRRR cohort.

Offline, about 100 seconds on 2 threads. Writes results/class_eval/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

import shash
from distribution import SCALES, TAUS, band_cdf_edges, design, quantile_cdf, relative_columns, scaled, to_bands
from experiment import (FORECAST_FEATURES, HOURS, PROBS, candidates, fit_scope, month_prior, season_features,
                        split_dates, stabilize)
from guidance_study import PAPER
from history_study import GROUPS, L2 as CURRENT_L2, OBSERVATIONS, SCOPE, load
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
FIRST = "2021-01-01"   # complete rows (HRRR, LAMP and NBM all present) start here
LONG = "2010-01-01"    # observations-only history from history.py
TEST_START = pd.Timestamp("2025-09-01")
CV_YEARS = (2022, 2023, 2024)
PATH_L2 = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
DEPTHS = (2, 3, 4)
GUIDANCE_N = (100, 200, 400, 700, None)          # None = every training day
LONG_N = (100, 200, 400, 700, 1040, 2000, 3500, None)
CLASSIFIERS = {"Logistic regression": "Logistic regression", "KNN": "KNN", "Gaussian naive Bayes": "Gaussian naive Bayes",
               "Random forest": "Random forest", "XGBoost": "XGBoost classifier"}
REFERENCE = {9: 0.1995, 12: 0.1585, 15: 0.0724}  # CURRENT test log loss quoted in the brief
TOLERANCE = 0.003
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#257de0", "#f47230", "#00b894", "#7d8994", "#191919", "#666666"


# Splits ---------------------------------------------------------------------------------------

def rolling_folds(index, eligible, years=CV_YEARS, start=FIRST):
    """Time-ordered folds: train on eligible rows from `start` up to the end of year - 1, validate on year."""
    index = pd.DatetimeIndex(index)
    folds = []
    for year in years:
        if pd.Timestamp(f"{year + 1}-01-01") > TEST_START:
            raise ValueError(f"validation year {year} overlaps the test period")
        folds.append({"year": year,
                      "train": np.asarray(eligible & (index >= start) & (index.year < year)),
                      "validation": np.asarray(eligible & (index.year == year))})
    return folds


def most_recent(mask, n):
    """The last n True positions of a date-sorted mask (all of them when n is None)."""
    positions = np.flatnonzero(mask)
    out = np.zeros(len(mask), dtype=bool)
    out[positions if n is None else positions[-n:]] = True
    return out


# Models: each returns stabilized band probabilities for every row ------------------------------

def prepare(samples, columns, train):
    """Design matrix and scoring inputs; standardization and the month prior use training rows only."""
    return {"x": design(samples, columns, train), "train": train,
            "z": np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos]),
            "rise": (samples.target_max - samples.LGA_max_so_far).to_numpy(),
            "edges": band_cdf_edges(samples), "floor": samples.LGA_max_so_far.to_numpy(),
            "prior": month_prior(samples.loc[train], samples),
            "target": samples.target_category.to_numpy()}


def censored(d, l2, skewed=False):
    t = d["train"]
    theta = shash.fit(d["x"][t], d["z"][t], d["rise"][t], l2, shape=False)
    out = {"Censored Gaussian": stabilize(to_bands(shash.cdf(theta, d["x"], d["z"], d["edges"])), d["prior"], d["floor"])}
    if skewed:
        theta = shash.fit(d["x"][t], d["z"][t], d["rise"][t], l2, shape=True, start=theta)
        out["Censored SHASH"] = stabilize(to_bands(shash.cdf(theta, d["x"], d["z"], d["edges"])), d["prior"], d["floor"])
    return out


def quantiles(d, depth):
    model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=TAUS, n_estimators=300, learning_rate=0.05,
                         max_depth=depth, min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
                         random_state=4260, n_jobs=2)
    model.fit(d["x"][d["train"]], d["rise"][d["train"]])
    return model.predict(d["x"])


def quantile_bands(d, q, scale):
    return stabilize(to_bands(quantile_cdf(scaled(q, scale), d["edges"])), d["prior"], d["floor"])


def classifiers(d):
    """The paper's 15 classifier settings (experiment.candidates) on the same design matrix."""
    t, target = d["train"], d["target"]
    if set(target[t]) != set(range(5)):
        raise ValueError("a training fold is missing a band")
    for family, config, model in candidates():
        model.fit(d["x"][t], target[t])
        yield CLASSIFIERS[family], config, stabilize(model.predict_proba(d["x"]), d["prior"], d["floor"])


def log_loss(p, target, mask):
    return float(daily_scores(p[mask], target[mask])["log_loss"].mean())


# Experiments ----------------------------------------------------------------------------------

def setup(samples, complete):
    splits = split_dates(samples.index)
    return {"validation": splits["validation"] & complete, "test": splits["test"] & complete,
            "train": splits["train"] & complete & (samples.index >= FIRST),
            "long_train": splits["train"] & (samples.index >= LONG),
            "guidance_columns": relative_columns(samples, SCOPE) + list(GROUPS),
            "observation_columns": relative_columns(samples, OBSERVATIONS)}


def fit_current(samples, complete):
    """CURRENT exactly as history_study.py: L2 from (1, 10, 30) on 2024, then one test score."""
    s = setup(samples, complete)
    d = prepare(samples, s["guidance_columns"], s["train"])
    fits = {l2: censored(d, l2)["Censored Gaussian"] for l2 in CURRENT_L2}
    l2 = min(fits, key=lambda value: log_loss(fits[value], d["target"], s["validation"]))
    p = fits[l2]
    return {"l2": l2, "n_train": int(s["train"].sum()), "train_log_loss": log_loss(p, d["target"], s["train"]),
            "validation_log_loss": log_loss(p, d["target"], s["validation"]),
            "test_log_loss": log_loss(p, d["target"], s["test"]), "probabilities": p}


def cross_validate(hour, samples, complete, grid):
    """Score every candidate on every rolling fold; return the 2021-2023 fold's probabilities by candidate."""
    s = setup(samples, complete)
    folds = rolling_folds(samples.index, complete & (samples.index < "2025-01-01"))
    assert np.array_equal(folds[-1]["train"], s["train"]) and np.array_equal(folds[-1]["validation"], s["validation"])
    final = {}
    for fold in folds:
        d = prepare(samples, s["guidance_columns"], fold["train"])
        scored = []
        for l2 in PATH_L2:
            for family, p in censored(d, l2, skewed=l2 in CURRENT_L2).items():
                scored.append((family, f"l2={l2:g}", p))
        for depth in DEPTHS:
            q = quantiles(d, depth)
            scored += [("Quantile boosting", f"depth={depth}, scale={scale:.1f}", quantile_bands(d, q, scale))
                       for scale in SCALES]
        scored += list(classifiers(d))
        for family, setting, p in scored:
            grid.append({"hour": hour, "fold": fold["year"], "family": family, "setting": setting,
                         "n_train": int(fold["train"].sum()), "n_validation": int(fold["validation"].sum()),
                         "validation_log_loss": log_loss(p, d["target"], fold["validation"])})
            if fold["year"] == CV_YEARS[-1]:
                final[(family, setting)] = p
    return final, s


def select(hour, grid, final, target, s, dates):
    """Pick the L2 and the model by the single 2024 split and by the rolling mean; score test afterwards.

    The rolling choice is compared with the single-split choice on test by a paired 7-day block bootstrap.
    """
    table = pd.DataFrame([row for row in grid if row["hour"] == hour])
    scores = table.pivot_table(index=["family", "setting"], columns="fold", values="validation_log_loss")
    scores["single_2024"] = scores[CV_YEARS[-1]]
    scores["rolling_mean"] = scores[list(CV_YEARS)].mean(axis=1)
    rows = []
    for task, pool in (("L2 of the censored Gaussian", scores.loc[["Censored Gaussian"]]), ("Model and setting", scores)):
        daily = {}
        for protocol in ("single_2024", "rolling_mean"):
            family, setting = pool[protocol].idxmin()
            # Test is read only here, after both choices are fixed.
            daily[protocol] = daily_scores(final[(family, setting)][s["test"]], target[s["test"]])["log_loss"].to_numpy()
            rows.append({"hour": hour, "task": task, "protocol": protocol, "family": family, "setting": setting,
                         "single_2024": pool.loc[(family, setting), "single_2024"],
                         "rolling_mean": pool.loc[(family, setting), "rolling_mean"],
                         "test_log_loss": float(daily[protocol].mean())})
        delta, low, high = block_bootstrap(dates, daily["rolling_mean"] - daily["single_2024"])
        rows[-1].update(test_delta_vs_single=delta, ci95_low=low, ci95_high=high)
    return rows, scores


def learning_curve(hour, samples, s, scope, base, columns, ns, out):
    """Fix hyperparameters on all of `base` by 2024 validation, then refit on the most recent N days."""
    d = prepare(samples, columns, base)
    target, validation = d["target"], s["validation"]
    fits = {l2: censored(d, l2)["Censored Gaussian"] for l2 in CURRENT_L2}
    l2 = min(fits, key=lambda value: log_loss(fits[value], target, validation))
    options = {}
    for depth in DEPTHS:
        q = quantiles(d, depth)
        for scale in SCALES:
            options[(depth, scale)] = (log_loss(quantile_bands(d, q, scale), target, validation), q)
    depth, scale = min(options, key=lambda key: options[key][0])
    for n in ns:
        train = most_recent(base, n)
        part = d if n is None else prepare(samples, columns, train)
        models = {"Censored Gaussian": (fits[l2] if n is None else censored(part, l2)["Censored Gaussian"], f"l2={l2:g}"),
                  "Quantile boosting": (quantile_bands(part, options[(depth, scale)][1] if n is None
                                                       else quantiles(part, depth), scale),
                                        f"depth={depth}, scale={scale:.1f}")}
        for model, (p, setting) in models.items():
            out.append({"hour": hour, "scope": scope, "model": model, "setting": setting, "n_train": int(train.sum()),
                        "first_train_date": samples.index[train].min().date().isoformat(),
                        "train_log_loss": log_loss(p, target, train),
                        "validation_log_loss": log_loss(p, target, validation),
                        "test_log_loss": log_loss(p, target, s["test"])})


def paper_families(hour, data, hrrr, s, samples, out, reproduction, reference):
    """Refit the paper's classifier families on its own cohort and inputs exactly as experiment.py does."""
    raw = season_features(pd.read_csv(data / f"samples_{hour:02}.csv.gz", parse_dates=["date"],
                                      index_col="date").sort_index())
    matched = raw.join(hrrr.loc[hrrr.hour == hour].set_index("date")[FORECAST_FEATURES], how="inner", validate="one_to_one")
    splits = split_dates(matched.index)
    prior = month_prior(matched.loc[splits["train"]], matched)
    predictions, selected = fit_scope(matched, splits, PAPER, prior, [])
    target = matched.target_category.to_numpy()
    masks = {"train": splits["train"], "validation": matched.index.isin(samples.index[s["validation"]]),
             "test": matched.index.isin(samples.index[s["test"]])}
    for family, p in predictions.items():
        out.append({"hour": hour, "inputs": "Paper: stations + trends + HRRR (matched-HRRR cohort)",
                    "system": CLASSIFIERS.get(family, family), "selected_on_2024": family == selected, "n_train": int(masks["train"].sum()),
                    **{f"{k}_log_loss": log_loss(p, target, m) for k, m in masks.items()}})
    ref = reference.loc[reference.hour.eq(hour)].set_index("date").loc[matched.index[masks["test"]]]
    p = predictions[selected][masks["test"]]
    reproduction.append({"hour": hour, "system": f"Paper selected ({selected})", "n_test": int(masks["test"].sum()),
                         "test_log_loss": log_loss(p, target[masks["test"]], np.ones(len(p), bool)),
                         "reference_test_log_loss": float(daily_scores(ref[PROBS].to_numpy(),
                                                                       ref.target_category)["log_loss"].mean()),
                         "max_abs_probability_difference": float(np.abs(p - ref[PROBS].to_numpy()).max())})


# Figures --------------------------------------------------------------------------------------

def style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def plot_learning_curves(curves, path):
    scopes = (("guidance", "guidance inputs, 2021-2023"), ("observations", "observations only, 2010-2023"))
    colors = {"Censored Gaussian": BLUE, "Quantile boosting": ORANGE}
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), sharex="row")
    for row, (scope, title) in enumerate(scopes):
        for col, hour in enumerate(HOURS):
            ax = axes[row, col]
            part = curves.loc[curves.scope.eq(scope) & curves.hour.eq(hour)]
            # Scale the axis to the curves once every season is in the training set (N >= 400).
            top = 1.25 * part.loc[part.n_train >= 400, ["train_log_loss", "validation_log_loss"]].to_numpy().max()
            notes = []
            for model, color in colors.items():
                m = part.loc[part.model.eq(model)].sort_values("n_train")
                ax.plot(m.n_train, m.validation_log_loss.clip(upper=top), "-", color=color, lw=2)
                ax.plot(m.n_train, m.train_log_loss.clip(upper=top), "--", color=color, lw=1.6)
                inside = m.validation_log_loss <= top
                ax.plot(m.n_train[inside], m.validation_log_loss[inside], "o", color=color, ms=6)
                ax.plot(m.n_train[~inside], np.full((~inside).sum(), top), "^", color=color, ms=7, clip_on=False)
                ax.plot(m.n_train, m.train_log_loss.clip(upper=top), "o", color=color, mfc="white", ms=6)
                short = "Gaussian" if model == "Censored Gaussian" else "Boosting"
                notes += [f"{short} validation at N={n:,}: {v:.2f}"
                          for n, v in zip(m.n_train[~inside], m.validation_log_loss[~inside])]
            if notes:
                ax.text(0.98, 0.03, "Off scale (triangle):\n" + "\n".join(notes), transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=7.5, color=MUTED)
            ax.set_xscale("log")
            ax.set_ylim(0, top)
            ax.set_title(f"{hour:02}:00, {title}", fontsize=9.5, color=INK, loc="left")
            style(ax)
            if col == 0:
                ax.set_ylabel("Log loss (lower is better)", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("Most recent N training days (log scale)", fontsize=9, color=INK)
            ticks = sorted(part.n_train.unique())
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:,}" for t in ticks], rotation=45 if row == 1 else 0, fontsize=8)
            ax.minorticks_off()
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", label="Censored Gaussian, 2024 validation"),
               Line2D([], [], color=BLUE, lw=1.6, ls="--", marker="o", mfc="white", label="Censored Gaussian, training"),
               Line2D([], [], color=ORANGE, lw=2, marker="o", label="Quantile boosting, 2024 validation"),
               Line2D([], [], color=ORANGE, lw=1.6, ls="--", marker="o", mfc="white", label="Quantile boosting, training")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.11, hspace=0.32, wspace=0.17)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_path(path_table, chosen, file):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    lines = (("train_log_loss", "Training 2021-2023", GRAY, "--"),
             ("validation_log_loss", "2024 validation (single split)", BLUE, "-"),
             ("rolling_mean", "Rolling-origin CV mean (2022-2024)", ORANGE, "-"),
             ("test_log_loss", "Test (not used to choose)", AQUA, ":"))
    for ax, hour in zip(axes, HOURS):
        part = path_table.loc[path_table.hour.eq(hour)].sort_values("l2")
        for column, label, color, line in lines:
            ax.plot(part.l2, part[column], line, color=color, lw=2, marker="o", ms=4.5, label=label)
        ax.axvline(chosen[(hour, "single_2024")], color=BLUE, lw=1.2, alpha=0.7, label="L2 chosen by 2024 split")
        ax.axvline(chosen[(hour, "rolling_mean")], color=ORANGE, lw=1.2, ls=(0, (4, 3)), label="L2 chosen by rolling CV")
        ax.set_xscale("log")
        ax.set_ylim(0, part[[c for c, *_ in lines]].to_numpy().max() * 1.05)
        ax.set_title(f"{hour:02}:00", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("L2 penalty (log scale)", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("Log loss (lower is better)", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(file, dpi=170)
    plt.close(fig)


# Main -----------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "class_eval")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    daily = pd.read_csv(ROOT / "results" / "daily_predictions.csv", parse_dates=["date"])
    reference = daily.loc[daily.cohort.eq("Matched HRRR") & daily.scope.eq(PAPER)]
    grid, selections, path_rows, curves, overfit, reproduction = [], [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples, complete = load(args.data, hour, hrrr, guidance)
            assert samples.index.is_monotonic_increasing
            current = fit_current(samples, complete)
            reproduction.append({"hour": hour, "system": f"CURRENT (censored Gaussian, l2={current['l2']:g})",
                                 "n_test": int((split_dates(samples.index)["test"] & complete).sum()),
                                 "test_log_loss": current["test_log_loss"], "reference_test_log_loss": REFERENCE[hour],
                                 "max_abs_probability_difference": np.nan})
            final, s = cross_validate(hour, samples, complete, grid)
            target = samples.target_category.to_numpy()
            rows, scores = select(hour, grid, final, target, s, samples.index[s["test"]])
            selections += rows
            for l2 in PATH_L2:
                p = final[("Censored Gaussian", f"l2={l2:g}")]
                path_rows.append({"hour": hour, "l2": l2, "train_log_loss": log_loss(p, target, s["train"]),
                                  "validation_log_loss": log_loss(p, target, s["validation"]),
                                  "rolling_mean": scores.loc[("Censored Gaussian", f"l2={l2:g}"), "rolling_mean"],
                                  "test_log_loss": log_loss(p, target, s["test"]),
                                  "current_grid": l2 in CURRENT_L2})
            # Overfitting: each family at its single-split choice on CURRENT's inputs and rows.
            single = scores["single_2024"]
            for family in single.index.get_level_values(0).unique():
                setting = single.loc[family].idxmin()
                if family == "Censored Gaussian":
                    setting = f"l2={current['l2']:g}"  # CURRENT's own grid (1, 10, 30)
                p = final[(family, setting)]
                overfit.append({"hour": hour, "inputs": "CURRENT: stations + trends + HRRR + LAMP/NBM (complete rows)",
                                "system": f"{family} ({setting})" + (" = CURRENT" if family == "Censored Gaussian" else ""),
                                "selected_on_2024": family == "Censored Gaussian", "n_train": int(s["train"].sum()),
                                **{f"{k}_log_loss": log_loss(p, target, s[k]) for k in ("train", "validation", "test")}})
            paper_families(hour, args.data, hrrr, s, samples, overfit, reproduction, reference)
            learning_curve(hour, samples, s, "guidance", s["train"], s["guidance_columns"], GUIDANCE_N, curves)
            learning_curve(hour, samples, s, "observations", s["long_train"], s["observation_columns"], LONG_N, curves)
            print(f"{hour:02}:00 complete", flush=True)
    grid = pd.DataFrame(grid)
    selections = pd.DataFrame(selections)
    path_table = pd.DataFrame(path_rows)
    curves = pd.DataFrame(curves)
    overfit = pd.DataFrame(overfit).sort_values(["hour", "inputs", "validation_log_loss"], kind="stable")
    overfit["gap_validation_minus_train"] = overfit.validation_log_loss - overfit.train_log_loss
    overfit["gap_test_minus_train"] = overfit.test_log_loss - overfit.train_log_loss
    reproduction = pd.DataFrame(reproduction)
    reproduction["difference"] = reproduction.test_log_loss - reproduction.reference_test_log_loss
    for name, table in (("cv_grid", grid), ("cv_selection", selections), ("regularization_path", path_table),
                        ("learning_curves", curves), ("overfitting", overfit), ("reproduction", reproduction)):
        table.to_csv(args.out / f"{name}.csv", index=False)
    chosen = {(r.hour, r.protocol): float(r.setting.split("=")[1])
              for r in selections.loc[selections.task.str.startswith("L2")].itertuples()}
    plot_learning_curves(curves, args.out / "learning_curves.png")
    plot_path(path_table, chosen, args.out / "regularization_path.png")
    with pd.option_context("display.width", 230, "display.float_format", "{:.4f}".format, "display.max_rows", 200,
                           "display.max_colwidth", 70):
        print(reproduction.to_string(index=False))
        print(selections.to_string(index=False))
        print(path_table.to_string(index=False))
        print(curves.to_string(index=False))
        print(overfit.to_string(index=False))
    bad = reproduction.loc[reproduction.difference.abs() > TOLERANCE]
    if len(bad):
        raise SystemExit(f"reproduction outside {TOLERANCE}:\n{bad.to_string(index=False)}")


if __name__ == "__main__":
    main()
