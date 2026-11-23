"""Reproduce the tables and figures in the CS4260 final paper."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
BOUNDS = np.array([40, 55, 70, 85])
CLASSES = np.arange(5)
HOURS = (9, 12, 15)
PROBS = [f"p{i}" for i in CLASSES]
FORECAST_FEATURES = ["hrrr_remaining_max", "hrrr_warming"]


def split_dates(index):
    return {
        "train": index < "2024-01-01",
        "validation": (index >= "2024-01-01") & (index < "2025-01-01"),
        "progress_overlap": (index >= "2025-01-01") & (index < "2025-09-01"),
        "test": (index >= "2025-09-01") & (index < "2026-09-01"),
    }


def season_features(samples):
    date_angle = 2 * np.pi * (samples.index.dayofyear - 1) / 365.25
    return samples.assign(season_sin=np.sin(date_angle), season_cos=np.cos(date_angle))


def feature_columns(samples, scope):
    station_features = [column for column in samples if column.split("_")[0] in ("LGA", "JFK", "EWR", "TEB")]
    if scope == "LGA":
        station_features = [column for column in station_features if column.startswith("LGA_")]
    if scope in ("LGA", "Four stations"):
        station_features = [column for column in station_features if "_trend_" not in column]
    if scope == "Four stations + trends + HRRR":
        station_features.extend(FORECAST_FEATURES)
    return ["season_sin", "season_cos", *station_features]


def month_prior(training, samples):
    counts = pd.crosstab(training.index.month, training.target_category)
    counts = counts.reindex(index=range(1, 13), columns=CLASSES, fill_value=0).to_numpy() + 1
    return (counts / counts.sum(axis=1, keepdims=True))[samples.index.month - 1]


def remaining_warming(training, samples):
    historical = training.dropna(subset=["LGA_max_so_far"])
    historical_high = historical["LGA_max_so_far"].to_numpy()
    rise = (historical["target_max"] - historical["LGA_max_so_far"]).clip(lower=0).to_numpy()
    months = historical.index.month.to_numpy()
    probabilities = month_prior(training, samples)
    for position, (day, high) in enumerate(samples["LGA_max_so_far"].items()):
        if np.isfinite(high):
            travel = np.abs(months - day.month)
            distance = np.minimum(travel, 12 - travel)
            weights = np.exp(-0.5 * ((distance / 1.5)**2 + ((historical_high - high) / 5)**2))
            categories = np.searchsorted(BOUNDS, high + rise, side="right")
            counts = np.bincount(categories, weights=weights, minlength=5)
            probabilities[position] = counts / counts.sum()
    return probabilities


def stabilize(probabilities, prior, high):
    """Mix 1% monthly prior and remove bands already surpassed by a closed hour."""
    mixed = 0.99 * probabilities + 0.01 * prior
    floor = np.searchsorted(BOUNDS, np.nan_to_num(high, nan=-np.inf), side="right")
    mixed = np.where(CLASSES >= floor[:, None], mixed, 0.0)
    return mixed / mixed.sum(axis=1, keepdims=True)


def candidates():
    for c in (0.1, 1, 10):
        yield "Logistic regression", f"C={c}", make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(C=c, max_iter=1000))
    for k in (15, 40, 80):
        yield "KNN", f"k={k}", make_pipeline(
            SimpleImputer(), StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    for smoothing in (1e-9, 0.01, 0.1):
        yield "Gaussian naive Bayes", f"var_smoothing={smoothing}", make_pipeline(
            SimpleImputer(), StandardScaler(), GaussianNB(var_smoothing=smoothing))
    for depth in (5, 10, None):
        yield "Random forest", f"depth={depth}", make_pipeline(
            SimpleImputer(), RandomForestClassifier(n_estimators=200, max_depth=depth,
                                                    min_samples_leaf=4, random_state=4260, n_jobs=2))
    for depth in (2, 3, 5):
        yield "XGBoost", f"depth={depth}", XGBClassifier(
            n_estimators=200, max_depth=depth, learning_rate=0.04, subsample=0.9,
            colsample_bytree=0.9, objective="multi:softprob", random_state=4260, n_jobs=2)


def metrics(target, probabilities):
    observed = np.eye(5)[target]
    predicted = probabilities.argmax(axis=1)
    return {
        "n": len(target),
        "log_loss": log_loss(target, probabilities, labels=CLASSES),
        "brier": np.mean(np.sum((observed - probabilities)**2, axis=1)),
        "accuracy": np.mean(target == predicted),
        "macro_f1": f1_score(target, predicted, labels=CLASSES, average="macro", zero_division=0),
    }


def fit_scope(samples, splits, scope, priors, tuning):
    """Use 2024 alone to choose one setting per family and the ensemble members."""
    train, validation = splits["train"], splits["validation"]
    columns = feature_columns(samples, scope)
    target = samples["target_category"].to_numpy()
    high = samples["LGA_max_so_far"].to_numpy()
    best = {}
    for family, config, model in candidates():
        model.fit(samples.loc[train, columns], target[train])
        val_probs = stabilize(model.predict_proba(samples.loc[validation, columns]),
                              priors[validation], high[validation])
        loss = log_loss(target[validation], val_probs, labels=CLASSES)
        tuning.append({"scope": scope, "model": family, "config": config, "validation_log_loss": loss})
        if family not in best or loss < best[family][0]:
            best[family] = (loss, config, model)
    families = sorted(best, key=lambda family: best[family][0])
    minimum = families[0]
    predictions = {family: stabilize(best[family][2].predict_proba(samples[columns]), priors, high)
                   for family in families}
    ensemble_name = "Two-model probability average"
    ensemble = np.mean([predictions[name] for name in families[:2]], axis=0)
    ensemble_loss = log_loss(target[validation], ensemble[validation], labels=CLASSES)
    tuning.append({"scope": scope, "model": ensemble_name, "config": " + ".join(families[:2]),
                   "validation_log_loss": ensemble_loss})
    predictions[ensemble_name] = ensemble
    selected = ensemble_name if ensemble_loss < best[minimum][0] else minimum
    return predictions, selected


def compare(hour, samples, cohort, tuning):
    splits = split_dates(samples.index)
    train = samples.loc[splits["train"]]
    prior = month_prior(train, samples)
    target = samples.target_category.to_numpy()
    groups = {
        "Monthly climatology": ({"Monthly climatology": prior}, "Monthly climatology"),
        "Remaining warming": ({"Remaining warming": stabilize(
            remaining_warming(train, samples), prior, samples.LGA_max_so_far.to_numpy())},
            "Remaining warming"),
    }
    scopes = ["LGA", "Four stations", "Four stations + trends"] if cohort == "Observations" else [
        "Four stations + trends", "Four stations + trends + HRRR"]
    for scope in scopes:
        groups[scope] = fit_scope(samples, splits, scope, prior, tuning)
    records, daily = [], []
    for scope, (models, selected) in groups.items():
        for model, probabilities in models.items():
            for split in ("validation", "progress_overlap", "test"):
                mask = splits[split]
                if mask.any():
                    records.append({"hour": hour, "cohort": cohort, "scope": scope, "model": model,
                                    "selected": model == selected, "split": split,
                                    **metrics(target[mask], probabilities[mask])})
            if model == selected:
                mask = splits["test"]
                result = samples.loc[mask, ["target_max", "target_category", "LGA_max_so_far"]].copy()
                result[PROBS] = probabilities[mask]
                result = result.assign(hour=hour, cohort=cohort, scope=scope, model=model)
                daily.append(result.reset_index())
    return records, daily


def analyze(daily, out):
    daily["predicted"] = daily[PROBS].to_numpy().argmax(axis=1)
    daily["confidence"] = daily[PROBS].max(axis=1)
    daily["correct"] = daily["predicted"] == daily.target_category
    daily["distance_to_boundary"] = np.abs(daily.target_max.to_numpy()[:, None] - BOUNDS).min(axis=1)
    daily["near_boundary"] = daily.distance_to_boundary <= 2
    daily["month"] = daily.date.dt.month
    daily["daily_loss"] = -np.log(daily[PROBS].to_numpy()[np.arange(len(daily)), daily.target_category].clip(1e-15))
    daily.to_csv(out / "daily_predictions.csv", index=False)
    keys = ["cohort", "scope", "hour"]
    comparisons = (
        ("Observations", "LGA", "Four stations"),
        ("Observations", "Four stations", "Four stations + trends"),
        ("Observations", "Remaining warming", "Four stations + trends"),
        ("Matched HRRR", "Four stations + trends", "Four stations + trends + HRRR"),
    )
    paired = []
    for cohort, baseline, candidate in comparisons:
        for hour in HOURS:
            match = daily.loc[daily.cohort.eq(cohort) & daily.hour.eq(hour)]
            losses = match.pivot(index="date", columns="scope", values="daily_loss")
            if candidate in losses:
                difference = losses[candidate] - losses[baseline]
                paired.append({"cohort": cohort, "hour": hour, "baseline": baseline,
                               "candidate": candidate, "n": difference.count(),
                               "delta_log_loss": difference.mean()})
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
    for column in ["near_boundary", "month"]:
        daily.groupby([*keys, column]).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                           log_loss=("daily_loss", "mean")).to_csv(out / f"by_{column}.csv")
    hrrr = daily.loc[daily.scope == "Four stations + trends + HRRR"].copy()
    main = hrrr if len(hrrr) else daily.loc[daily.scope == "Four stations + trends"].copy()
    main.loc[~main.correct].sort_values("confidence", ascending=False).to_csv(out / "errors.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), sharex=True, sharey=True)
    for hour, ax in zip(HOURS, axes):
        now = main.loc[main.hour == hour].copy()
        now["bin"] = pd.cut(now.confidence, np.linspace(0, 1, 11), include_lowest=True)
        bins = now.groupby("bin", observed=True).agg(confidence=("confidence", "mean"),
                                                     accuracy=("correct", "mean"), n=("correct", "size"))
        ax.plot([0.2, 1], [0.2, 1], "--", color="#7d8994", linewidth=1)
        ax.scatter(bins.confidence, bins.accuracy, s=10 + bins.n * 0.8, color="#146e87")
        ax.set(title=f"{hour:02}:00", xlabel="Mean top-class probability", xlim=(0.2, 1.02), ylim=(0.18, 1.02))
    axes[0].set_ylabel("Observed accuracy")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    forecast_file = args.data / "hrrr_features.csv.gz"
    forecast = pd.read_csv(forecast_file, parse_dates=["date"]) if forecast_file.exists() else None
    records, daily, tuning = [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples = pd.read_csv(args.data / f"samples_{hour:02}.csv.gz", parse_dates=["date"], index_col="date")
            samples = season_features(samples.sort_index())
            cohorts = {"Observations": samples}
            if forecast is not None:
                predictors = forecast.loc[forecast.hour == hour].set_index("date")[FORECAST_FEATURES]
                cohorts["Matched HRRR"] = samples.join(predictors, how="inner", validate="one_to_one")
            for name, cohort in cohorts.items():
                settings = []
                rows, predictions = compare(hour, cohort, name, settings)
                tuning.extend({"hour": hour, "cohort": name, **setting} for setting in settings)
                records.extend(rows)
                daily.extend(predictions)
                print(f"{hour:02}:00 {name} complete", flush=True)
    all_metrics = pd.DataFrame(records)
    all_metrics.to_csv(args.out / "all_metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.out / "validation_grid.csv", index=False)
    main_table = all_metrics.loc[all_metrics.selected & all_metrics.split.eq("test")].drop(columns=["selected", "split"])
    main_table.to_csv(args.out / "main_table.csv", index=False)
    analyze(pd.concat(daily, ignore_index=True), args.out)


if __name__ == "__main__":
    main()
