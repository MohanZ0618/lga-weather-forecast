"""Download NOAA observations and run the same-day LGA experiments."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TZ = "America/New_York"
YEARS = (2023, 2024, 2025)
HOURS = (9, 12, 15)
STATIONS = {
    "LGA": "72503014732",
    "JFK": "74486094789",
    "EWR": "72502014734",
    "TEB": "72502594741",
}
SCOPES = {"LGA only": ("LGA",), "Four stations": tuple(STATIONS)}
NOAA_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station_id}.csv"
NOAA_COLUMNS = ["DATE", "TMP", "DEW", "WND", "REPORT_TYPE"]
QUALITY_CODES = {"0", "1", "4", "5"}
OBS_COLUMNS = ["temp", "rh", "wind"]
STATION_FEATURES = [*OBS_COLUMNS, "max_so_far", "change_3h", "previous_max"]
CLASSES = np.arange(5)
CATEGORY_LABELS = ["Below 40°F", "40 to <55°F", "55 to <70°F", "70 to <85°F", "85°F or above"]
PROB_COLUMNS = [f"prob_{category}" for category in CLASSES]
ONE_HOUR = pd.Timedelta(hours=1)


def validate_noaa_csv(content, source):
    first_line = content.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace")
    header = next(csv.reader([first_line]), [])
    if not set(NOAA_COLUMNS).issubset(header) or b"\n" not in content.rstrip():
        raise ValueError(f"Expected a NOAA CSV with observations from {source}")


def download_noaa(refresh=False):
    manifest = []
    for station, station_id in STATIONS.items():
        for year in YEARS:
            url = NOAA_URL.format(year=year, station_id=station_id)
            path = RAW / str(year) / f"{station_id}.csv"
            if refresh or not path.is_file():
                print(f"Downloading {station} {year}", flush=True)
                response = requests.get(url, timeout=(15, 60))
                response.raise_for_status()
                content = response.content
                validate_noaa_csv(content, url)
                path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write(content)
                try:
                    temporary_path.replace(path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            else:
                content = path.read_bytes()
                validate_noaa_csv(content, path)
            manifest.append({
                "station": station,
                "year": year,
                "url": url,
                "file": path.relative_to(RAW).as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def relative_humidity(temp_c, dew_c):
    humidity = 100 * np.exp(
        17.625 * dew_c / (243.04 + dew_c) - 17.625 * temp_c / (243.04 + temp_c)
    )
    return humidity.clip(0, 100)


def observation_table(timestamps, temp_c, dew_c, wind_ms):
    """Use a common schema and preserve the original timestamp for both providers."""
    table = pd.DataFrame({
        "timestamp": timestamps,
        "temp": temp_c * 1.8 + 32,
        "rh": relative_humidity(temp_c, dew_c),
        "wind": wind_ms,
    }).dropna(subset=["timestamp"])
    table = table.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    return table.tz_convert(TZ)


def noaa_value(series, value_field, quality_field):
    fields = series.str.split(",")
    values = pd.to_numeric(fields.str[value_field], errors="coerce")
    quality = fields.str[quality_field].str.strip()
    return values.where(quality.isin(QUALITY_CODES) & values.abs().ne(9999)).astype(float) / 10


def read_noaa(path):
    frame = pd.read_csv(path, usecols=NOAA_COLUMNS, dtype="string")
    routine = frame[frame["REPORT_TYPE"].str.strip().isin(["FM-15", "FM-16"])]
    observations = observation_table(
        pd.to_datetime(routine["DATE"], format="ISO8601", errors="coerce", utc=True),
        noaa_value(routine["TMP"], 0, 1),
        noaa_value(routine["DEW"], 0, 1),
        noaa_value(routine["WND"], 3, 4),
    )
    return observations, len(routine)


def hourly_rows(observations):
    utc_hours = observations.index.tz_convert("UTC").floor("h")
    return observations.loc[~utc_hours.duplicated(keep="last")]


def process_noaa(manifest):
    by_station = {station: [] for station in STATIONS}
    quality = []
    for entry in manifest:
        station, year = entry["station"], entry["year"]
        observations, routine_count = read_noaa(RAW / entry["file"])
        hourly = hourly_rows(observations)
        hourly.to_csv(PROCESSED / f"{station}_{year}_hourly.csv")
        by_station[station].append(observations)
        quality.append({
            "station": station,
            "source_year": year,
            "routine_reports": routine_count,
            "hourly_rows": len(hourly),
            "missing_temp": hourly["temp"].isna().sum(),
            "missing_rh": hourly["rh"].isna().sum(),
            "missing_wind": hourly["wind"].isna().sum(),
            "first_local_observation": hourly.index.min(),
            "last_local_observation": hourly.index.max(),
        })
    pd.DataFrame(quality).to_csv(PROCESSED / "data_quality.csv", index=False)
    return {station: pd.concat(tables).sort_index() for station, tables in by_station.items()}


def daily_labels(hourly):
    valid = hourly.dropna(subset=["temp"])
    days = valid.assign(date=valid.index.date, period=valid.index.hour // 6)
    coverage = days.groupby("date").agg(
        valid_hours=("temp", "size"), covered_periods=("period", "nunique"),
        target_max=("temp", "max"),
    )
    eligible = coverage.loc[
        (coverage["valid_hours"] >= 20) & (coverage["covered_periods"] == 4)
    ].copy()
    eligible["target_category"] = np.searchsorted([40, 55, 70, 85], eligible["target_max"], side="right")
    eligible.index = pd.DatetimeIndex(eligible.index, name="date")
    return eligible


def local_cutoff(day, hour):
    return (pd.Timestamp(day) + pd.Timedelta(hours=hour)).tz_localize(TZ)


def previous_day_maxima(observations):
    hourly = hourly_rows(observations)
    return hourly["temp"].groupby(hourly.index.date).max()


def latest_index(observations, limit):
    position = observations.index.searchsorted(limit, side="right") - 1
    if position >= 0 and limit - observations.index[position] <= ONE_HOUR:
        return position
    return None


def make_features(observations_by_station, maxima_by_station, stations, day, hour):
    """Return one feature row and the timestamps used for its recent inputs."""
    day = pd.Timestamp(day)
    cutoff = local_cutoff(day, hour)
    midnight = local_cutoff(day, 0)
    yesterday = (day - pd.Timedelta(days=1)).date()
    features = {"month": day.month}
    timestamps = {}
    for station in stations:
        observations = observations_by_station[station]
        recent = latest_index(observations, cutoff)
        older = latest_index(observations, cutoff - 3 * ONE_HOUR)
        latest = observations.iloc[recent] if recent is not None else {}
        older_temp = observations.iloc[older]["temp"] if older is not None else np.nan
        timestamps[station] = observations.index[recent] if recent is not None else None

        # Trim before reducing to hourly rows so a later report in the cutoff hour
        # cannot replace an observation already available at prediction time.
        today = hourly_rows(observations.loc[midnight:cutoff])
        values = {
            **{column: latest.get(column, np.nan) for column in OBS_COLUMNS},
            "max_so_far": today["temp"].max(),
            "change_3h": latest.get("temp", np.nan) - older_temp,
            "previous_max": maxima_by_station[station].get(yesterday, np.nan),
        }
        features.update({f"{station}_{name}": value for name, value in values.items()})
    return features, timestamps


def feature_columns(stations):
    return ["month", *(f"{station}_{name}" for station in stations for name in STATION_FEATURES)]


def build_samples(observations, labels, hour):
    maxima = {station: previous_day_maxima(table) for station, table in observations.items()}
    rows = [make_features(observations, maxima, STATIONS, day, hour)[0] for day in labels.index]
    return pd.DataFrame(rows, index=labels.index).join(labels[["target_max", "target_category"]])


def new_models():
    return {
        "Climatology": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1, max_iter=2000),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=2),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            XGBClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                n_jobs=2, eval_metric="mlogloss",
            ),
        ),
    }


def five_probabilities(model, features):
    probabilities = np.zeros((len(features), len(CLASSES)))
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(features)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def scores(target, probabilities):
    predictions = CLASSES[probabilities.argmax(axis=1)]
    actual = np.eye(len(CLASSES))[np.asarray(target, dtype=int)]
    return {
        "n": len(target),
        "accuracy": accuracy_score(target, predictions),
        "macro_f1": f1_score(target, predictions, labels=CLASSES, average="macro", zero_division=0),
        "log_loss": log_loss(target, probabilities, labels=CLASSES),
        "brier": np.mean(np.sum((probabilities - actual) ** 2, axis=1)),
    }


def run_experiments(observations, labels):
    results, selected, all_predictions = [], [], []
    for hour in HOURS:
        samples = build_samples(observations, labels, hour)
        samples.to_csv(PROCESSED / f"samples_{hour:02d}.csv")
        train, validation, test = (samples.loc[samples.index.year == year] for year in YEARS)
        if any(split.empty for split in (train, validation, test)):
            raise ValueError("The experiments require eligible dates in all three years.")
        if set(train["target_category"]) != set(CLASSES):
            raise ValueError("This five-class XGBoost experiment requires all categories in 2023.")

        candidates = []
        for scope, stations in SCOPES.items():
            columns = feature_columns(stations)
            for name, model in new_models().items():
                model.fit(train[columns], train["target_category"])
                validation_probs = five_probabilities(model, validation[columns])
                validation_scores = scores(validation["target_category"], validation_probs)
                results.append({
                    "scope": scope, "hour": hour, "model": name, "split": "validation",
                    **validation_scores,
                })
                artifact = {"model": model, "feature_columns": columns, "stations": stations, "hour": hour}
                candidates.append((scope, name, validation_scores["log_loss"], artifact))

        best = min((item for item in candidates if item[1] != "Climatology"), key=lambda item: item[2])
        scope, name, validation_loss, artifact = best
        file = f"model_{hour:02d}.joblib"
        joblib.dump(artifact, OUTPUTS / file)
        selected.append({
            "hour": hour, "scope": scope, "model": name, "stations": list(artifact["stations"]),
            "validation_log_loss": validation_loss, "artifact": file,
        })
        print(f"{hour:02d}:00 selected {scope}, {name} (validation log loss {validation_loss:.4f})", flush=True)

        # The selected model is already fixed before scoring this cutoff on 2025.
        for candidate in candidates:
            scope, name, _, artifact = candidate
            probabilities = five_probabilities(artifact["model"], test[artifact["feature_columns"]])
            results.append({
                "scope": scope, "hour": hour, "model": name, "split": "test",
                **scores(test["target_category"], probabilities),
            })
            if candidate is best:
                predictions = test[["target_max", "target_category"]].copy()
                predictions["hour"] = hour
                predictions["scope"] = scope
                predictions["model"] = name
                predictions[PROB_COLUMNS] = probabilities
                predictions["predicted_category"] = CLASSES[probabilities.argmax(axis=1)]
                predictions["confidence"] = probabilities.max(axis=1)
                all_predictions.append(predictions)

    pd.DataFrame(results).to_csv(OUTPUTS / "metrics.csv", index=False)
    (OUTPUTS / "selected_models.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    predictions = pd.concat(all_predictions).sort_values(["date", "hour"])
    predictions.to_csv(OUTPUTS / "selected_test_predictions.csv")
    errors = predictions.loc[predictions["predicted_category"] != predictions["target_category"]]
    errors.sort_values("confidence", ascending=False, kind="stable").to_csv(OUTPUTS / "prediction_errors.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Replace cached NOAA CSVs with fresh downloads")
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    observations = process_noaa(download_noaa(args.refresh))
    labels = daily_labels(hourly_rows(observations["LGA"]))
    labels = labels.loc[labels.index.year.isin(YEARS)]
    labels.to_csv(PROCESSED / "lga_daily_labels.csv")
    print(f"Eligible local target dates by year: {labels.groupby(labels.index.year).size().to_dict()}", flush=True)
    run_experiments(observations, labels)


if __name__ == "__main__":
    main()
