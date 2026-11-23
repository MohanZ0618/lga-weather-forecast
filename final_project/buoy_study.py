"""Compare the existing HRRR model with and without archived NY Harbor buoy observations."""

import argparse
import calendar
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiment import (BOUNDS, FORECAST_FEATURES, HOURS, PROBS, ROOT, candidates, feature_columns,
                        metrics, month_prior, season_features, split_dates, stabilize)

DATA = ROOT / "data"
OUT = ROOT / "results" / "buoy"
SCOPE = "Four stations + trends + HRRR"
BASE = "https://www.ndbc.noaa.gov/data"
BUOY = ["sea_f", "marine_air_f", "east_wind_ms", "north_wind_ms"]
EXTRA = ["sea_f", "east_wind_ms", "north_wind_ms", "land_minus_sea_f"]


def archive_urls():
    for year in range(2021, 2026):
        yield f"{BASE}/historical/stdmet/44065h{year}.txt.gz"
    for month in range(1, 9):
        yield f"{BASE}/stdmet/{calendar.month_abbr[month]}/44065{month}2026.txt.gz"


def clean_archive(content):
    raw = pd.read_csv(io.BytesIO(content), compression="gzip", sep=r"\s+", skiprows=[1])
    clock = raw[["#YY", "MM", "DD", "hh", "mm"]].copy()
    clock.columns = ["year", "month", "day", "hour", "minute"]
    timestamps = pd.to_datetime(clock, utc=True, errors="raise")
    def numeric(name, low, high):
        values = pd.to_numeric(raw[name], errors="coerce")
        return values.where(values.between(low, high))

    # Exclude historical missing tokens (999 deg/degC, 99 m/s) and implausible values.
    speed = numeric("WSPD", 0, 75)
    direction = np.deg2rad(numeric("WDIR", 0, 360))
    table = pd.DataFrame({
        "sea_f": numeric("WTMP", -3, 45) * 1.8 + 32,
        "marine_air_f": numeric("ATMP", -60, 55) * 1.8 + 32,
        "east_wind_ms": (-speed * np.sin(direction)).mask(speed.eq(0), 0),
        "north_wind_ms": (-speed * np.cos(direction)).mask(speed.eq(0), 0),
    })
    table.index = pd.DatetimeIndex(timestamps, name="observed_utc")
    return table


def archives(fetch):
    cache = DATA / "raw" / "ndbc"
    cache.mkdir(parents=True, exist_ok=True)
    frames, manifest = [], []
    for url in archive_urls():
        path = cache / Path(url).name
        if path.exists():
            content = path.read_bytes()
        elif fetch:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            content = response.content
        else:
            raise FileNotFoundError(f"Missing {path.name}. Run with --fetch to populate the cache.")
        frame = clean_archive(content)
        if not path.exists():
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        frames.append(frame)
        manifest.append(dict(url=url, sha256=hashlib.sha256(content).hexdigest(), rows=len(frame),
                             first=frame.index.min(), last=frame.index.max(),
                             **{f"missing_{col}": int(frame[col].isna().sum()) for col in BUOY}))
    obs = pd.concat(frames).sort_index(kind="stable")
    return obs.loc[~obs.index.duplicated(keep="last")], pd.DataFrame(manifest)


def asof_features(days, hour, observations):
    # Archive timestamps are observation times. Allow 30 minutes for receipt and
    # reject reports older than two hours at local issue time, including across DST.
    cutoff = (days + pd.Timedelta(hours=hour)).tz_localize("America/New_York").tz_convert("UTC")
    issues = pd.DataFrame({"cutoff_utc": cutoff, "ready_by": cutoff - pd.Timedelta(minutes=30)})
    result = pd.merge_asof(issues, observations.reset_index(), left_on="ready_by",
                           right_on="observed_utc", tolerance=pd.Timedelta(minutes=90))
    result.index = days
    return result.drop(columns="ready_by")


def selected_models(grid, hour):
    settings = grid.loc[grid.hour.eq(hour) & grid.cohort.eq("Matched HRRR") & grid.scope.eq(SCOPE)]
    best = settings.sort_values("validation_log_loss").drop_duplicates("model").set_index("model")
    winner = best.index[0]
    families = best.loc[winner, "config"].split(" + ") if winner == "Two-model probability average" else [winner]
    pool = {(family, config): model for family, config, model in candidates()}
    return [pool[(family, best.loc[family, "config"])] for family in families]


def threshold_probabilities(data, train_dates, prior, extra):
    # Share the limited training days across the four ordered boundaries.
    # Positive margins mean the forecast/observation already exceeds a boundary.
    base = data[["season_sin", "season_cos", "LGA_trend_1h", "LGA_rh", "LGA_wind",
                 "hrrr_warming", *extra]].copy()
    base["inland_gradient"] = data.EWR_temp - data.LGA_temp
    base["forecast_bias"] = data.LGA_temp - (data.hrrr_remaining_max - data.hrrr_warming)
    blocks = [base.assign(forecast_margin=data.hrrr_remaining_max - limit,
                          high_margin=data.LGA_max_so_far - limit,
                          current_margin=data.LGA_temp - limit) for limit in BOUNDS]
    inputs = pd.concat(blocks, ignore_index=True)
    historical_high = data.loc[train_dates, "target_max"]
    training = pd.concat([block.loc[train_dates].assign(crossed=historical_high.ge(limit))
                          for limit, block in zip(BOUNDS, blocks)])
    training = training.loc[~training.high_margin.ge(0)]
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1, max_iter=1000))
    model.fit(training[inputs.columns], training.crossed)
    crossing = model.predict_proba(inputs)[:, 1].reshape(len(BOUNDS), len(data)).T
    high = data.LGA_max_so_far.to_numpy()
    return crossing_to_bands(crossing, high, prior)


def crossing_to_bands(crossing, high, prior):
    crossing = np.where(high[:, None] >= BOUNDS, 1.0, crossing)
    cdf = 1 - np.minimum.accumulate(crossing, axis=1)
    return stabilize(np.diff(np.column_stack([np.zeros(len(high)), cdf, np.ones(len(high))])), prior, high)


def run(fetch):
    obs, manifest = archives(fetch)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "sources.csv", index=False)
    hrrr = pd.read_csv(DATA / "hrrr_features.csv.gz", parse_dates=["date"])
    grid = pd.read_csv(ROOT / "results" / "validation_grid.csv")
    scores, predictions, features = [], [], []
    for hour in HOURS:
        station = pd.read_csv(DATA / f"samples_{hour:02}.csv.gz", index_col="date", parse_dates=["date"])
        numerical = hrrr.loc[hrrr.hour.eq(hour)].set_index("date")[FORECAST_FEATURES]
        data = season_features(station.join(numerical, how="inner", validate="one_to_one").sort_index())
        marine = asof_features(data.index, hour, obs)
        marine["land_minus_sea_f"] = data.LGA_temp - marine.sea_f
        features.append(marine.assign(hour=hour))
        data = data.join(marine[EXTRA])
        columns = feature_columns(data, SCOPE)
        splits = split_dates(data.index)
        train = data.loc[splits["train"]]
        prior = month_prior(train, data)
        high = data.LGA_max_so_far.to_numpy()
        comparisons = {}
        for name, inputs in [("original", columns), ("with_buoy", columns + EXTRA)]:
            members = []
            for model in selected_models(grid, hour):
                model.fit(train[inputs], train.target_category)
                members.append(stabilize(model.predict_proba(data[inputs]), prior, high))
            comparisons[name] = np.mean(members, axis=0)
        for name, extra in [("threshold", []), ("threshold_with_buoy", EXTRA)]:
            comparisons[name] = threshold_probabilities(data, splits["train"], prior, extra)
        for name, prob in comparisons.items():
            for split in ["validation", "progress_overlap", "test"]:
                mask = splits[split]
                target = data.loc[mask, "target_category"].to_numpy()
                scores.append(dict(hour=hour, split=split, model=name, train_days=len(train),
                                   complete_buoy_days=int(marine.loc[mask, EXTRA].notna().all(axis=1).sum()),
                                   **metrics(target, prob[mask])))
                rows = pd.DataFrame(prob[mask], columns=PROBS, index=data.index[mask])
                predictions.append(rows.assign(hour=hour, model=name, split=split, target=target))
    pd.concat(features).to_csv(OUT / "features.csv.gz", compression={"method": "gzip", "mtime": 0})
    pd.concat(predictions).to_csv(OUT / "predictions.csv.gz", compression={"method": "gzip", "mtime": 0})
    summary = pd.DataFrame(scores)
    summary.to_csv(OUT / "metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download missing public NOAA archives.")
    with threadpool_limits(limits=2):
        run(parser.parse_args().fetch)
