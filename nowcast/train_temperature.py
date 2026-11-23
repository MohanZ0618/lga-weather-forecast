"""Train temperature estimators, calibrate day-ahead bands, and replay both horizons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from features import BOUNDS, archive_stations, labels
from sources import CACHE, HERE, point
from temperature import (FINE_BOUNDS, FINE_COLUMNS, MODEL, anchor, day_features, day_probabilities,
                         fine_probabilities, issue_time, predict_points, target_cycle)
from models import score

OUT = HERE / "results"
TEST = "2026-09-01"


def cycle_for(day, fetch):
    run = target_cycle(day)
    cached = CACHE / "cycles" / f"{run:%Y%m%d%H}-TMP.json"
    if cached.exists():
        return json.loads(cached.read_text())
    if fetch and (str(day) >= TEST or day.toordinal() % 3 == 0) and day.year >= 2021:
        return point(run)
    return None


def daily_table(stations, truth, fetch):
    days = list(truth.index)
    with ThreadPoolExecutor(max_workers=8) as pool:
        cycles = list(pool.map(lambda d: cycle_for(d, fetch), days))
    rows = []
    for day, cycle in zip(days, cycles):
        row = day_features(stations, day, cycle)
        if row is not None:
            rows.append({**row, "day": str(day), "actual": truth[day]})
    return pd.DataFrame(rows)


def fit(frame, tier, columns):
    train = frame.loc[frame.day < "2025-01-01"]
    val = frame.loc[frame.day.between("2025-01-01", "2025-08-31")]
    cal = frame.loc[frame.day.between("2025-09-01", "2026-08-31")]
    if min(map(len, (train, val, cal))) < 20:
        raise ValueError(f"Insufficient historical samples for {tier}: {len(train)}/{len(val)}/{len(cal)}")
    candidates = {"baseline": None, "ridge": make_pipeline(SimpleImputer(keep_empty_features=True),
                  StandardScaler(), Ridge(alpha=30)),
                  "gradient_boosting": HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=12,
                  min_samples_leaf=15, l2_regularization=3, random_state=4260)}
    target = train.actual - anchor(train, tier)
    specs, scores = {}, []
    for name, estimator in candidates.items():
        if estimator is not None:
            estimator.fit(train[columns], target)
        spec = {"method": name, "columns": columns, "estimator": estimator}
        specs[name] = spec
        diff = val.assign(error=np.abs(predict_points(val, tier, spec) - val.actual))
        scores.append({"tier":tier, "method":name, "validation_mae":diff.groupby("day").error.mean().mean(),
                       "train_rows":len(train), "validation_rows":len(val), "calibration_rows":len(cal)})
    best = specs[min(scores, key=lambda r: r["validation_mae"])["method"]]
    best["residuals"] = np.sort(cal.actual.to_numpy() - predict_points(cal, tier, best))
    return best, scores


def train(fetch):
    stations = archive_stations()
    truth = labels(stations["LGA"])
    print("Loading previous-day cycles for", len(truth), "target days", flush=True)
    daily = daily_table(stations, truth, fetch)
    rolling = pd.read_parquet(OUT / "sampled.parquet", filters=[("delay", "==", 0)])
    base_columns = [c for c in daily if c not in ("day","actual","tomorrow_hrrr","direction")]
    roll_columns = base_columns + ["expected_rise", "nwp_high", "nwp_warm", "nwp_bias",
                                   "sat10", "sat40", "upwind", "delta30", "clear_sun"]
    tiers, rows = {}, []
    for tier, frame, columns in [("day_station",daily,base_columns),
                                 ("day_hrrr",daily.dropna(subset=["tomorrow_hrrr"]),base_columns+["tomorrow_hrrr"]),
                                 ("rolling",rolling,roll_columns)]:
        tiers[tier], grid = fit(frame, tier, columns)
        rows.extend(grid)
    pd.DataFrame(rows).to_csv(OUT/"temperature_validation.csv",index=False)
    limits = (daily.day >= TEST) & (daily.day <= rolling.day.max())
    days = daily.loc[limits].copy()
    days["tier"] = np.where(days.tomorrow_hrrr.notna(), "day_hrrr", "day_station")
    prob_cols = [f"p{i}" for i in range(5)]
    for tier, sub in days.groupby("tier"):
        values = predict_points(sub, tier, tiers[tier])
        days.loc[sub.index, "forecast"] = values
        days.loc[sub.index, prob_cols] = day_probabilities(values,tiers[tier]["residuals"])
        days.loc[sub.index, FINE_COLUMNS] = fine_probabilities(values,np.full(len(values),-np.inf),
            days.loc[sub.index,prob_cols].to_numpy(),tiers[tier]["residuals"])
    held = rolling.loc[rolling.day >= TEST].copy()
    held["forecast"] = predict_points(held,"rolling",tiers["rolling"])
    probabilities = pd.read_parquet(OUT/"predictions.parquet",filters=[("delay","==",0),("level","==","deployed")])
    held = held.merge(probabilities[["cutoff",*prob_cols]], on="cutoff", validate="one_to_one")
    held[FINE_COLUMNS] = fine_probabilities(held.forecast.to_numpy(),held.high_so_far.to_numpy(),
                                           held[prob_cols].to_numpy(),tiers["rolling"]["residuals"])
    common = sorted(set(days.day) & set(held.day))
    days, held = days.loc[days.day.isin(common)].copy(), held.loc[held.day.isin(common)].copy()
    days["cutoff"] = days.day.map(lambda d: issue_time(d).isoformat())
    days["horizon"], held["horizon"] = "Day ahead", "Same day"
    output = pd.concat([days,held],ignore_index=True)
    output["error"] = output.forecast - output.actual
    output["label"] = np.searchsorted(BOUNDS,output.actual,side="right")
    summary = []
    def metrics(name, frame):
        stats = score(frame,frame[prob_cols].to_numpy())
        daily_error = frame.assign(abserr=frame.error.abs(),sq=frame.error**2).groupby("day")
        fine = frame[FINE_COLUMNS].to_numpy()
        fine_y = np.searchsorted(FINE_BOUNDS,frame.actual,side="right")
        detail = pd.DataFrame({"day":frame.day, "fine_accuracy":fine.argmax(axis=1)==fine_y,
            "fine_log_loss":-np.log(fine[np.arange(len(fine)),fine_y].clip(1e-15))}).groupby("day").mean().mean()
        summary.append({"horizon":name, **stats, **detail.to_dict(), "mae":daily_error.abserr.mean().mean(),
                        "rmse":np.sqrt(daily_error.sq.mean().mean()), "bias":daily_error.error.mean().mean()})
    for horizon, frame in output.groupby("horizon"):
        metrics(horizon,frame)
    for name, mask in [("Same day 06–10",held.hour < 10),("Same day 10–15",held.hour.between(10,15,inclusive="left")),
                       ("Same day 15–20",held.hour >= 15)]:
        sub = output.loc[output.horizon.eq("Same day") & output.cutoff.isin(held.loc[mask,"cutoff"])]
        if len(sub):
            metrics(name,sub)
    output[["day","cutoff","horizon","tier","forecast","actual","error",*prob_cols,*FINE_COLUMNS]].to_parquet(
        OUT/"temperature_replay.parquet",index=False)
    obs = stations["LGA"]
    obs = obs.groupby(obs.index.tz_convert("UTC").floor("h")).tail(1)
    obs.loc[np.isin(obs.index.strftime("%Y-%m-%d"), common),["temp"]].to_parquet(OUT/"temperature_observations.parquet")
    metrics_df = pd.DataFrame(summary)
    metrics_df.to_csv(OUT/"temperature_metrics.csv",index=False)
    print(metrics_df.to_string(index=False), flush=True)
    # Include the serialized estimators and calibration data in the cache identity.
    version = joblib.hash(tiers)[:12]
    temporary = MODEL.with_suffix(".tmp")
    joblib.dump({"tiers":tiers,"version":version},temporary,compress=3)
    temporary.replace(MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-weather", action="store_true", help="Fill the sampled historical 12 UTC HRRR archive")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(args.fetch_weather)
