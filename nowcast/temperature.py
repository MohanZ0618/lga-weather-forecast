"""Point forecasts and day-ahead probabilities for the LGA daily hourly maximum."""
from datetime import timedelta
from functools import lru_cache
import json

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.special import ndtr

from analogue import augment
from features import BOUNDS, TZ, fusion, ground, labels
from runtime import metar_tables
from sources import CACHE, HERE, get, point, save

MODEL = HERE / "artifacts/temperature.joblib"
FINE_BOUNDS = np.arange(25, 100, 5)
FINE_LABELS = [f"< {FINE_BOUNDS[0]}"] + [f"{a}–< {b}" for a,b in zip(FINE_BOUNDS,FINE_BOUNDS[1:])] + [f"≥ {FINE_BOUNDS[-1]}"]
FINE_COLUMNS = [f"fine_{i}" for i in range(len(FINE_LABELS))]


def issue_time(day):
    previous = pd.Timestamp(day).date() - timedelta(days=1)
    return (pd.Timestamp(previous) + pd.Timedelta(hours=12)).tz_localize(TZ)


def target_cycle(day):
    return pd.Timestamp(issue_time(day).date(), tz="UTC") + pd.Timedelta(hours=12)


def next_day_high(cycle, day):
    issued = issue_time(day)
    if not cycle or cycle.get("missing"):
        return np.nan
    run = pd.Timestamp(cycle["run"])
    if run != target_cycle(day) or not run <= pd.Timestamp(cycle["available"]) <= issued:
        return np.nan
    start = pd.Timestamp(day).tz_localize(TZ)
    end = start + pd.DateOffset(days=1)
    required = pd.date_range(start, end, inclusive="left", freq="h")
    leads = ((required - run) / pd.Timedelta(hours=1)).astype(int)
    if leads.min() < 1 or leads.max() > len(cycle["values"]):
        return np.nan
    return float(np.max(np.asarray(cycle["values"])[leads - 1]))


def day_features(stations, day, cycle):
    row, _ = ground(stations, issue_time(day))
    if row is None or not np.isfinite(row["high_so_far"]):
        return None
    return {**row, "tomorrow_hrrr": next_day_high(cycle, day)}


def anchor(frame, tier):
    if tier == "day_hrrr":
        return frame.tomorrow_hrrr.to_numpy()
    if tier == "day_station":
        return frame.LGA_temp.to_numpy()
    prior = frame.high_so_far + frame.expected_rise
    return np.maximum(frame.high_so_far, frame.nwp_high.fillna(prior)).to_numpy()


def predict_points(frame, tier, spec):
    values = anchor(frame, tier)
    if spec["estimator"] is not None:
        values = values + spec["estimator"].predict(frame[spec["columns"]])
    if tier == "rolling":
        values = np.maximum(values, frame.high_so_far)
    return values


def day_probabilities(values, residuals):
    # Residuals from a separate calibration year. Strict inequality assigns ties
    # at 40, 55, 70 and 85 to the same upper band used for the observed labels.
    counts = np.searchsorted(residuals, BOUNDS[None, :] - values[:, None], side="left")
    bands = np.diff(np.column_stack([np.zeros(len(values)), counts,
                                      np.full(len(values), len(residuals))]), axis=1)
    return (bands + 0.5) / (len(residuals) + 2.5)


def fine_probabilities(points, floors, broad, residuals):
    """Allocate each published band among its 5°F bins using calibration errors."""
    points, floors = np.asarray(points), np.asarray(floors)
    centers = points[:,None] + residuals
    # A 1°F Gaussian kernel smooths the empirical error distribution.
    cdf = ndtr(FINE_BOUNDS[None,None,:] - centers[:,:,None]).mean(axis=1)
    cdf = np.where(FINE_BOUNDS <= floors[:,None], 0., cdf)
    mass = np.diff(np.column_stack([np.zeros(len(points)),cdf,np.ones(len(points))]),axis=1)
    possible = np.r_[FINE_BOUNDS, np.inf] > floors[:,None]
    mass = np.where(possible, np.maximum(mass,1e-12), 0.)
    parents = np.searchsorted(BOUNDS, np.r_[-np.inf,FINE_BOUNDS], side="right")
    for i in range(len(BOUNDS)+1):
        section = parents == i
        total = mass[:,section].sum(axis=1,keepdims=True)
        mass[:,section] *= np.divide(broad[:,i,None],total,out=np.zeros_like(total),where=total>0)
    return mass


@lru_cache(maxsize=1)
def model():
    return joblib.load(MODEL)


def day_prediction(day, records, cycle):
    row = day_features(metar_tables(records), day, cycle)
    if row is None:
        return None
    tier = "day_hrrr" if np.isfinite(row["tomorrow_hrrr"]) else "day_station"
    spec = model()["tiers"][tier]
    value = predict_points(pd.DataFrame([row]), tier, spec)
    probs = day_probabilities(value, spec["residuals"])[0]
    return {"day": str(day), "issued": issue_time(day).isoformat(), "high": float(value[0]),
            "tier": tier, "method": spec["method"], "probabilities": probs.tolist(),
            "hrrr_run": cycle["run"] if tier == "day_hrrr" else None}


def recent_metars(hours=48):
    response = get("https://aviationweather.gov/api/data/metar",
                   params={"ids":"KLGA,KJFK,KEWR,KTEB","format":"json","hours":hours})
    response.raise_for_status()
    return response.json() if response.status_code != 204 else []


def live_day(day, now, records=None):
    if issue_time(day) > now:
        return None
    path = CACHE / "day_ahead" / model()["version"] / f"{day}.json"
    if path.exists():
        return json.loads(path.read_text())
    if records is None:
        records = recent_metars()
    try:
        cycle = point(target_cycle(day))
    except (requests.RequestException, ValueError, OSError):
        cycle = None
    result = day_prediction(day, records, cycle)
    if result is None and pd.Timedelta(hours=30) < now-issue_time(day) < pd.Timedelta(hours=62):
        result = day_prediction(day, recent_metars(72), cycle)
    if result:
        result["reconstructed_at"] = pd.Timestamp.now(tz=TZ).isoformat()
        save(path, result)
    return result


def rolling_point(at, payload, reference):
    base, _ = ground(metar_tables(payload["metars"]), at)
    if base is None or not np.isfinite(base["high_so_far"]):
        return None
    row = augment(fusion(at, base, payload["external"]), at, reference)
    return float(predict_points(pd.DataFrame([row]), "rolling", model()["tiers"]["rolling"])[0])


def observations(records, day, as_of):
    table = metar_tables(records)["LGA"]
    table = table.loc[(table.index.date == day) & (table.index <= as_of) & (table.received <= as_of)]
    hourly = table.groupby(table.index.tz_convert("UTC").floor("h")).tail(1)
    end = pd.Timestamp(day).tz_localize(TZ) + pd.DateOffset(days=1)
    has_last_hour = len(hourly) and hourly.index[-1] >= end - pd.Timedelta(hours=1)
    actual = labels(hourly).get(day) if has_last_hour and as_of >= end + pd.Timedelta(hours=1) else None
    return hourly.temp.dropna(), actual
