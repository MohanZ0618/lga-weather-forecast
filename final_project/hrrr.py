"""Extract the archived 10 UTC HRRR 2-meter temperature forecast near LGA."""

from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
import hashlib
from pathlib import Path
from threading import local
import time

from numcodecs.blosc import decompress
import numpy as np
import pandas as pd
import requests

from prepare import DATA, END, HOURS, local_cutoff

BUCKET = "https://hrrrzarr.s3.us-west-1.amazonaws.com"
# Nearest grid point to LGA (40.77945, -73.87397). The official grid
# index and Lambert conformal projection give y=698 and x=1556.
GRID_POINT = (698, 1556)
TILE_SIZE = 150
START = "2021-01-01"
HTTP = local()


def download(day):
    date = day.strftime("%Y%m%d")
    y_tile, y = divmod(GRID_POINT[0], TILE_SIZE)
    x_tile, x = divmod(GRID_POINT[1], TILE_SIZE)
    level = "2m_above_ground/TMP"
    url = f"{BUCKET}/sfc/{date}/{date}_10z_fcst.zarr/{level}/{level}/0.{y_tile}.{x_tile}"
    if not hasattr(HTTP, "session"):
        HTTP.session = requests.Session()
    for attempt in range(3):
        try:
            response = HTTP.session.get(url, timeout=35)
            if response.status_code == 404:
                return {"date": day, "url": url, "status": "missing"}
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                return {"date": day, "url": url, "status": "request_error"}
            time.sleep(2**attempt)
    modified = pd.Timestamp(parsedate_to_datetime(response.headers["Last-Modified"]))
    initial = day.tz_localize("UTC") + pd.Timedelta(hours=10)
    record = {"date": day, "url": url, "archived_utc": modified,
              "sha256": hashlib.sha256(response.content).hexdigest()}
    if not initial <= modified <= local_cutoff(day, HOURS[0]):
        return {**record, "status": "outside_issue_window"}
    size = response.content[3]
    if size not in (2, 4):
        raise ValueError(f"Unexpected float size for {url}: {size}")
    grid = np.frombuffer(decompress(response.content), dtype=f"<f{size}").reshape(18, TILE_SIZE, TILE_SIZE)
    values = (grid[:, y, x].astype(float) - 273.15) * 1.8 + 32
    if not np.all(np.isfinite(values) & (values > -90) & (values < 150)):
        return {**record, "status": "invalid_values"}
    return {**record, "status": "ok", **{f"f{lead:02}": value for lead, value in enumerate(values, 1)}}


def forecast_features(day, hour, archived, temperatures):
    cutoff = local_cutoff(day, hour)
    if not day.tz_localize("UTC") + pd.Timedelta(hours=10) <= archived <= cutoff:
        raise ValueError("Forecast was outside the issue window")
    next_midnight = local_cutoff(day + pd.Timedelta(days=1), 0)
    initialization = day.tz_localize("UTC") + pd.Timedelta(hours=10)
    valid_times = pd.date_range(initialization + pd.Timedelta(hours=1), periods=18, freq="h")
    future = (valid_times >= cutoff) & (valid_times < next_midnight)
    at_cutoff = temperatures[valid_times == cutoff].item()
    maximum = temperatures[future].max()
    return maximum, maximum - at_cutoff


def main():
    cache = Path(DATA / "hrrr_cycles.csv.gz")
    previous = pd.read_csv(cache, parse_dates=["date"]) if cache.exists() else pd.DataFrame()
    keep = previous.loc[~previous.status.eq("request_error")] if len(previous) else previous
    days = pd.date_range(START, END)
    if len(keep):
        days = days.difference(keep.date)
    records = keep.to_dict("records")
    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, record in enumerate(pool.map(download, days), 1):
            records.append(record)
            if count % 100 == 0:
                pd.DataFrame(records).to_csv(cache, index=False)
                print(f"HRRR {count}/{len(days)} dates checked", flush=True)
    archive = pd.DataFrame(records).sort_values("date")
    archive.to_csv(cache, index=False)
    predictors = []
    for row in archive.loc[archive.status.eq("ok")].to_dict("records"):
        temps = np.array([row[f"f{lead:02}"] for lead in range(1, 19)])
        for hour in HOURS:
            maximum, warming = forecast_features(row["date"], hour, pd.Timestamp(row["archived_utc"]), temps)
            predictors.append({"date": row["date"], "hour": hour, "hrrr_remaining_max": maximum,
                               "hrrr_warming": warming})
    pd.DataFrame(predictors).to_csv(DATA / "hrrr_features.csv.gz", index=False)
    print(archive.status.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
