"""Build the final study's dated feature tables from NOAA station archives."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from weather import (HOURS, TZ, daily_labels, hourly_rows, local_cutoff,
                     make_features, previous_day_maxima)

DATA = ROOT / "data"
LCD_URL = "https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access"
STATION_IDS = {
    "LGA": "USW00014732", "JFK": "USW00094789",
    "EWR": "USW00014734", "TEB": "USW00094741",
}
START = "2018-01-01"
END = "2026-08-31"
FIELDS = {
    "HourlyDryBulbTemperature": "temp", "HourlyRelativeHumidity": "rh",
    "HourlyWindSpeed": "wind",
}
EASTERN_STANDARD = timezone(timedelta(hours=-5))


def lcd_local_time(values):
    """LCD CSV timestamps use fixed local standard time throughout the year."""
    return pd.to_datetime(values).dt.tz_localize(EASTERN_STANDARD).dt.tz_convert(TZ)


def fetch_station(key):
    station, year = key
    name = f"LCD_{STATION_IDS[station]}_{year}.csv"
    path = DATA / "raw" / name
    url = f"{LCD_URL}/{year}/{name}"
    if not path.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        path.with_suffix(".part").write_bytes(response.content)
        path.with_suffix(".part").replace(path)
    frame = pd.read_csv(path, usecols=["DATE", "REPORT_TYPE", *FIELDS], dtype="string")
    frame = frame.loc[frame["REPORT_TYPE"].isin(["FM-15", "FM-16"])].copy()
    frame.index = lcd_local_time(frame["DATE"])
    frame = frame.rename(columns=FIELDS)[list(FIELDS.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    # ASOS measures whole degrees Fahrenheit, and METAR and LCD store that reading in tenths of a degree
    # Celsius (40 F is stored as 4.4 C). Rounding the conversion recovers the sensor's whole degree;
    # without it 40 F would read 39.92 F and fall into the colder band. Special reports without a tenths
    # group carry whole degrees Celsius, and rounding gives the nearest whole degree Fahrenheit.
    frame["temp"] = (frame["temp"].where(frame["temp"].between(-75, 60)) * 1.8 + 32).round()
    frame["rh"] = frame["rh"].where(frame["rh"].between(0, 100))
    frame["wind"] = frame["wind"].where(frame["wind"].between(0, 120))
    meta = {
        "station": station, "source_year": year, "url": url,
        "sha256": sha256(path.read_bytes()).hexdigest(), "reports": len(frame),
        "first": str(frame.index.min()), "last": str(frame.index.max()),
    }
    return station, frame, meta


def station_features(observations, maxima, day, hour):
    cutoff = local_cutoff(day, hour)
    # Each station sees only the observations available by the cutoff.
    features, _ = make_features(observations, maxima, observations, day, hour)
    for station, table in observations.items():
        closed_hours = table.loc[local_cutoff(day, 0):cutoff - pd.Timedelta(nanoseconds=1)]
        features[f"{station}_max_so_far"] = hourly_rows(closed_hours)["temp"].max()
        for lag in (1, 3):
            reference = cutoff - pd.Timedelta(hours=lag)
            before = table.loc[reference - pd.Timedelta(hours=1):reference, "temp"].dropna()
            latest = table.loc[cutoff - pd.Timedelta(hours=1):cutoff, "temp"].dropna()
            if len(before) and len(latest):
                elapsed = (latest.index[-1] - before.index[-1]).total_seconds() / 3600
                if elapsed > 0:
                    features[f"{station}_trend_{lag}h"] = (latest.iloc[-1] - before.iloc[-1]) / elapsed
                    continue
            features[f"{station}_trend_{lag}h"] = np.nan
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-labels", type=Path, default=ROOT.parent / "data/processed/lga_daily_labels.csv")
    args = parser.parse_args()
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    stations = {station: [] for station in STATION_IDS}
    manifest = []
    keys = [(station, year) for station in stations for year in range(2018, 2027)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        for station, frame, meta in workers.map(fetch_station, keys):
            stations[station].append(frame)
            manifest.append(meta)
            print(f"{station} {meta['source_year']} {len(frame)} reports", flush=True)
    stations = {station: pd.concat(tables).sort_index() for station, tables in stations.items()}
    stations = {station: table[~table.index.duplicated(keep="last")] for station, table in stations.items()}
    maxima = {station: previous_day_maxima(table) for station, table in stations.items()}
    labels = daily_labels(hourly_rows(stations["LGA"])).loc[START:END]
    pd.DataFrame(manifest).to_csv(DATA / "station_sources.csv", index=False)
    if args.legacy_labels.exists():
        legacy = pd.read_csv(args.legacy_labels, index_col="date", parse_dates=True)
        # Global Hourly stores the same tenths-Celsius values, so round it the same way before comparing.
        legacy["target_max"] = legacy["target_max"].round()
        legacy["target_category"] = np.searchsorted([40, 55, 70, 85], legacy["target_max"], side="right")
        labels.loc["2024"].join(legacy.add_prefix("legacy_"), how="inner").to_csv(DATA / "archive_overlap_2024.csv")
        hourly_path = args.legacy_labels.with_name("LGA_2024_hourly.csv")
        if hourly_path.exists():
            hourly = pd.read_csv(hourly_path, usecols=["timestamp", "temp"])
            hourly["temp"] = hourly["temp"].round()
            hourly.index = pd.to_datetime(hourly.pop("timestamp"), utc=True)
            overlap = stations["LGA"].loc["2024", ["temp"]].join(
                hourly.rename(columns={"temp": "legacy_temp"}), how="inner"
            ).dropna()
            checks = [{"pairs": len(overlap),
                       "mean_abs_difference_f": (overlap.temp - overlap.legacy_temp).abs().mean()}]
            pd.DataFrame(checks).to_csv(DATA / "timestamp_overlap_2024.csv", index=False)
    for hour in HOURS:
        rows = [station_features(stations, maxima, day, hour) for day in labels.index]
        samples = pd.DataFrame(rows, index=labels.index).join(labels)
        samples.to_csv(DATA / f"samples_{hour:02}.csv.gz", index_label="date",
                       compression={"method": "gzip", "mtime": 0})
        print(f"{hour:02}:00: {len(samples)} eligible dates", flush=True)


if __name__ == "__main__":
    main()
