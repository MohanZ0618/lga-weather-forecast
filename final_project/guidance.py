"""Station guidance for KLGA from the Iowa Environmental Mesonet MOS archive: LAMP, NBM text and GFS MOS.

`python final_project/guidance.py fetch` downloads one CSV per model and calendar year (public IEM endpoint,
sequential, 3 s apart) into data/raw/iem_mos/ and records URL, rows and SHA-256 in data/guidance_sources.csv.
`python final_project/guidance.py build` turns them into data/guidance_features.csv.gz: for every date and
cutoff, the newest run of each product that had been issued by the cutoff, and its forecast of the highest
temperature still to come today. The bundled feature table lets experiments run offline.

Products (all are statistical post-processing at the station itself, so they are already bias-corrected):
- LAV  LAMP, hourly temperature forecasts for the next 25 hours. Archive starts 2020-07-11.
- NBS  National Blend of Models short-range text, 3-hourly temperatures and 12-hour max/min. From 2018-11.
- GFS  GFS MOS (MAV), 3-hourly temperatures and 12-hour max/min. From 2016.

Availability. The archive has no receipt times, so a run counts only after a conservative delay from its
initialization: LAMP 55 min, NBS 66 min, GFS MOS 5 h. These are assumptions, documented as such.
"""

import argparse
import csv
import hashlib
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import DATA, END, HOURS, local_cutoff

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATION = "KLGA"
FIRST_YEAR = {"LAV": 2020, "NBS": 2018, "GFS": 2018}
DELAY = {"LAV": pd.Timedelta(minutes=55), "NBS": pd.Timedelta(minutes=66), "GFS": pd.Timedelta(hours=5)}
RAW = DATA / "raw" / "iem_mos"
SOURCES = DATA / "guidance_sources.csv"


def fetch(end="2026-09-01T00:00Z"):
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    last = int(end[:4])
    for model, first in FIRST_YEAR.items():
        for year in range(first, last + 1):
            sts, ets = f"{year}-01-01T00:00Z", (f"{year + 1}-01-01T00:00Z" if year < last else end)
            url = f"{URL}?{urllib.parse.urlencode(dict(station=STATION, model=model, sts=sts, ets=ets, format='csv'))}"
            path = RAW / model / f"{STATION}-{year}.csv"
            if not path.exists():
                body = None
                for attempt in range(5):
                    try:
                        body = urllib.request.urlopen(url, timeout=300).read()
                        break
                    except OSError:
                        time.sleep(15 * (attempt + 1))
                if body is None:
                    raise RuntimeError(f"IEM request failed: {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_suffix(".part").write_bytes(body)
                path.with_suffix(".part").replace(path)
                time.sleep(3)
            body = path.read_bytes()
            rows = list(csv.reader(io.StringIO(body.decode())))[1:]
            runs = sorted({row[0] for row in rows if row})
            records.append({"model": model, "year": year, "url": url, "rows": len(rows),
                            "first_run": runs[0] if runs else "", "last_run": runs[-1] if runs else "",
                            "sha256": hashlib.sha256(body).hexdigest()})
            print(model, year, len(rows), flush=True)
    pd.DataFrame(records).to_csv(SOURCES, index=False)


CLOUD = {"CL": 0.0, "FW": 0.19, "SC": 0.44, "BK": 0.75, "OV": 1.0}  # category midpoints as sky fraction


def load(model):
    frames = [pd.read_csv(path, low_memory=False) for path in sorted((RAW / model).glob(f"{STATION}-*.csv"))]
    table = pd.concat(frames, ignore_index=True)
    for column in ("runtime", "ftime"):
        table[column] = pd.to_datetime(table[column]).dt.tz_localize("UTC")
    table = table.drop_duplicates(["runtime", "ftime"], keep="last").sort_values(["runtime", "ftime"])
    return {run: rows for run, rows in table.groupby("runtime")}


def newest_run(runs, starts, model, cutoff, oldest):
    """Latest run whose assumed availability (initialization + delay) is at or before the cutoff."""
    position = starts.searchsorted(cutoff - DELAY[model], side="right") - 1
    while position >= 0 and starts[position] >= oldest:
        yield starts[position], runs[starts[position]]
        position -= 1


def lamp_features(runs, starts, cutoff, midnight, evening):
    for run, rows in newest_run(runs, starts, "LAV", cutoff, cutoff - pd.Timedelta(hours=12)):
        temp = pd.to_numeric(rows["tmp"], errors="coerce")
        rest = rows["ftime"].ge(cutoff) & rows["ftime"].lt(midnight) & temp.notna()
        now = rows["ftime"].eq(cutoff) & temp.notna()
        if not rest.any() or not now.any():
            continue
        day = rows["ftime"].ge(cutoff) & rows["ftime"].le(evening)
        direction = np.deg2rad(pd.to_numeric(rows.loc[day, "wdr"], errors="coerce"))
        speed = pd.to_numeric(rows.loc[day, "wsp"], errors="coerce")
        return {"lamp_run": run, "lamp_age_h": (cutoff - run) / pd.Timedelta(hours=1),
                "lamp_remaining_max": temp[rest].max(), "lamp_now": temp[now].iloc[0],
                "lamp_cloud": rows.loc[day, "cld"].astype("string").str.strip().map(CLOUD).mean(),
                # Wind blows from `direction`: u > 0 is a west wind, v > 0 a south wind.
                "lamp_u": (-speed * np.sin(direction)).mean(), "lamp_v": (-speed * np.cos(direction)).mean()}
    return {}


def nbm_features(runs, starts, cutoff, day_end):
    """Daytime maximum (the 12-hour period ending 00 UTC) and its NBM standard deviation."""
    for run, rows in newest_run(runs, starts, "NBS", cutoff, cutoff - pd.Timedelta(hours=24)):
        hit = rows.loc[rows["ftime"].eq(day_end), ["txn", "xnd"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(hit):
            return {"nbm_run": run, "nbm_max": hit["txn"].iloc[0], "nbm_max_sd": hit["xnd"].iloc[0]}
    return {}


def build(extra_delay=0, out=DATA / "guidance_features.csv.gz"):
    """extra_delay (minutes) is added to every assumed availability delay, for a timing stress test."""
    for model in DELAY:
        DELAY[model] += pd.Timedelta(minutes=extra_delay)
    lamp, nbm = load("LAV"), load("NBS")
    lamp_starts, nbm_starts = pd.DatetimeIndex(sorted(lamp)), pd.DatetimeIndex(sorted(nbm))
    records = []
    for day in pd.date_range("2020-07-12", END):
        midnight = local_cutoff(day + pd.Timedelta(days=1), 0).tz_convert("UTC")
        evening = local_cutoff(day, 18).tz_convert("UTC")
        day_end = (day + pd.Timedelta(days=1)).tz_localize("UTC")
        for hour in HOURS:
            cutoff = local_cutoff(day, hour).tz_convert("UTC")
            records.append({"date": day, "hour": hour,
                            **lamp_features(lamp, lamp_starts, cutoff, midnight, evening),
                            **nbm_features(nbm, nbm_starts, cutoff, day_end)})
    table = pd.DataFrame(records)
    table.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    coverage = table.groupby([table.date.dt.year, "hour"])[["lamp_remaining_max", "nbm_max"]].count()
    print(coverage.unstack("hour").to_string(), flush=True)
    print(table.groupby("hour")[["lamp_age_h"]].describe().round(2).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fetch", "build"])
    parser.add_argument("--extra-delay", type=int, default=0, help="minutes added to every availability delay")
    parser.add_argument("--out", type=Path, default=DATA / "guidance_features.csv.gz")
    args = parser.parse_args()
    if args.step == "fetch":
        fetch()
    else:
        build(args.extra_delay, args.out)


if __name__ == "__main__":
    main()
