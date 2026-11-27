"""Newest HRRR run released by each cutoff, instead of the fixed 10 UTC run, at the LGA grid point.

hrrr.py reads only the 10 UTC run, which is about 3 h old at 09:00 New York time, 6 h at noon and 9 h at
15:00. HRRR runs every hour (18 one-hour leads, 48 at 00/06/12/18 UTC). For each date and cutoff this
module tries the runs initialized 1, 2, ..., 6 h before the cutoff, newest first. A run counts only if its
leads reach the last hour of the local day; the first one whose 2 m temperature tile has an S3
Last-Modified at or before the cutoff is kept (the availability rule of hrrr.py). Same Utah hrrrzarr
archive, same tile and grid point (698, 1556) and the same decoding as hrrr.py.

  python hrrr_fresh.py fetch [--start YYYY-MM-DD --end YYYY-MM-DD]
      HEADs the candidates, GETs only the chosen tile, appends one JSON line per (date, cutoff) to
      data/raw/hrrr_fresh/records.jsonl with the decoded point series. Resumable: rerunning skips keys
      already recorded, except request errors. Progress goes to data/raw/hrrr_fresh/progress.log.
  python hrrr_fresh.py build
      writes data/hrrr_fresh_cycles.csv.gz (run, URL, Last-Modified, SHA-256, status, point temperatures)
      and data/hrrr_fresh_features.csv.gz (hrrr_fresh_remaining_max, hrrr_fresh_warming, hrrr_fresh_now,
      hrrr_fresh_age_h, hrrr_fresh_run_hour).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
import hashlib
import json
from threading import local
import time

from numcodecs.blosc import decompress
import numpy as np
import pandas as pd
import requests

from hrrr import BUCKET, GRID_POINT, START, TILE_SIZE
from prepare import DATA, END, HOURS, local_cutoff

RAW = DATA / "raw" / "hrrr_fresh"
RECORDS = RAW / "records.jsonl"
PROGRESS = RAW / "progress.log"
CYCLES = DATA / "hrrr_fresh_cycles.csv.gz"
FEATURES = DATA / "hrrr_fresh_features.csv.gz"
MAX_AGE = 6  # hours between the newest and oldest candidate initialization
MAX_LEADS = 48
WORKERS = 8
HTTP = local()


def cutoff_utc(day, hour):
    return local_cutoff(pd.Timestamp(day), hour).tz_convert("UTC")


def leads_for(init):
    """hrrrzarr fcst.zarr holds leads 1..18, or 1..48 for the synoptic runs (checked on 2021-2026 dates)."""
    return 48 if init.hour % 6 == 0 else 18


def valid_times(init, leads):
    return pd.date_range(init + pd.Timedelta(hours=1), periods=leads, freq="h")


def day_hours(day, hour):
    """UTC valid hours from the cutoff through the last hour of the local day."""
    cutoff = cutoff_utc(day, hour)
    midnight = local_cutoff(pd.Timestamp(day) + pd.Timedelta(days=1), 0).tz_convert("UTC")
    return pd.date_range(cutoff, midnight - pd.Timedelta(hours=1), freq="h")


def covers_day(init, leads, day, hour):
    return bool(day_hours(day, hour).isin(valid_times(init, leads)).all())


def candidate_runs(day, hour):
    """Initializations from cutoff - 1 h back to cutoff - MAX_AGE h whose leads reach local midnight."""
    cutoff = cutoff_utc(day, hour)
    runs = [cutoff - pd.Timedelta(hours=age) for age in range(1, MAX_AGE + 1)]
    return [init for init in runs if covers_day(init, leads_for(init), day, hour)]


def run_url(init):
    date = init.strftime("%Y%m%d")
    y_tile, _ = divmod(GRID_POINT[0], TILE_SIZE)
    x_tile, _ = divmod(GRID_POINT[1], TILE_SIZE)
    level = "2m_above_ground/TMP"
    return f"{BUCKET}/sfc/{date}/{date}_{init.hour:02}z_fcst.zarr/{level}/{level}/0.{y_tile}.{x_tile}"


def available(init, archived, cutoff):
    """The hrrr.py rule: archived after initialization and no later than the cutoff."""
    return init <= archived <= cutoff


def features(day, hour, init, temperatures):
    """Forecast at the cutoff hour, highest forecast for the rest of the local day, warming, run age."""
    temperatures = np.asarray(temperatures, dtype=float)
    cutoff = cutoff_utc(day, hour)
    times = valid_times(init, len(temperatures))
    hours = day_hours(day, hour)
    if not hours.isin(times).all():
        raise ValueError("Run does not reach the end of the local day")
    now = temperatures[times == cutoff].item()
    maximum = temperatures[times.isin(hours)].max()
    return {"hrrr_fresh_remaining_max": maximum, "hrrr_fresh_warming": maximum - now, "hrrr_fresh_now": now,
            "hrrr_fresh_age_h": (cutoff - init) / pd.Timedelta(hours=1), "hrrr_fresh_run_hour": init.hour}


def decode(content, leads):
    """Same as hrrr.py: blosc chunk of (leads, 150, 150) little-endian floats, Kelvin to Fahrenheit."""
    y = GRID_POINT[0] % TILE_SIZE
    x = GRID_POINT[1] % TILE_SIZE
    size = content[3]
    if size not in (2, 4):
        raise ValueError(f"Unexpected float size: {size}")
    flat = np.frombuffer(decompress(content), dtype=f"<f{size}")
    if flat.size != leads * TILE_SIZE * TILE_SIZE:
        raise ValueError(f"Expected {leads} leads, got {flat.size / TILE_SIZE**2:g}")
    return (flat.reshape(leads, TILE_SIZE, TILE_SIZE)[:, y, x].astype(float) - 273.15) * 1.8 + 32


def request(method, url):
    if not hasattr(HTTP, "session"):
        HTTP.session = requests.Session()
    for attempt in range(4):
        try:
            response = HTTP.session.request(method, url, timeout=35)
            if response.status_code == 404:
                return response
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                return None
            time.sleep(2 ** (attempt + 1))


def modified(response):
    return pd.Timestamp(parsedate_to_datetime(response.headers["Last-Modified"])).tz_convert("UTC")


def fetch_one(key):
    """Newest available run for one (date, cutoff); a request error stops the search (no silent fallback)."""
    day, hour = pd.Timestamp(key[0]), key[1]
    cutoff = cutoff_utc(day, hour)
    record = {"date": day.strftime("%Y-%m-%d"), "hour": hour, "cutoff_utc": cutoff.isoformat()}
    tried = []
    for init in candidate_runs(day, hour):
        url = run_url(init)
        head = request("HEAD", url)
        if head is None:
            return {**record, "status": "request_error", "tried": ";".join(tried + [f"{init.hour:02}z:error"])}
        if head.status_code == 404:
            tried.append(f"{init.hour:02}z:missing")
            continue
        archived = modified(head)
        if not available(init, archived, cutoff):
            tried.append(f"{init.hour:02}z:late:{archived.isoformat()}")
            continue
        response = request("GET", url)
        if response is None or response.status_code == 404:
            return {**record, "status": "request_error", "tried": ";".join(tried + [f"{init.hour:02}z:error"])}
        archived = modified(response)
        if not available(init, archived, cutoff):  # rewritten between HEAD and GET
            tried.append(f"{init.hour:02}z:late:{archived.isoformat()}")
            continue
        tried.append(f"{init.hour:02}z:ok:{archived.isoformat()}")
        out = {**record, "run_utc": init.isoformat(), "url": url, "archived_utc": archived.isoformat(),
               "sha256": hashlib.sha256(response.content).hexdigest(), "tried": ";".join(tried)}
        leads = leads_for(init)
        values = decode(response.content, leads)
        if not np.all(np.isfinite(values) & (values > -90) & (values < 150)):
            return {**out, "status": "invalid_values", "n_leads": leads}
        return {**out, "status": "ok", "n_leads": leads, "temps": [round(float(v), 4) for v in values]}
    return {**record, "status": "no_run", "tried": ";".join(tried)}


def read_records():
    if not RECORDS.exists():
        return pd.DataFrame()
    rows = []
    for line in RECORDS.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut short by an interrupted run is fetched again
            continue
    table = pd.DataFrame(rows)
    # The last line for a key wins (a request error is retried and later recorded again).
    return table.drop_duplicates(["date", "hour"], keep="last").reset_index(drop=True)


def fetch(start, end):
    RAW.mkdir(parents=True, exist_ok=True)
    done = read_records()
    finished = set()
    if len(done):
        finished = set(zip(done.loc[~done.status.eq("request_error"), "date"], done.hour))
    keys = [(d.strftime("%Y-%m-%d"), h) for d in pd.date_range(start, end) for h in HOURS]
    keys = [k for k in keys if k not in finished]
    began = time.time()
    with open(PROGRESS, "a") as log:
        log.write(f"{pd.Timestamp.now('UTC').isoformat()} start {len(keys)} keys ({start} to {end})\n")
        log.flush()
        statuses = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool, open(RECORDS, "a") as out:
            futures = [pool.submit(fetch_one, key) for key in keys]
            for count, future in enumerate(as_completed(futures), 1):
                record = future.result()
                out.write(json.dumps(record) + "\n")
                out.flush()
                statuses[record["status"]] = statuses.get(record["status"], 0) + 1
                if count % 100 == 0 or count == len(keys):
                    log.write(f"{pd.Timestamp.now('UTC').isoformat()} {count}/{len(keys)} "
                              f"{time.time() - began:.0f}s {statuses}\n")
                    log.flush()
        log.write(f"{pd.Timestamp.now('UTC').isoformat()} done {statuses}\n")
    print(statuses, flush=True)


def build():
    table = read_records().sort_values(["date", "hour"]).reset_index(drop=True)
    temps = np.full((len(table), MAX_LEADS), np.nan)
    rows = []
    for i, row in enumerate(table.to_dict("records")):
        if isinstance(row.get("temps"), list):
            temps[i, :len(row["temps"])] = row["temps"]
        if row["status"] != "ok":
            continue
        init, cutoff = pd.Timestamp(row["run_utc"]), pd.Timestamp(row["cutoff_utc"])
        if not available(init, pd.Timestamp(row["archived_utc"]), cutoff):
            raise ValueError(f"Run after cutoff: {row['date']} {row['hour']}")
        rows.append({"date": row["date"], "hour": row["hour"],
                     **features(pd.Timestamp(row["date"]), row["hour"], init, temps[i, :int(row["n_leads"])])})
    cycles = table.drop(columns="temps", errors="ignore")
    cycles["run_hour"] = pd.to_datetime(cycles.get("run_utc")).dt.hour
    cycles = pd.concat([cycles, pd.DataFrame(np.round(temps, 3), columns=[f"f{lead:02}" for lead in
                                                                          range(1, MAX_LEADS + 1)])], axis=1)
    gzip = {"method": "gzip", "mtime": 0}
    cycles.to_csv(CYCLES, index=False, compression=gzip)
    pd.DataFrame(rows).to_csv(FEATURES, index=False, compression=gzip)
    print(cycles.groupby("hour").status.value_counts().unstack(fill_value=0).to_string(), flush=True)
    print(pd.DataFrame(rows).groupby("hour").hrrr_fresh_age_h.value_counts().unstack(fill_value=0).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "build"))
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()
    fetch(args.start, args.end) if args.command == "fetch" else build()


if __name__ == "__main__":
    main()
