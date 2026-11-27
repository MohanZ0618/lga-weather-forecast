"""Long-record station MOS for KLGA from the Iowa Environmental Mesonet MOS archive: GFS (MAV), NAM (MET), MEX.

LAMP, NBM and the HRRR cache start between 2018 and 2021, so a model that needs them can only train on a
few years. The older MOS products are the same kind of station guidance and reach back further, which lets
"train on more years" be tested without imputing a missing source.

`python final_project/mos.py fetch` downloads one CSV per model and calendar year 2008-2026 (public IEM
endpoint, sequential, 3 s apart, retries with backoff) into data/raw/iem_mos/<MODEL>/ (gitignored) and
records URL, rows, runs and SHA-256 in data/mos_sources.csv. Files already cached (GFS 2018-2026 from
guidance.py) are not requested again.
`python final_project/mos.py build` writes data/mos_features.csv.gz: for every date and cutoff, the newest
run of each product that had been issued by the cutoff, and what it says about the rest of the local day.

Availability. The archive has no receipt times, so a run counts only from an assumed release time after its
initialization: GFS and NAM MOS 5 h, MEX 6 h. The same rule as guidance.newest_run, with these delays.

Features per product P in {gfs, nam} (MEX is 12-hourly, so only its daytime max is used):
  P_max            n_x at the 00 UTC projection that ends the local day: the daytime (7 am to 7 pm LST)
                   maximum, from the newest released run that still forecasts it (12 and 18 UTC runs no
                   longer do, so the 06 UTC run usually serves the afternoon)
  P_remaining_max  max of the 3-hourly tmp projections from the cutoff to local midnight (newest released
                   run with any such projection)
  P_cloud, P_u, P_v  mean sky fraction and wind components from the cutoff to 18:00 local (same run)
  P_now            tmp interpolated linearly to the cutoff from the newest released run whose projections
                   bracket it, to measure today's MOS error against the latest observation
  mex_max          n_x at the 00 UTC projection ending the local day, newest released MEX run

Coverage (data/mos_sources.csv): GFS and NAM every year 2008-2026 (NAM has only the 00 and 12 UTC runs);
the archive's MEX starts 2020-07-11, so MEX cannot serve a long training window.
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

from prepare import DATA, HOURS, local_cutoff

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATION = "KLGA"
MODELS = ("GFS", "NAM", "MEX")
YEARS = range(2008, 2027)
END = "2026-09-01T00:00Z"
LAST_DAY = "2026-08-31"
DELAY = {"GFS": pd.Timedelta(hours=5), "NAM": pd.Timedelta(hours=5), "MEX": pd.Timedelta(hours=6)}
RAW = DATA / "raw" / "iem_mos"
SOURCES = DATA / "mos_sources.csv"
FEATURES = DATA / "mos_features.csv.gz"
COVERAGE = DATA.parent / "results" / "mos_history" / "feature_coverage.csv"  # dates with each feature, by year
CLOUD = {"CL": 0.0, "FW": 0.19, "SC": 0.44, "BK": 0.75, "OV": 1.0}  # same midpoints as guidance.py


def url_for(model, year):
    last = int(END[:4])
    sts, ets = f"{year}-01-01T00:00Z", (f"{year + 1}-01-01T00:00Z" if year < last else END)
    return f"{URL}?{urllib.parse.urlencode(dict(station=STATION, model=model, sts=sts, ets=ets, format='csv'))}"


def fetch(models=MODELS, years=YEARS):
    records = []
    for model in models:
        for year in years:
            url, path = url_for(model, year), RAW / model / f"{STATION}-{year}.csv"
            if not path.exists():
                body = None
                for attempt in range(5):
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": "cs4260-course-project"})
                        body = urllib.request.urlopen(request, timeout=300).read()
                        break
                    except OSError as error:
                        print(f"{model} {year} attempt {attempt + 1} failed: {error}", flush=True)
                        time.sleep(15 * 2 ** attempt)
                if body is None:
                    raise RuntimeError(f"IEM request failed: {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_suffix(".part").write_bytes(body)
                path.with_suffix(".part").replace(path)
                time.sleep(3)
            body = path.read_bytes()
            rows = [row for row in list(csv.reader(io.StringIO(body.decode())))[1:] if row]
            runs = sorted({row[0] for row in rows})
            records.append({"model": model, "year": year, "url": url, "rows": len(rows), "runs": len(runs),
                            "first_run": runs[0] if runs else "", "last_run": runs[-1] if runs else "",
                            "sha256": hashlib.sha256(body).hexdigest()})
            print(model, year, len(rows), len(runs), flush=True)
    pd.DataFrame(records).to_csv(SOURCES, index=False)


def load(model):
    """{run initialization: dict of numpy arrays} for one product; ftime as UTC nanoseconds."""
    frames = [pd.read_csv(path, low_memory=False) for path in sorted((RAW / model).glob(f"{STATION}-*.csv"))]
    table = pd.concat([f for f in frames if len(f)], ignore_index=True)
    for column in ("runtime", "ftime"):
        table[column] = pd.to_datetime(table[column]).dt.tz_localize("UTC")
    table = table.drop_duplicates(["runtime", "ftime"], keep="last").sort_values(["runtime", "ftime"])
    blank = pd.Series(np.nan, index=table.index)  # MEX has no wind direction column
    numeric = {c: pd.to_numeric(table.get(c, blank), errors="coerce").to_numpy(float)
               for c in ("n_x", "tmp", "wdr", "wsp")}
    numeric["cld"] = table["cld"].astype("string").str.strip().map(CLOUD).astype(float).to_numpy()
    numeric["ftime"] = table["ftime"].to_numpy("datetime64[ns]")
    runs, starts = {}, table["runtime"].to_numpy()
    edges = np.flatnonzero(np.r_[True, table["runtime"].to_numpy()[1:] != table["runtime"].to_numpy()[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        runs[pd.Timestamp(starts[a])] = {k: v[a:b] for k, v in numeric.items()}
    return runs


def released(starts, cutoff, delay, oldest):
    """Run initializations released (initialization + delay <= cutoff), newest first, not older than oldest."""
    position = starts.searchsorted(cutoff - delay, side="right") - 1
    while position >= 0 and starts[position] >= oldest:
        yield starts[position]
        position -= 1


def _ns(stamp):
    return np.datetime64(pd.Timestamp(stamp).tz_convert("UTC").tz_localize(None), "ns")


def daytime_max(runs, starts, model, cutoff, day_end):
    """n_x at the 00 UTC projection that ends the local day, from the newest released run that has it."""
    target = _ns(day_end)
    for run in released(starts, cutoff, DELAY[model], cutoff - pd.Timedelta(hours=36)):
        rows = runs[run]
        hit = (rows["ftime"] == target) & np.isfinite(rows["n_x"])
        if hit.any():
            return {"max_run": run, "max": float(rows["n_x"][hit][0])}
    return {}


def _mean(values):
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def rest_of_day(runs, starts, model, cutoff, midnight, evening):
    """Projections from the cutoff to local midnight, from the newest released run that has any.

    The temperature at the cutoff needs a run whose projections bracket it (a 12 UTC run starts at 18 UTC),
    so `now` may come from an older run than the rest; both run times are recorded.
    """
    now, end, late = _ns(cutoff), _ns(midnight), _ns(evening)
    out = {}
    for run in released(starts, cutoff, DELAY[model], cutoff - pd.Timedelta(hours=24)):
        rows = runs[run]
        ftime, temp = rows["ftime"], rows["tmp"]
        valid = np.isfinite(temp)
        rest = valid & (ftime >= now) & (ftime < end)
        if "rest_run" not in out and rest.any():
            day = (ftime >= now) & (ftime <= late)
            direction, speed = np.deg2rad(rows["wdr"][day]), rows["wsp"][day]
            out.update({"rest_run": run, "remaining_max": float(temp[rest].max()), "cloud": _mean(rows["cld"][day]),
                        # Wind blows from `direction`: u > 0 is a west wind, v > 0 a south wind.
                        "u": _mean(-speed * np.sin(direction)), "v": _mean(-speed * np.cos(direction))})
        before, after = valid & (ftime <= now), valid & (ftime >= now)
        if "now_run" not in out and before.any() and after.any():
            i, j = np.flatnonzero(before)[-1], np.flatnonzero(after)[0]
            span = (ftime[j] - ftime[i]) / np.timedelta64(1, "h")
            if span <= 6:
                weight = 0.0 if span == 0 else ((now - ftime[i]) / np.timedelta64(1, "h")) / span
                out.update({"now_run": run, "now": float(temp[i] + weight * (temp[j] - temp[i]))})
        if "rest_run" in out and "now_run" in out:
            break
    return out


def day_features(products, day, hour):
    """All MOS features for one date and cutoff hour. products: {model: (runs, starts)}."""
    day = pd.Timestamp(day)
    cutoff = local_cutoff(day, hour).tz_convert("UTC")
    midnight = local_cutoff(day + pd.Timedelta(days=1), 0).tz_convert("UTC")
    evening = local_cutoff(day, 18).tz_convert("UTC")
    day_end = (day + pd.Timedelta(days=1)).tz_localize("UTC")
    row = {"date": day, "hour": hour}
    for model, (runs, starts) in products.items():
        prefix = model.lower()
        values = daytime_max(runs, starts, model, cutoff, day_end)
        if model != "MEX":
            values.update(rest_of_day(runs, starts, model, cutoff, midnight, evening))
        row.update({f"{prefix}_{k}": v for k, v in values.items()})
        for stamp in ("max_run", "rest_run", "now_run"):
            if f"{prefix}_{stamp}" in row:
                row[f"{prefix}_{stamp}"] = row[f"{prefix}_{stamp}"].tz_localize(None)
    return row


def build(out=FEATURES, first_day="2008-01-02"):
    products = {}
    for model in MODELS:
        runs = load(model)
        products[model] = (runs, pd.DatetimeIndex(sorted(runs)))
    records = [day_features(products, day, hour) for day in pd.date_range(first_day, LAST_DAY) for hour in HOURS]
    table = pd.DataFrame(records)
    order = ["date", "hour"] + [c for m in ("gfs", "nam", "mex") for c in table if c.startswith(m + "_")]
    table = table[order]
    for column in [c for c in table if c.endswith("_run")]:  # initialization times, UTC
        table[column] = pd.to_datetime(table[column]).dt.strftime("%Y-%m-%d %H:%M")
    table.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.4g")
    coverage = table.groupby([table.date.dt.year, "hour"])[["gfs_max", "gfs_remaining_max", "nam_max",
                                                             "nam_remaining_max", "mex_max"]].count()
    coverage = coverage.unstack("hour")
    coverage.columns = [f"{column}_{hour:02}" for column, hour in coverage.columns]
    COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    coverage.assign(days=table.groupby(table.date.dt.year).date.nunique()).rename_axis("year").to_csv(COVERAGE)
    print(coverage.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fetch", "build"])
    args = parser.parse_args()
    fetch() if args.step == "fetch" else build()


if __name__ == "__main__":
    main()
