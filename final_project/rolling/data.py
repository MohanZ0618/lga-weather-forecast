"""As-of inputs for the three fixed local issue times. Run with --external to backfill."""
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent / "nowcast"))
from prepare import DATA, STATION_IDS, lcd_local_time
from weather import daily_labels, hourly_rows
from features import coverage
import sources

OUT = ROOT / "results" / "rolling"
STORE = DATA / "rolling"
HOURS = (9, 12, 15)
DELAYS = (0, 15, 30)
TZ = "America/New_York"
SNAPSHOT_END = "2026-11-21"  # Last completed local day when this historical study was requested.
FIELDS = {"temp": "HourlyDryBulbTemperature", "dew": "HourlyDewPointTemperature",
          "rh": "HourlyRelativeHumidity", "wind": "HourlyWindSpeed", "dir": "HourlyWindDirection",
          "sky": "HourlySkyConditions", "vis": "HourlyVisibility"}
THERMO = [f"{s}_{k}" for s in STATION_IDS for k in ("spread", "east", "north")]
SKY = [f"{s}_{k}" for s in STATION_IDS for k in ("sky", "vis")]


def issue_index(days, hour):
    return (days + pd.Timedelta(hours=hour)).tz_localize(TZ)


def latest_cycle(at):
    # A HEAD request can exclude an unpublished tile before transferring its body.
    end = at.normalize() + pd.DateOffset(days=1) - pd.Timedelta(hours=1)
    for age in range(13):
        run = at.tz_convert("UTC") - pd.Timedelta(hours=age)
        if run + pd.Timedelta(hours=48 if run.hour % 6 == 0 else 18) < end:
            continue
        path = sources.CACHE / "cycles" / f"{run:%Y%m%d%H}-TMP.json"
        if not path.exists():
            level = "2m_above_ground/TMP"
            url = f"{sources.HRRR}/{run:%Y%m%d}/{run:%Y%m%d_%H}z_fcst.zarr/{level}/{level}/0.4.10"
            def metadata():
                # Tile metadata shares the cache across the three daily cutoffs.
                session = getattr(sources.THREAD, "head_client", None)
                if session is None:
                    session = requests.Session()
                    retry = sources.Retry(total=3,backoff_factor=.4,status_forcelist=(429,500,502,503,504))
                    session.mount("https://",sources.HTTPAdapter(max_retries=retry))
                    sources.THREAD.head_client = session
                response = session.head(url, timeout=(8, 20))
                if response.status_code == 404:
                    return {"missing": True}
                response.raise_for_status()
                return {"available": parsedate_to_datetime(response.headers["Last-Modified"]).isoformat()}
            meta = sources.cached(STORE / "metadata" / f"{run:%Y%m%d%H}.json", metadata)
            if meta.get("missing") or pd.Timestamp(meta["available"]) > at:
                continue
        cycle = sources.point(run)
        if sources.eligible_cycle(cycle, at):
            return cycle
    return None


def remote_day(day):
    path = STORE / "days" / f"{day:%Y-%m-%d}.json"
    previous = json.loads(path.read_text()) if path.exists() else []
    if len(previous) == 3 and all(not row["errors"] for row in previous):
        return previous
    rows = []
    for hour in HOURS:
        at = issue_index(pd.DatetimeIndex([day]), hour)[0]
        row = {"date": str(day.date()), "hour": hour, "errors": {}}
        try:
            cycle = latest_cycle(at)
            if cycle:
                run = pd.Timestamp(cycle["run"])
                leads = pd.date_range(run + pd.Timedelta(hours=1), periods=len(cycle["values"]), freq="h")
                future = (leads >= at) & (leads < at.normalize() + pd.DateOffset(days=1))
                temps = np.asarray(cycle["values"])
                if at in leads and future.any():
                    row.update(latest_high=float(temps[future].max()),
                               latest_warm=float(temps[future].max()-temps[leads == at][0]),
                               cycle_url=cycle["url"], cycle_available=cycle["available"], cycle_run=cycle["run"])
                    cloud = sources.point(run, cloud=True)
                    if sources.eligible_cycle(cloud, at) and len(cloud["values"]) == len(leads):
                        row.update(model_cloud=float(np.mean(np.asarray(cloud["values"])[future]))/100,
                                   cloud_url=cloud["url"], cloud_available=cloud["available"])
        except (requests.RequestException, ValueError, OSError, KeyError) as exc:
            row["errors"]["hrrr"] = str(exc)
        try:
            record = sources.find_scan(at)
            if record:
                scan = sources.cloud_summary(record)
                row.update(sat10=scan["fraction"][0], sat40=scan["fraction"][1],
                           goes_url=scan["url"], goes_end=scan["end"], goes_available=scan["available"])
        except (requests.RequestException, ValueError, OSError, KeyError) as exc:
            row["errors"]["goes"] = str(exc)
        rows.append(row)
    sources.save(path, rows)
    return rows


def external(workers, start, end):
    days = pd.date_range(start, end)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, rows in enumerate(pool.map(remote_day, days), 1):
            if number % 100 == 0:
                print(f"{number}/{len(days)} archive days through {rows[0]['date']}", flush=True)
    if (start,end) != ("2021-01-01",SNAPSHOT_END):
        return
    records = [r for d in days for r in json.loads((STORE / "days" / f"{d:%Y-%m-%d}.json").read_text())]
    out = pd.DataFrame(records)
    failed = int(out.errors.map(bool).sum())
    out["errors"] = out.errors.map(json.dumps)
    out.to_parquet(STORE / "external.parquet", index=False)
    print(f"{len(out)} issues, {failed} source errors", flush=True)
    if failed:
        raise RuntimeError("Re-run --external to retry the archived days with errors")


def clean(path):
    raw = pd.read_csv(path, usecols=["DATE", "REPORT_TYPE", *FIELDS.values()], dtype="string")
    keep = raw.loc[raw.REPORT_TYPE.isin(["FM-15", "FM-16"])].rename(columns={v:k for k,v in FIELDS.items()})
    data = pd.DataFrame(index=keep.index)
    for col, low, high in [("temp",-75,60), ("dew",-80,60), ("rh",0,100), ("dir",0,360),
                           ("wind",0,120), ("vis",0,200)]:
        values = pd.to_numeric(keep[col], errors="coerce").astype(float)
        data[col] = values.where(values.between(low, high))
    data["sky"] = keep.sky.map(coverage)
    data["spread"] = (data.temp-data.dew)*1.8
    data["temp"] = data.temp*1.8+32
    angle = np.deg2rad(data.dir)
    data["east"] = (-data.wind*np.sin(angle)).mask(data.wind.eq(0),0)
    data["north"] = (-data.wind*np.cos(angle)).mask(data.wind.eq(0),0)
    data.index = pd.DatetimeIndex(lcd_local_time(keep.DATE))
    data = data.sort_index(kind="stable")
    counts = dict(raw=len(raw), reports=len(keep), duplicates=int(data.index.duplicated().sum()))
    data = data.loc[~data.index.duplicated(keep="last")]
    return data, {**counts, "clean":len(data), **data.isna().sum().add_prefix("missing_").to_dict()}


def station_inputs(obs, days, hour, delay):
    at = issue_index(days, hour)
    stamp = at - pd.Timedelta(minutes=delay)
    one = pd.Timedelta(hours=1)
    def lookup(times):
        rows = obs.assign(observed=obs.index).reindex(times, method="ffill", tolerance=one)
        rows.index = days
        return rows
    current = lookup(stamp)
    valid_temp = obs.temp.dropna()
    def fresh_temp(times):
        positions = valid_temp.index.get_indexer(times,method="ffill",tolerance=one)
        ok = positions>=0
        out = pd.Series(np.nan,index=days)
        ts = pd.Series(pd.NaT,index=days,dtype=times.dtype)
        out.loc[ok] = valid_temp.iloc[positions[ok]].to_numpy()
        ts.loc[ok] = valid_temp.index[positions[ok]]
        return out,ts
    latest_temp,latest_time = fresh_temp(stamp)
    cols = ["temp","rh","wind","spread","east","north","sky","vis"]
    values = current[cols].copy()
    for lag in (1, 3):
        older = lookup(stamp-lag*one)
        change = current.temp-older.temp
        earlier,earlier_time = fresh_temp(stamp-lag*one)
        elapsed = (latest_time-earlier_time)/one
        values[f"trend_{lag}h"] = (latest_temp-earlier)/elapsed.where(elapsed.gt(0))
        if lag == 3:
            values["change_3h"] = change
    hourly = hourly_rows(obs).temp
    maxima = hourly.groupby(hourly.index.tz_localize(None).normalize()).max()
    values["previous_max"] = maxima.reindex(days-pd.Timedelta(days=1)).to_numpy()
    # Only closed hours as of the receipt-adjusted cutoff can establish a hard floor.
    stop = stamp.floor("h")
    day_start = at.normalize()
    starts = hourly.index.searchsorted(day_start)
    ends = hourly.index.searchsorted(stop)
    values["max_so_far"] = [hourly.iloc[i:j].max() for i,j in zip(starts,ends)]
    values["observed"] = current.observed
    return values


def stations():
    manifest = pd.read_csv(DATA / "station_sources.csv").set_index(["station","source_year"])
    tables, audit = {}, []
    for station, code in STATION_IDS.items():
        frames = []
        for path in sorted((DATA / "raw").glob(f"LCD_{code}_*.csv")):
            year = int(path.stem[-4:])
            full, counts = clean(path)
            src = manifest.loc[(station,year)]
            digest = sha256(path.read_bytes()).hexdigest()
            if digest != src.sha256:
                raise ValueError(f"Station checksum mismatch: {path.name}")
            audit.append(dict(station=station, year=year, url=src.url, sha256=digest, **counts))
            tail = STORE/"raw_tail"/path.name
            if year==2026 and tail.exists() and sha256(tail.read_bytes()).hexdigest()!=digest:
                updated,updated_counts = clean(tail)
                audit.append(dict(station=station,year=year,url=src.url,sha256=sha256(tail.read_bytes()).hexdigest(),
                                  snapshot="tail from 2026-11-07",**updated_counts))
                full = pd.concat([full.loc[full.index<"2026-11-07"],updated.loc["2026-11-07":]])
            frames.append(full)
        tables[station] = pd.concat(frames).sort_index(kind="stable")
    labels = daily_labels(hourly_rows(tables["LGA"])).loc["2018-01-01":SNAPSHOT_END]
    result = []
    for hour in HOURS:
        for delay in DELAYS:
            xs = [station_inputs(t,labels.index,hour,delay).add_prefix(s+"_") for s,t in tables.items()]
            result.append(pd.concat([labels,*xs],axis=1).assign(hour=hour, delay=delay).reset_index())
    STORE.mkdir(parents=True,exist_ok=True)
    ground = pd.concat(result,ignore_index=True)
    ground.to_parquet(STORE / "stations.parquet", index=False)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(OUT / "station_quality.csv",index=False)
    selected = [c for c in ground if any(c.startswith(s+"_") for s in STATION_IDS)]
    ground[selected].notna().groupby([ground.date.dt.year.rename("year"),ground.hour,ground.delay]).mean().to_csv(
        OUT / "station_feature_coverage.csv")


def autumn_fixed():
    from hrrr import download, forecast_features
    rows = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for cycle in pool.map(download, pd.date_range("2026-09-01",SNAPSHOT_END)):
            if cycle["status"] == "request_error":
                raise RuntimeError(f"Retry autumn fixed cycle {cycle['date']}")
            if cycle["status"] == "ok":
                values = np.array([cycle[f"f{i:02}"] for i in range(1,19)])
                for hour in HOURS:
                    high, warm = forecast_features(cycle["date"],hour,pd.Timestamp(cycle["archived_utc"]),values)
                    rows.append(dict(date=cycle["date"],hour=hour,hrrr_remaining_max=high,hrrr_warming=warm,
                                     url=cycle["url"],sha256=cycle["sha256"],available=cycle["archived_utc"]))
    pd.DataFrame(rows).to_csv(STORE/"autumn_fixed.csv",index=False)


def refresh_tail():
    manifest = pd.read_csv(DATA/"station_sources.csv").query("source_year==2026")
    checks = []
    for url in manifest.url:
        path = STORE/"raw_tail"/Path(url).name
        if not path.exists():
            response = sources.get(url)
            response.raise_for_status()
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(response.content)
            clean(tmp)
            tmp.replace(path)
        updated,_ = clean(path)
        checks.append(dict(url=url,sha256=sha256(path.read_bytes()).hexdigest(),
                           first=str(updated.index.min()),last=str(updated.index.max())))
    pd.DataFrame(checks).to_csv(OUT/"lcd_tail_sources.csv",index=False)


def retry_errors(workers):
    failed = [pd.Timestamp(path.stem) for path in (STORE/"days").glob("*.json")
              if any(row["errors"] for row in json.loads(path.read_text()))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        remaining = sum(bool(r["errors"]) for day in pool.map(remote_day,failed) for r in day)
    print(f"Retried {len(failed)} days; {remaining} issue errors remain.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external",action="store_true")
    parser.add_argument("--autumn-fixed",action="store_true")
    parser.add_argument("--refresh-tail",action="store_true")
    parser.add_argument("--retry-errors",action="store_true")
    parser.add_argument("--workers",type=int,default=20)
    parser.add_argument("--start",default="2021-01-01")
    parser.add_argument("--end",default=SNAPSHOT_END)
    args = parser.parse_args()
    if args.external:
        external(args.workers,args.start,args.end)
    elif args.autumn_fixed:
        autumn_fixed()
    elif args.refresh_tail:
        refresh_tail()
    elif args.retry_errors:
        retry_errors(args.workers)
    else:
        stations()
