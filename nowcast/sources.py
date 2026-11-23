"""Read public NOAA objects as they were available at a forecast cutoff."""
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import local
from uuid import uuid4
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
import requests
from numcodecs.blosc import decompress
from pyproj import Proj
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
TZ = "America/New_York"
LAT, LON = 40.77945, -73.87397
THREAD = local()
HRRR = "https://hrrrzarr.s3.us-west-1.amazonaws.com/sfc"
SWITCH = pd.Timestamp("2025-04-07T15:10Z")


def get(url, **kwargs):
    if not hasattr(THREAD, "http"):
        THREAD.http = requests.Session()
        retry = Retry(total=2, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
        THREAD.http.mount("https://", HTTPAdapter(max_retries=retry))
    return THREAD.http.get(url, timeout=(8, 20), **kwargs)


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_name(path.name + uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def cached(path, build):
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    value = build()
    if value is not None:
        save(path, value)
    return value


def utc(value):
    return pd.Timestamp(value).tz_convert("UTC")


def goes_time(token):
    return pd.to_datetime(token[1:14], format="%Y%j%H%M%S", utc=True) + pd.Timedelta(
        milliseconds=int(token[14:15]) * 100)


def scan_list(hour, sat):
    prefix = f"ABI-L2-ACMC/{hour:%Y/%j/%H}/"
    bucket = f"https://noaa-goes{sat}.s3.amazonaws.com"
    def fetch():
        response = get(bucket, params={"list-type": 2, "prefix": prefix})
        response.raise_for_status()
        rows = []
        for obj in ET.fromstring(response.content).findall("{*}Contents"):
            key = obj.findtext("{*}Key")
            parts = Path(key).stem.split("_")
            if len(parts) < 5 or not parts[-1].startswith("c"):
                continue
            end, creation = map(goes_time, parts[-2:])
            available = max(creation, pd.Timestamp(obj.findtext("{*}LastModified")))
            rows.append({"url": bucket + "/" + key, "end": end.isoformat(),
                         "available": available.isoformat()})
        return rows
    if hour + pd.Timedelta(hours=2) > pd.Timestamp.now(tz="UTC"):
        return fetch()
    return cached(CACHE / "lists" / f"{sat}-{hour:%Y%m%d%H}.json", fetch)


def find_scan(at):
    at = utc(at)
    for hour in [at.floor("h"), at.floor("h") - pd.Timedelta(hours=1)]:
        for sat in ((19, 16) if hour == SWITCH.floor("h") else (19,) if hour >= SWITCH else (16,)):
            valid = [r for r in scan_list(hour, sat)
                     if at - pd.Timedelta(minutes=30) <= pd.Timestamp(r["end"]) <= pd.Timestamp(r["available"]) <= at
                     and (pd.Timestamp(r["end"]) >= SWITCH) == (sat == 19)]
            if valid:
                return max(valid, key=lambda r: r["end"])
    return None


@lru_cache(maxsize=12)
def region(lon0, height, major, minor, sweep, xs, ys):
    projection = Proj(proj="geos", lon_0=lon0, h=height, a=major, b=minor, sweep=sweep)
    x, y = np.frombuffer(xs, dtype="float64"), np.frombuffer(ys, dtype="float64")
    center = projection(LON, LAT)
    ix = np.argmin(abs(x - center[0]))
    iy = np.argmin(abs(y - center[1]))
    sy, sx = slice(iy-32, iy+33), slice(ix-32, ix+33)
    lon, lat = projection(*np.meshgrid(x[sx], y[sy]), inverse=True)
    east = (lon - LON) * 111.32 * np.cos(np.deg2rad(LAT))
    north = (lat - LAT) * 111.32
    distance = np.hypot(east, north)
    bearing = (np.rad2deg(np.arctan2(east, north)) + 360) % 360
    masks = [distance <= 10, distance <= 40]
    masks.extend((distance >= 10) & (distance <= 40) &
                 (abs((bearing - direction + 180) % 360 - 180) <= 45)
                 for direction in range(0, 360, 45))
    return sy, sx, masks


def cloud_summary(record):
    def decode():
        r = get(record["url"])
        r.raise_for_status()
        with h5py.File(BytesIO(r.content)) as f:
            p = f["goes_imager_projection"].attrs
            def number(name):
                return float(np.asarray(p[name]).item())
            def axis(name):
                d = f[name]
                return ((d[:].astype(float)*d.attrs["scale_factor"] + d.attrs["add_offset"])
                        * number("perspective_point_height")).astype("float64").tobytes()
            sy, sx, masks = region(number("longitude_of_projection_origin"),
                                   number("perspective_point_height"), number("semi_major_axis"),
                                   number("semi_minor_axis"), p["sweep_angle_axis"].decode(),
                                   axis("x"), axis("y"))
            cloud = f["BCM"][sy, sx]
            valid = (f["DQF"][sy, sx] == 0) & np.isin(cloud, [0, 1])
            fractions = [float(cloud[valid & m].mean()) if (valid & m).sum() >= m.sum()*0.5 else None
                         for m in masks]
        return {**record, "fraction": fractions}
    return cached(CACHE / "cloud" / (Path(record["url"]).stem + ".json"), decode)


def satellite(at):
    scans = []
    for minutes in (0, 30, 60):
        try:
            record = find_scan(at - pd.Timedelta(minutes=minutes))
            scans.append(cloud_summary(record) if record else None)
        except (requests.RequestException, OSError, ValueError):
            if minutes == 0:
                raise
            scans.append(None)
        if minutes == 0 and (not scans[0] or scans[0]["fraction"][0] is None):
            break
    return scans


def point(run, cloud=False):
    name = "TCDC" if cloud else "TMP"
    level = "entire_atmosphere" if cloud else "2m_above_ground"
    url = f"{HRRR}/{run:%Y%m%d}/{run:%Y%m%d_%H}z_fcst.zarr/{level}/{name}/{level}/{name}/0.4.10"
    def fetch():
        r = get(url)
        if r.status_code == 404:
            # Past misses are stable. A current model cycle may arrive on the next refresh.
            return {"missing": True} if run < pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=2) else None
        r.raise_for_status()
        size = r.content[3]
        if size not in (2, 4):
            raise ValueError("Unexpected HRRR element size")
        values = np.frombuffer(decompress(r.content), dtype=f"<f{size}").reshape(-1,150,150)[:,98,56].astype(float)
        if not cloud:
            values = (values - 273.15)*1.8+32
        limits = (0, 100) if cloud else (-90, 150)
        if not np.all((values >= limits[0]) & (values <= limits[1])):
            return {"missing": True}
        return {"run": run.isoformat(), "available": parsedate_to_datetime(r.headers["Last-Modified"]).isoformat(),
                "values": values.tolist(), "url": url}
    return cached(CACHE / "cycles" / f"{run:%Y%m%d%H}-{name}.json", fetch)


def eligible_cycle(row, at):
    if not row or row.get("missing"):
        return False
    run = pd.Timestamp(row["run"])
    end = (at.normalize() + pd.DateOffset(days=1)).tz_convert("UTC") - pd.Timedelta(hours=1)
    return run <= pd.Timestamp(row["available"]) <= at and run + pd.Timedelta(hours=len(row["values"])) >= end


def forecasts(at, include_fixed=True):
    stamp = utc(at).floor("h")
    latest = None
    for age in range(0, 13):
        run = stamp - pd.Timedelta(hours=age)
        # Every ordinary cycle has 18 leads, synoptic 6-hourly cycles up to 48.
        max_lead = 48 if run.hour % 6 == 0 else 18
        if run + pd.Timedelta(hours=max_lead) < at.normalize()+pd.DateOffset(days=1)-pd.Timedelta(hours=1):
            continue
        row = point(run)
        if eligible_cycle(row, at):
            latest = row
            break
    try:
        cloud = point(pd.Timestamp(latest["run"]), True) if latest else None
    except (requests.RequestException, OSError, ValueError):
        cloud = None
    fixed = None
    if include_fixed:
        fixed_run = pd.Timestamp(at.date(), tz="UTC") + pd.Timedelta(hours=10)
        if fixed_run <= at:
            candidate = point(fixed_run)
            if eligible_cycle(candidate, at):
                fixed = candidate
    return {"latest": latest, "cloud": cloud if eligible_cycle(cloud, at) else None, "fixed": fixed}


def external(at, include_fixed=True):
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {"scans": pool.submit(satellite, at), "forecast": pool.submit(forecasts, at, include_fixed)}
        out = {"scans": [], "forecast": {}, "errors": {}}
        for key, task in jobs.items():
            try:
                out[key] = task.result()
            except (requests.RequestException, OSError, ValueError, KeyError) as exc:
                out["errors"][key] = str(exc)
    return out

