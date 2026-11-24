"""Extract KLGA from public NOAA NBH cards, with issue-time availability and resumable cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from threading import Lock, local
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "bracket_nbm.parquet"
CACHE = ROOT / "build" / "nbm_cards"
NY = ZoneInfo("America/New_York")
HOST = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
THREAD = local()
OFFSETS = {}
LOCK = Lock()
HEADER = re.compile(rb"(?m)^ *KLGA +NBM +V(\S+) +NBH GUIDANCE +(\d+/\d+/\d{4}) +(\d{4}) UTC")


def client():
    if not hasattr(THREAD, "client"):
        THREAD.client = requests.Session()
        THREAD.client.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=.3, status_forcelist=(429, 500, 502, 503, 504))))
    return THREAD.client


def read_card(url, at):
    # A small known window works for v4.2. Discover historical layout changes once per size.
    def fetch(start):
        response = client().get(url, headers={"Range": f"bytes={start}-{start+8191}"}, timeout=(10, 35))
        if response.status_code == 416 and start:
            return fetch(0)
        if response.status_code == 416:
            return None
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response

    start = 16993800
    response = fetch(start)
    if response is None:
        return None
    published = parsedate_to_datetime(response.headers["Last-Modified"])
    if published > at:
        return None
    size = int(response.headers.get("Content-Range", "/0").split("/")[-1])
    content = response.content
    if not HEADER.search(content):
        with LOCK:
            if size not in OFFSETS:
                full = client().get(url, timeout=(10, 65))
                full.raise_for_status()
                found = HEADER.search(full.content)
                if not found:
                    return None
                OFFSETS[size] = max(0, found.start()-2)
        response = fetch(OFFSETS[size])
        if response is None:
            return None
        content = response.content
    published = parsedate_to_datetime(response.headers["Last-Modified"])
    if published > at:
        return None
    match = HEADER.search(content)
    if not match:
        # Station position can change despite an unchanged file size.
        full = client().get(url, timeout=(10, 65))
        full.raise_for_status()
        published = parsedate_to_datetime(full.headers["Last-Modified"])
        if published > at:
            return None
        content = full.content
        match = HEADER.search(content)
        if not match:
            return None
        with LOCK:
            OFFSETS[size] = max(0, match.start()-2)
    block = content[match.start():].splitlines()
    lines = [block[0]]
    for line in block[1:]:
        if b"NBH GUIDANCE" in line or not line.strip():
            break
        lines.append(line)
    return b"\n".join(lines), published


def parse_card(card, run, at):
    header = HEADER.search(card)
    if header is None:
        raise ValueError("missing KLGA NBH header")
    stamp = datetime.strptime((header[2]+b" "+header[3]).decode(), "%m/%d/%Y %H%M").replace(tzinfo=timezone.utc)
    if stamp != run:
        raise ValueError("NBH header cycle differs from requested cycle")
    fields = {}
    for line in card.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0] in (b"UTC", b"TMP", b"TSD", b"SKY"):
            # NBH uses fixed-width 3-character cells, including adjacent triple-digit values.
            fields[parts[0]] = [line[i:i+3].strip() for i in range(5, len(line.rstrip()), 3)]
    if not {b"UTC",b"TMP"}.issubset(fields):
        return None
    n = len(fields[b"UTC"])
    valid = [run+timedelta(hours=i+1) for i in range(n)]
    if any(int(cell) != d.hour for cell,d in zip(fields[b"UTC"],valid)):
        raise ValueError("unexpected hourly lead sequence")
    remaining = [i for i,d in enumerate(valid) if d>=at and d.astimezone(NY).date()==at.date()]
    if not remaining or valid[remaining[-1]].astimezone(NY).hour != 23:
        return None
    def values(name,low,high):
        parts = fields.get(name, [])
        nums = np.array([float(v) if v.lstrip(b"-").isdigit() else np.nan for v in parts])
        if len(nums)!=n:
            return np.full(n,np.nan)
        return np.where((nums>=low)&(nums<=high),nums,np.nan)
    temps = values(b"TMP",-100,140)[remaining]
    if not np.isfinite(temps).all():
        return None
    peak = remaining[int(np.argmax(temps))]
    sky = values(b"SKY",0,100)[remaining]
    return dict(nbm_high=float(max(temps)),nbm_peak_hours=(valid[peak]-at).total_seconds()/3600,
                nbm_spread=values(b"TSD",0,50)[peak],
                nbm_cloud=float(np.mean(sky)) if np.isfinite(sky).all() else np.nan,
                version=header[1].decode())


def extract(key):
    day,hour = key
    path = CACHE/f"{day}-{hour:02}.json"
    if path.exists():
        return json.loads(path.read_text())
    at = datetime.fromisoformat(day).replace(hour=hour,tzinfo=NY)
    row = dict(date=day,hour=hour,status="unavailable")
    for age in (1,2,3):
        run = at.astimezone(timezone.utc)-timedelta(hours=age)
        url = f"{HOST}/blend.{run:%Y%m%d}/{run:%H}/text/blend_nbhtx.t{run:%H}z"
        card = read_card(url,at)
        if card:
            record = parse_card(card[0],run,at)
            if record:
                row.update(record,status="ok",url=url,published=card[1].isoformat(),run=run.isoformat(),
                           card_sha256=sha256(card[0]).hexdigest())
                break
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(path)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start",default="2021-01-01")
    parser.add_argument("--end",default="2026-11-21")
    parser.add_argument("--workers",type=int,default=14)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--cached-development",action="store_true",
                        help="Assemble only cached pre-freeze inputs for offline model selection")
    args = parser.parse_args()
    if args.cached_development:
        args.start,args.end = "2021-01-01","2025-08-31"
    CACHE.mkdir(parents=True,exist_ok=True)
    dates = ["2021-01-15","2023-01-15","2024-06-15","2025-07-15","2026-07-15"] if args.pilot else [
        d.date().isoformat() for d in pd.date_range(args.start,args.end)]
    keys = [(day,h) for day in dates for h in (9,12,15)]
    rows = []
    def read(key):
        if args.cached_development:
            return json.loads((CACHE/f"{key[0]}-{key[1]:02}.json").read_text())
        return extract(key)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(read,keys):
            rows.append(row)
            if len(rows)%500==0:
                print(f"{len(rows)}/{len(keys)}",flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["hour","status"]).size(),flush=True)
    if args.pilot:
        print(frame.to_string(index=False))
    else:
        destination = CACHE.parent/"nbm_development.parquet" if args.cached_development else TABLE
        frame.to_parquet(destination.with_suffix(".tmp"),index=False)
        destination.with_suffix(".tmp").replace(destination)


if __name__=="__main__":
    main()
