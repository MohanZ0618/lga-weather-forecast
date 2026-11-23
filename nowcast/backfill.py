"""Fetch a deterministic clock- and season-stratified sample of forecast snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pandas as pd
from sources import CACHE, TZ, external, save


def schedule(pilot=False):
    pilot_days = set()
    for start in ["2024-01-12", "2024-04-12", "2024-07-12", "2024-10-12"]:
        pilot_days.update(pd.date_range(start, periods=7))
    dates = sorted(pilot_days) if pilot else pd.date_range("2021-01-01", "2026-11-06")
    for day in dates:
        digest = hashlib.sha256(str(day.date()).encode()).digest()
        if not pilot and day not in pilot_days and day < pd.Timestamp("2026-09-01") and day.toordinal() % 3:
            continue
        slots = [digest[0] % 16, 16 + digest[1] % 20, 36 + digest[2] % 21]
        if day not in pilot_days and day < pd.Timestamp("2026-09-01"):
            slots = [slots[day.toordinal() // 3 % 3]]
        for slot in slots:
            yield (day + pd.Timedelta(hours=6, minutes=15*slot)).tz_localize(TZ)


def capture(at):
    path = CACHE / "snapshots" / f"{at:%Y%m%d-%H%M}.json"
    if path.exists() and not json.loads(path.read_text())["errors"]:
        return "cached"
    result = external(at)
    save(path, {"cutoff": at.isoformat(), **result})
    return "error" if result["errors"] else "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    counts = {}
    stamps = list(schedule(args.pilot))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, status in enumerate(pool.map(capture, stamps), 1):
            counts[status] = counts.get(status, 0)+1
            if i % 25 == 0 or i == len(stamps):
                print(i, len(stamps), counts, flush=True)

