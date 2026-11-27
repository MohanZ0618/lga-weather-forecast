"""Observed weather at LaGuardia and nearby ASOS sites from the Iowa Environmental Mesonet METAR archive.

fetch  One request per station and calendar year (routine and special METARs), sequential, at least
       3 s apart (6 s after IEM began answering 429 Too Many Requests), up to 4 attempts with backoff.
       Years 2020-2026 for LGA, JFK, EWR, TEB and NYC; 2021-2026 for HPN, ISP, FRG, CDW and SMQ to keep the
       total under 80 requests (training starts 2021-01-01 and no cutoff looks back past that day).
       Raw CSVs are cached in data/raw/iem_asos/ (gitignored); the URL, row count, first and last valid
       time and sha256 of each file go to data/asos_sources.csv. A rerun with the cache makes no requests.
build  data/asos_features.csv.gz, one row per date, cutoff hour (09, 12, 15 New York time) and delivery
       allowance. Timing rule: a report counts only if its valid time is strictly before
       cutoff - allowance, and a value is used only if its report is at most 90 minutes older than the
       cutoff. The primary allowance is 10 minutes. Because routine METARs at these sites are taken at
       :51 to :56, a 10-minute allowance drops the report from the hour just before the cutoff, so the
       newest usable routine report is about 64 to 69 minutes old. A 5-minute allowance (the :51
       report counts, the timing the paper's own LCD station features use) is built as a sensitivity.

Features, each taken from the newest usable report that carries the field:
  LGA  u and v wind (kt, meteorological convention: wind FROM 270 deg gives u > 0; calm or variable
       direction gives 0), gust (kt; the sustained speed when no gust is reported), altimeter (hPa) and
       its change over 3 h (value at the cutoff minus value at cutoff - 3 h, same timing rule),
       dew point depression (F), maximum sky cover fraction over the reported layers
       (CLR/FEW/SCT/BKN/OVC = 0/0.19/0.44/0.75/1, VV = 1), lowest ceiling (lowest BKN, OVC or VV base,
       kft, capped at 12, no ceiling = 12), precipitation in the 3 h before cutoff - allowance (sum of
       the hourly p01i in routine reports; trace = 0.0001 in).
  Nearby  temperature minus LGA's (both from the newest usable report) for the sites outside the paper's
       four (NYC, HPN, ISP, FRG, CDW, SMQ); JFK, EWR and TEB temperatures are already paper features.
       Maximum sky cover fraction for every non-LGA site.
"""

import argparse
import hashlib
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import HOURS
from prepare import DATA, local_cutoff

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
RAW = DATA / "raw" / "iem_asos"
SOURCES = DATA / "asos_sources.csv"
FEATURES = DATA / "asos_features.csv.gz"
PAPER_STATIONS = ("LGA", "JFK", "EWR", "TEB")
NEW_STATIONS = ("NYC", "HPN", "ISP", "FRG", "CDW", "SMQ")
STATIONS = PAPER_STATIONS + NEW_STATIONS
YEARS = range(2020, 2027)
LATE_STATIONS = ("HPN", "ISP", "FRG", "CDW", "SMQ")
LATE_YEARS = range(2021, 2027)
PAUSE_S = 6
LAST_DAY = (2026, 9, 1)  # exclusive end of the 2026 request; the test year ends 2026-08-31
FIELDS = ["tmpf", "dwpf", "drct", "sknt", "gust", "alti", "mslp", "skyc1", "skyc2", "skyc3",
          "skyl1", "skyl2", "skyl3", "p01i", "vsby", "wxcodes"]
SKY = {"CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0, "FEW": 0.19, "SCT": 0.44, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}
CEILING = ("BKN", "OVC", "VV")
CEILING_CAP_KFT = 12.0
ALLOWANCES = (10, 5)  # minutes; 10 is the primary rule
STALE = pd.Timedelta(minutes=90)
INHG_TO_HPA = 33.8639
LGA_FEATURES = ["LGA_u", "LGA_v", "LGA_gust", "LGA_alti", "LGA_alti_change_3h", "LGA_dewpoint_depression",
                "LGA_sky", "LGA_ceiling_kft", "LGA_precip_3h"]
NEARBY_FEATURES = [f"{s}_temp_vs_lga" for s in NEW_STATIONS] + [f"{s}_sky" for s in STATIONS[1:]]


def request_url(station, year):
    end = LAST_DAY if year == LAST_DAY[0] else (year + 1, 1, 1)
    query = [("station", station)] + [("data", f) for f in FIELDS] + [
        ("year1", year), ("month1", 1), ("day1", 1), ("year2", end[0]), ("month2", end[1]), ("day2", end[2]),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("missing", "empty"),
        ("report_type", 3), ("report_type", 4)]
    return f"{URL}?{urllib.parse.urlencode(query)}"


def station_years(station):
    return LATE_YEARS if station in LATE_STATIONS else YEARS


def fetch(stations=STATIONS):
    RAW.mkdir(parents=True, exist_ok=True)
    records, requests = [], 0
    for station in stations:
        for year in station_years(station):
            url = request_url(station, year)
            path = RAW / f"{station}_{year}.csv"
            if not path.exists():
                body = None
                for attempt in range(4):
                    requests += 1
                    try:
                        body = urllib.request.urlopen(url, timeout=300).read()
                        if not body.startswith(b"station,valid"):
                            raise OSError(f"unexpected response {body[:80]!r}")
                        break
                    except OSError as error:
                        body = None
                        print(f"{station} {year} attempt {attempt + 1} failed: {error}", flush=True)
                        time.sleep(15 * (attempt + 1))
                if body is None:
                    raise RuntimeError(f"IEM request failed: {url}")
                path.with_suffix(".part").write_bytes(body)
                path.with_suffix(".part").replace(path)
                time.sleep(PAUSE_S)
            body = path.read_bytes()
            frame = pd.read_csv(io.BytesIO(body), usecols=["valid"])
            records.append({"station": station, "year": year, "url": url, "rows": len(frame),
                            "first_valid_utc": frame.valid.min(), "last_valid_utc": frame.valid.max(),
                            "sha256": hashlib.sha256(body).hexdigest()})
            print(station, year, len(frame), flush=True)
    pd.DataFrame(records).to_csv(SOURCES, index=False)
    print(f"{requests} HTTP requests", flush=True)


# Report cleaning ----------------------------------------------------------------------------------

def wind_components(direction, speed):
    """Meteorological convention: direction is where the wind blows FROM, so FROM 270 (west) gives u > 0."""
    angle = np.deg2rad(np.asarray(direction, dtype=float))
    speed = np.asarray(speed, dtype=float)
    calm = (speed == 0) | (np.isnan(angle) & np.isfinite(speed))  # calm, or variable (VRB) with no mean vector
    u = np.where(calm, 0.0, -speed * np.sin(angle))
    v = np.where(calm, 0.0, -speed * np.cos(angle))
    return u, v


def clean(raw):
    """Typed, range-checked reports indexed by UTC valid time, one row per valid time (last kept)."""
    frame = raw.copy()
    frame.index = pd.to_datetime(frame.pop("valid")).dt.tz_localize("UTC")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    numeric = ["tmpf", "dwpf", "drct", "sknt", "gust", "alti", "skyl1", "skyl2", "skyl3", "p01i"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["tmpf"] = frame.tmpf.where(frame.tmpf.between(-40, 120))
    frame["dwpf"] = frame.dwpf.where(frame.dwpf.between(-60, 90) & (frame.dwpf <= frame.tmpf + 0.5))
    frame["sknt"] = frame.sknt.where(frame.sknt.between(0, 100))
    frame["drct"] = frame.drct.where(frame.drct.between(0, 360))
    frame["gust"] = frame.gust.where(frame.gust.between(0, 120))
    frame["alti"] = frame.alti.where(frame.alti.between(28.0, 31.5)) * INHG_TO_HPA
    frame["p01i"] = frame.p01i.where(frame.p01i.between(0, 10))
    frame["u"], frame["v"] = wind_components(frame.drct, frame.sknt)
    frame.loc[frame.sknt.isna(), ["u", "v"]] = np.nan
    frame["gust_or_speed"] = frame.gust.fillna(frame.sknt)
    frame["dewpoint_depression"] = frame.tmpf - frame.dwpf
    covers = np.column_stack([frame[f"skyc{i}"].astype("string").str.strip().map(SKY).astype(float)
                              for i in (1, 2, 3)])
    heights = frame[["skyl1", "skyl2", "skyl3"]].to_numpy(float) / 1000
    ceiling = np.column_stack([frame[f"skyc{i}"].astype("string").str.strip().isin(CEILING).fillna(False)
                               for i in (1, 2, 3)])
    reported = np.isfinite(covers[:, 0])
    frame["sky"] = np.where(reported, np.nanmax(np.where(np.isfinite(covers), covers, -1), axis=1), np.nan)
    lowest = np.nanmin(np.where(ceiling & np.isfinite(heights), heights, np.inf), axis=1)
    frame["ceiling_kft"] = np.where(reported, np.minimum(lowest, CEILING_CAP_KFT), np.nan)
    minute = frame.index.minute
    routine_minute = pd.Series(minute).mode().iloc[0]
    frame["routine"] = minute == routine_minute
    return frame


def load_reports(stations=STATIONS):
    out = {}
    for station in stations:
        parts = [pd.read_csv(RAW / f"{station}_{year}.csv", dtype="string") for year in station_years(station)
                 if (RAW / f"{station}_{year}.csv").exists()]
        if parts:
            out[station] = clean(pd.concat(parts, ignore_index=True))
    return out


# Causal lookups -----------------------------------------------------------------------------------

def latest(frame, column, cutoffs, allowance):
    """Value and valid time of the newest report with `column` present, valid < cutoff - allowance and
    no more than STALE older than the cutoff. cutoffs: UTC DatetimeIndex."""
    have = frame[column].dropna()
    times = have.index.asi8
    limit = (cutoffs - allowance).asi8
    position = np.searchsorted(times, limit, side="left") - 1  # strictly before the limit
    ok = position >= 0
    when = np.where(ok, times[np.maximum(position, 0)], np.iinfo(np.int64).min)
    ok &= when >= (cutoffs - STALE).asi8
    values = np.where(ok, have.to_numpy(float)[np.maximum(position, 0)], np.nan)
    return values, pd.DatetimeIndex(np.where(ok, when, np.iinfo(np.int64).min).astype("datetime64[ns]"), tz="UTC").where(ok)


def precipitation(frame, cutoffs, allowance, hours=3):
    """Sum of routine-report hourly precipitation with valid time in [limit - 3 h, limit)."""
    have = frame.loc[frame.routine, "p01i"].dropna()
    times, total = have.index.asi8, np.r_[0.0, np.cumsum(have.to_numpy(float))]
    limit = (cutoffs - allowance).asi8
    start = limit - pd.Timedelta(hours=hours).value
    a, b = np.searchsorted(times, start, side="left"), np.searchsorted(times, limit, side="left")
    return np.where(b > a, total[b] - total[a], np.nan)


def build_rows(reports, dates, hour, allowance_min):
    """Feature table for one cutoff hour and allowance. reports: {station: cleaned frame}."""
    allowance = pd.Timedelta(minutes=allowance_min)
    cutoffs = pd.DatetimeIndex([local_cutoff(day, hour) for day in dates]).tz_convert("UTC")
    out = pd.DataFrame({"date": pd.DatetimeIndex(dates), "hour": hour, "allowance_min": allowance_min,
                        "cutoff_utc": cutoffs})
    lga = reports["LGA"]
    lga_temp, out["LGA_report_utc"] = latest(lga, "tmpf", cutoffs, allowance)
    out["LGA_u"], _ = latest(lga, "u", cutoffs, allowance)
    out["LGA_v"], _ = latest(lga, "v", cutoffs, allowance)
    out["LGA_gust"], _ = latest(lga, "gust_or_speed", cutoffs, allowance)
    out["LGA_alti"], _ = latest(lga, "alti", cutoffs, allowance)
    before, _ = latest(lga, "alti", cutoffs - pd.Timedelta(hours=3), allowance)
    out["LGA_alti_change_3h"] = out.LGA_alti - before
    out["LGA_dewpoint_depression"], _ = latest(lga, "dewpoint_depression", cutoffs, allowance)
    out["LGA_sky"], _ = latest(lga, "sky", cutoffs, allowance)
    out["LGA_ceiling_kft"], _ = latest(lga, "ceiling_kft", cutoffs, allowance)
    out["LGA_precip_3h"] = precipitation(lga, cutoffs, allowance)
    for station in STATIONS[1:]:
        if station not in reports:
            continue
        if station in NEW_STATIONS:
            temp, _ = latest(reports[station], "tmpf", cutoffs, allowance)
            out[f"{station}_temp_vs_lga"] = temp - lga_temp
        out[f"{station}_sky"], _ = latest(reports[station], "sky", cutoffs, allowance)
    return out


def build(first="2020-01-01", last="2026-08-31"):
    reports = load_reports()
    dates = pd.date_range(first, last, freq="D")
    table = pd.concat([build_rows(reports, dates, hour, a) for a in ALLOWANCES for hour in HOURS], ignore_index=True)
    table["cutoff_utc"] = table.cutoff_utc.dt.strftime("%Y-%m-%d %H:%M")
    table["LGA_report_utc"] = table.LGA_report_utc.dt.strftime("%Y-%m-%d %H:%M")
    table.to_csv(FEATURES, index=False, float_format="%.4f", compression={"method": "gzip", "mtime": 0})
    primary = table.loc[table.allowance_min.eq(ALLOWANCES[0])]
    print(primary.groupby("hour")[LGA_FEATURES + NEARBY_FEATURES].apply(lambda f: f.notna().mean()).T.round(3))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch", "build", "all"])
    args = parser.parse_args()
    if args.step in ("fetch", "all"):
        fetch()
    if args.step in ("build", "all"):
        build()


if __name__ == "__main__":
    main()
