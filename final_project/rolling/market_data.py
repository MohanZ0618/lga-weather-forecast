"""Archive NYC temperature contract definitions, final resolutions and LGA METAR reports."""
import gzip
import hashlib
import json
import re

import numpy as np
import pandas as pd
import requests

from data import OUT, SNAPSHOT_END, STORE, TZ, sources

MARKET = OUT/"polymarket"
RAW = STORE/"raw_tail"/"polymarket"
API = "https://gamma-api.polymarket.com/events"
SERIES = "10005"


def bounds(title):
    value = title.strip().lower().replace("°f","").replace("–","-").replace("−","-").strip()
    m = re.fullmatch(r"([<>])\s*(-?\d+)",value)
    if m:
        return (None,int(m[2])-1) if m[1]=="<" else (int(m[2])+1,None)
    m = re.fullmatch(r"(-?\d+)\s*(?:or below|or lower|or less)",value)
    if m:
        return None,int(m[1])
    m = re.fullmatch(r"(-?\d+)\s*(?:or higher|or above|or more)",value)
    if m:
        return int(m[1]),None
    m = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)",value)
    if m and int(m[1])<=int(m[2]):
        return int(m[1]),int(m[2])
    if re.fullmatch(r"-?\d+",value):
        return int(value),int(value)
    raise ValueError(f"unknown range {title}")


def partition(items):
    bins = sorted(items,key=lambda b:-np.inf if b["low"] is None else b["low"])
    if bins[0]["low"] is not None or bins[-1]["high"] is not None:
        raise ValueError("missing outer range")
    if any(a["high"] is None or b["low"]!=a["high"]+1 for a,b in zip(bins,bins[1:])):
        raise ValueError("gap or overlapping brackets")
    return bins


def regime(text):
    t = text.lower()
    if "laguardia" not in t and "klga" not in t:
        raise ValueError("different station")
    if "hourly temperature remark" in t and "speci" in t and "routine" in t:
        return "routine_metar"
    if "recorded by noaa" in t and "weather.gov/wrh/timeseries" in t:
        return "noaa_timeseries"
    if "wunderground" in t or "weather underground" in t:
        return "wunderground"
    return "other_KLGA"


def decode(event):
    if not event["closed"] or not event["slug"].startswith(("highest-temperature-in-nyc-on-",
                                                            "highest-temperature-in-new-york-city-on-")):
        raise ValueError("not closed NYC daily high")
    if event.get("seriesSlug") not in (None,"nyc-daily-weather"):
        raise ValueError("different series")
    guess = pd.Timestamp(event.get("eventDate") or event["endDate"])
    named = re.search(r"\bon\s+([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b",event["title"],re.I)
    if not named:
        raise ValueError("missing target date in title")
    day = pd.Timestamp(f"{named[1]} {named[2]} {guess.year}").date()
    if day>pd.Timestamp(SNAPSHOT_END).date():
        raise ValueError("outside snapshot")
    bins = []
    for m in event["markets"]:
        title = m["groupItemTitle"]
        lo,hi = bounds(title)
        outcomes,prices = json.loads(m["outcomes"]),json.loads(m["outcomePrices"])
        if not m["closed"] or m.get("umaResolutionStatus")!="resolved" or set(outcomes)!={"Yes","No"}:
            raise ValueError("incomplete final resolution")
        yes,no = (float(prices[outcomes.index(q)]) for q in ("Yes","No"))
        if (yes,no) not in [(1.,0.),(0.,1.)]:
            raise ValueError("nonbinary payout")
        start = m.get("startDate") or m.get("createdAt")
        if not start:
            raise ValueError("missing listing timestamp")
        bins.append(dict(date=str(day),event_id=event["id"],slug=event["slug"],market_id=m["id"],
                         bracket=title,low=lo,high=hi,winner=bool(yes),listed=start,closed=m.get("closedTime"),
                         regime=regime(m["description"]),rules=m["description"]))
    if len(set(b["regime"] for b in bins))!=1 or sum(b["winner"] for b in bins)!=1:
        raise ValueError("inconsistent rules or no unique winning bracket")
    return partition(bins)


def events():
    RAW.mkdir(parents=True,exist_ok=True)
    MARKET.mkdir(parents=True,exist_ok=True)
    all_events,manifest = {},[]
    offset = 0
    while True:
        params = dict(series_id=SERIES,closed="true",order="endDate",ascending="true",limit=50,offset=offset,
                      end_date_max="2026-11-22T00:00:00Z")
        path = RAW/f"events-{offset:04}.json.gz"
        if path.exists():
            content = gzip.decompress(path.read_bytes())
        else:
            req = sources.get(API,params=params)
            req.raise_for_status()
            content = req.content
            page = json.loads(content)
            if not isinstance(page,list):
                raise ValueError("Unexpected Gamma response")
            path.write_bytes(gzip.compress(content,mtime=0))
        page = json.loads(content)
        url = requests.Request("GET",API,params=params).prepare().url
        manifest.append(dict(url=url,sha256=hashlib.sha256(content).hexdigest(),events=len(page)))
        before = len(all_events)
        all_events.update({e["id"]:e for e in page})
        if len(page)<50:
            break
        if before==len(all_events):
            raise ValueError("No progress in Gamma pagination")
        offset += len(page)
    rows,skipped = [],[]
    for event in all_events.values():
        try:
            rows.extend(decode(event))
        except (ValueError,KeyError,TypeError) as exc:
            skipped.append(dict(id=event["id"],slug=event["slug"],reason=str(exc)))
    frame = pd.DataFrame(rows)
    duplicates = frame.groupby("date").event_id.nunique().loc[lambda x:x>1].index
    for date in duplicates:
        skipped.append(dict(slug=date,reason="ambiguous duplicate target date"))
    frame = frame.loc[~frame.date.isin(duplicates)]
    frame.to_parquet(MARKET/"contracts.parquet",index=False)
    pd.DataFrame(skipped,columns=["id","slug","reason"]).to_csv(MARKET/"excluded_events.csv",index=False)
    pd.DataFrame(manifest).to_csv(MARKET/"gamma_sources.csv",index=False)
    print("Validated events",frame.date.nunique(),"of",len(all_events),"by regime",
          frame.groupby("regime").date.nunique().to_dict(),flush=True)


def whole_f(fahrenheit):
    return np.floor(np.asarray(fahrenheit)+0.5)


def metar_tenths(report):
    tokens = str(report).strip().split()
    if not tokens:
        return np.nan
    body,_,remark = (" "+" ".join(tokens)+" ").partition(" RMK ")
    match = re.search(r"\bT([01])(\d{3})\d{4}\b",remark)
    if match:
        return (-1 if match[1]=="1" else 1)*int(match[2])
    match = re.search(r"\b(M?\d{2})/(?:M?\d{2}|//)?(?=\s|=|$)",body)
    if match:
        return int(match[1].replace("M","-"))*10
    return np.nan


def awc_day(date):
    day = pd.Timestamp(date)
    end = day+pd.Timedelta(days=1,hours=8)
    url = "https://aviationweather.gov/api/data/metar"
    params = dict(ids="KLGA",format="raw",date=end.strftime("%Y-%m-%dT%H:%M:%SZ"),hours=30)
    path = RAW/f"awc-{date}.txt"
    status = 200
    if path.exists():
        content = path.read_bytes()
    else:
        response = sources.get(url,params=params)
        status = response.status_code
        if status in (204,400,404):
            content = b""
        else:
            response.raise_for_status()
            content = response.content
        if content.strip():
            path.write_bytes(content)
    rows = []
    for line in content.decode().splitlines():
        stamp = re.match(r"^(?:METAR\s+)?(?:COR\s+)?KLGA\s+(\d\d)(\d\d)(\d\d)Z\b",line.strip())
        if not stamp:
            continue
        dd,hh,mm = map(int,stamp.groups())
        for d in (day,day+pd.Timedelta(days=1)):
            if d.day!=dd:
                continue
            clock = (d+pd.Timedelta(hours=hh,minutes=mm)).tz_localize("UTC").tz_convert(TZ)
            if clock.date()==day.date():
                value = metar_tenths(line)
                if np.isfinite(value):
                    rows.append(dict(time=clock,f=(value*9+1600)/50,special=False))
    meta = dict(url=requests.Request("GET",url,params=params).prepare().url,status=status,
                sha256=hashlib.sha256(content).hexdigest() if content else None,rows=len(rows),date=date)
    return rows,meta


def make_labels():
    from data import DATA,lcd_local_time
    frames,meta = [],[]
    for path in sorted((DATA/"raw").glob("LCD_USW00014732_*.csv")):
        raw = pd.read_csv(path,usecols=["DATE","REPORT_TYPE","HourlyDryBulbTemperature"],dtype="string")
        raw = raw.loc[raw.REPORT_TYPE.isin(["FM-15","FM-16"])]
        c = pd.to_numeric(raw.HourlyDryBulbTemperature,errors="coerce")
        frame = pd.DataFrame(dict(time=lcd_local_time(raw.DATE),f=c.where(c.between(-75,60))*1.8+32,
                                  special=raw.REPORT_TYPE.eq("FM-16")))
        frames.append(frame)
    obs = pd.concat(frames).set_index("time").sort_index(kind="stable").loc["2018-01-01":SNAPSHOT_END]
    routine = obs.loc[~obs.special & obs.f.notna()].copy()
    routine = routine.loc[~routine.index.duplicated(keep="last")]
    regime_dates = pd.read_parquet(MARKET/"contracts.parquet").query("regime=='routine_metar'").date.unique()
    direct = []
    for date in sorted(regime_dates):
        rows,info = awc_day(date)
        meta.append(info)
        direct.extend(rows)
    authoritative = pd.DataFrame(direct,columns=["time","f","special"]).set_index("time")
    official_days = set()
    if len(authoritative):
        authoritative = authoritative.loc[~authoritative.index.duplicated(keep="first")].sort_index()
        for date,group in authoritative.groupby(authoritative.index.date):
            # A truncated response at the edge of the API archive must not displace a complete archive day.
            if group.index.tz_convert("UTC").floor("h").nunique()>=20 and len(set(group.index.hour//6))==4:
                official_days.add(date)
        routine = pd.concat([routine.loc[~np.isin(routine.index.date,list(official_days))],
                             authoritative.loc[np.isin(authoritative.index.date,list(official_days))]]).sort_index()
    day = routine.index.tz_localize(None).normalize()
    grouped = routine.assign(h=routine.index.tz_convert("UTC").floor("h"),block=routine.index.hour//6).groupby(day)
    daily = grouped.agg(f=("f","max"),hours=("h","nunique"),blocks=("block","nunique"))
    daily["routine_f"] = whole_f(daily.f)
    daily["routine_source"] = np.where(np.isin(daily.index.date,list(official_days)),"AWC raw","LCD FM-15 proxy")
    from late_replay import station_archive
    tail = station_archive()["LGA"].temp
    cutoff = pd.Timestamp("2026-11-07",tz=TZ)
    all_temp = pd.concat([obs.loc[obs.index<cutoff,"f"],tail.loc[tail.index>=cutoff]])
    all_highs = all_temp.groupby(all_temp.index.tz_localize(None).normalize()).max().loc[:SNAPSHOT_END]
    daily = daily.reindex(daily.index.union(all_highs.index)).sort_index()
    daily["all_f"] = whole_f(all_highs.reindex(daily.index))
    for h in (9,12,15):
        reported = routine.loc[routine.index.hour<h]
        daily[f"routine_floor_{h}"] = whole_f(reported.f.groupby(reported.index.tz_localize(None).normalize()).max()
                                                .reindex(daily.index))
    daily.index.name="date"
    daily.to_csv(MARKET/"integer_labels.csv")
    pd.DataFrame(meta).to_csv(MARKET/"awc_sources.csv",index=False)
    print(len(daily),"local days;",len(official_days),"complete days from the contract's AWC raw endpoint",flush=True)


if __name__=="__main__":
    events()
    make_labels()
