"""Shared station, numerical forecast and satellite feature definitions."""
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sources import HERE, TZ

STATIONS = {"LGA":"14732", "JFK":"94789", "EWR":"14734", "TEB":"94741"}
BOUNDS = np.array([40., 55., 70., 85.])
COVER = {"CLR":0, "SKC":0, "FEW":0.25, "SCT":0.5, "BKN":0.875, "OVC":1, "VV":1}
ONE_HOUR = pd.Timedelta(hours=1)
SAT_LOCAL = ["sat10","sat40","sat_minus_model","clear_sun"]
SAT_MOTION = ["upwind","delta30","delta60","upwind_delta30","incoming_shade"]
NWP = ["nwp_high","nwp_warm","peak_hours","nwp_bias","nwp_cloud","run_age"]


def coverage(text):
    values = [COVER[c] for c in re.findall(r"CLR|SKC|FEW|SCT|BKN|OVC|VV", str(text))]
    return max(values, default=np.nan)


def archive_stations():
    result = {}
    for station, code in STATIONS.items():
        parts = []
        for path in sorted((HERE.parent/"final_project/data/raw").glob(f"LCD_USW000{code}_*.csv")):
            refreshed = HERE / "cache/lcd" / path.name
            if refreshed.exists():
                path = refreshed
            fields = ["DATE","REPORT_TYPE","HourlyDryBulbTemperature","HourlyDewPointTemperature",
                      "HourlyWindSpeed","HourlyWindDirection","HourlySkyConditions"]
            df = pd.read_csv(path, usecols=fields, dtype="string")
            df = df.loc[df.REPORT_TYPE.isin(["FM-15","FM-16"])]
            frame = pd.DataFrame(index=df.index)
            for field, column in zip(fields[2:6], ["temp", "dew", "speed","direction"]):
                frame[column] = pd.to_numeric(df[field], errors="coerce").astype(float)
            for col in ["temp","dew"]:
                frame[col] = frame[col].where(frame[col].between(-75,60))*1.8+32
            frame["cover"] = df.HourlySkyConditions.map(coverage)
            frame.index = pd.DatetimeIndex(df.DATE).tz_localize("Etc/GMT+5").tz_convert(TZ)
            parts.append(frame)
        df = pd.concat(parts).sort_index()
        result[station] = df[~df.index.duplicated(keep="last")]
    return result


def labels(lga):
    closed = lga.groupby(lga.index.tz_convert("UTC").floor("h")).tail(1).dropna(subset=["temp"])
    grouped = closed.assign(period=closed.index.hour//6).groupby(closed.index.date)
    bounds = grouped.agg(high=("temp","max"), n=("temp","size"), periods=("period","nunique"))
    complete = bounds.loc[(bounds.n >= 20) & (bounds.periods == 4)]
    return complete.high


def fresh(table, time):
    i = table.index.searchsorted(time, side="right")-1
    if i < 0 or time - table.index[i] > ONE_HOUR:
        return pd.Series(dtype=float), None
    return table.iloc[i], table.index[i]


def ground(stations, at, delay=0):
    """Historical delay simulates transmission latency. Live tables use actual receipt time."""
    stamp = at - pd.Timedelta(minutes=delay)
    hours = at.hour + at.minute/60
    day_angle = 2*np.pi*(at.dayofyear-1)/365.25
    decl = np.deg2rad(23.44)*np.sin(2*np.pi*(at.dayofyear-81)/365.25)
    # Equation of time and site longitude convert UTC to apparent solar time.
    b = 2*np.pi*(at.dayofyear-81)/364
    eq = 9.87*np.sin(2*b)-7.53*np.cos(b)-1.5*np.sin(b)
    solar_time = at.tz_convert("UTC").hour + at.minute/60 - 73.87397/15 + eq/60
    lat = np.deg2rad(40.77945)
    features = {"hour":hours,"season_sin":np.sin(day_angle),"season_cos":np.cos(day_angle),
        "sun":np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(np.deg2rad(15*(solar_time-12)))}
    stamps = {}
    for name in STATIONS:
        table = stations[name]
        if "received" in table:
            table = table.loc[table.received <= at]
        row, ts = fresh(table, stamp)
        stamps[name] = ts.isoformat() if ts is not None else None
        if name == "LGA" and (ts is None or not np.isfinite(row.get("temp", np.nan)) or at-ts > ONE_HOUR):
            return None, stamps
        current = row.get("temp",np.nan)
        for col in ["temp","dew","cover","speed"]:
            features[f"{name}_{col}"] = row.get(col,np.nan)
        direction = np.deg2rad(row.get("direction",np.nan))
        features[f"{name}_east"] = np.sin(direction)*row.get("speed",np.nan)
        features[f"{name}_north"] = np.cos(direction)*row.get("speed",np.nan)
        previous, prev_ts = fresh(table, stamp-ONE_HOUR)
        features[f"{name}_trend"] = ((current-previous.get("temp",np.nan))/((ts-prev_ts)/ONE_HOUR)
                                      if ts is not None and prev_ts is not None and ts>prev_ts else np.nan)
        if name == "LGA":
            available = table.loc[at.normalize():stamp]
            # In delayed replay only hours fully observed by stamp may impose a lower bound.
            end = min(stamp.floor("h"), (at-pd.Timedelta(minutes=15)).floor("h"))
            obs = available.loc[available.index < end]
            hourly = obs.groupby(obs.index.tz_convert("UTC").floor("h")).tail(1)
            features["high_so_far"] = hourly.temp.max()
            features["direction"] = row.get("direction",np.nan) if row.get("speed",0)>0 else np.nan
        else:
            features[f"{name}_minus_LGA"] = current-features["LGA_temp"]
    high = features["high_so_far"]
    warmer = BOUNDS[BOUNDS>high]
    features["to_boundary"] = warmer.min()-high if len(warmer) else np.nan
    return features, stamps


def numerical(at, ground_features, cycle, cloud):
    data = dict.fromkeys(NWP,np.nan)
    if not cycle:
        return data
    run = pd.Timestamp(cycle["run"])
    if pd.Timestamp(cycle["available"]) > at or run > at:
        return data
    leads = np.arange(1,len(cycle["values"])+1)
    values = np.asarray(cycle["values"])
    elapsed = (at-run)/ONE_HOUR
    midnight = (at.normalize()+pd.DateOffset(days=1)-run)/ONE_HOUR
    if leads[-1] < midnight-1 or elapsed < 1:
        return data
    future = (leads >= elapsed) & (leads < midnight)
    if not future.any():
        return data
    current = np.interp(elapsed,leads,values)
    peak = np.argmax(values[future])
    high = values[future][peak]
    data.update(nwp_high=high,nwp_warm=high-current,peak_hours=leads[future][peak]-elapsed,
                nwp_bias=ground_features["LGA_temp"]-current,run_age=elapsed)
    if cloud and cloud["run"] == cycle["run"] and pd.Timestamp(cloud["available"]) <= at:
        data["nwp_cloud"] = np.interp(elapsed,np.arange(1,len(cloud["values"])+1),cloud["values"])/100
    return data


def fusion(at, base, snapshot, fixed=False):
    fc = snapshot.get("forecast",{})
    data = {**base, **numerical(at,base,fc.get("fixed" if fixed else "latest"), fc.get("cloud"))}
    data.update(dict.fromkeys(SAT_LOCAL+SAT_MOTION,np.nan))
    scans = snapshot.get("scans",[])
    def value(i, region):
        if i >= len(scans) or not scans[i]:
            return np.nan
        scan = scans[i]
        ref = at-pd.Timedelta(minutes=(0,30,60)[i])
        if pd.Timestamp(scan["available"]) > ref or not 0 <= (ref-pd.Timestamp(scan["end"])).total_seconds() <= 1800:
            return np.nan
        v = scan["fraction"][region]
        return np.nan if v is None else v
    data["sat10"], data["sat40"] = value(0,0),value(0,1)
    data["clear_sun"] = max(0,base["sun"])*(1-data["sat10"])
    data["delta30"],data["delta60"] = value(0,1)-value(1,1),value(0,1)-value(2,1)
    data["sat_minus_model"] = data["sat40"]-data["nwp_cloud"]
    if np.isfinite(base["direction"]):
        sector = int((base["direction"]+22.5)//45)%8+2
        data["upwind"] = value(0,sector)
        data["upwind_delta30"] = value(0,sector)-value(1,sector)
        data["incoming_shade"] = (data["upwind"]-data["sat10"])*max(0,base["sun"])
    return data

