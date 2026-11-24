"""Offline 09:00 features. Build every calendar date without consulting its outcome."""
import json

import numpy as np
import pandas as pd

from panels import ROOT, TABLE, TZ
from data import STATION_IDS, clean, issue_index, station_inputs
from experiment import season_features
from late_replay import station_archive
from market_study import membership

SNAPSHOT = ROOT/"data"/"morning_09.parquet"
TIMES = pd.date_range("2021-01-01","2026-11-21",name="date")
TAIL = pd.Timestamp("2026-11-07")


def published(frame,column,issues):
    stamps = pd.to_datetime(frame[column],utc=True,format="mixed",errors="coerce")
    return stamps.notna() & (stamps <= issues)


def observation_features(obs,days):
    values = station_inputs(obs,days,9,5)
    at = issue_index(days,9)
    if (values.observed>at-pd.Timedelta(minutes=5)).any():
        raise ValueError("station report later than receipt-adjusted cutoff")
    clock = obs.index.tz_localize(None)
    early = clock-clock.normalize()<=pd.Timedelta(hours=8,minutes=55)
    temp = obs.temp.loc[early]
    values["early_high"] = temp.groupby(temp.index.tz_localize(None).normalize()).max().reindex(days)
    return values


def build():
    days = TIMES
    later = station_archive(cached_only=True)
    parts = []
    for station,code in STATION_IDS.items():
        paths = sorted((ROOT/"data"/"raw").glob(f"LCD_{code}_*.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing NOAA LCD sources for {station}")
        raw = pd.concat([clean(path)[0] for path in paths]).sort_index(kind="stable")
        obs = pd.concat([raw.loc[raw.index<TAIL.tz_localize(TZ)],later[station].loc[
            later[station].index>=TAIL.tz_localize(TZ)]])
        obs = obs.loc[~obs.index.duplicated(keep="last")]
        parts.append(observation_features(obs,days).add_prefix(station+"_"))
    f = season_features(pd.concat(parts,axis=1))
    ext = pd.read_parquet(ROOT/"data/rolling/external.parquet").query("hour==9")
    ext = ext.set_index(pd.to_datetime(ext.date)).reindex(days)
    issues = pd.Series(issue_index(days,9),index=days)
    nwp = published(ext,"cycle_available",issues)&published(ext,"cycle_run",issues)
    cloud = published(ext,"cloud_available",issues)&nwp
    sat = published(ext,"goes_available",issues)&published(ext,"goes_end",issues)
    for k in ("latest_high","latest_warm"):
        f[k] = ext[k].where(nwp)
    f["model_cloud"] = ext.model_cloud.where(cloud)
    f["sat10"] = ext.sat10.where(sat)
    f["sat40"] = ext.sat40.where(sat)
    nbm = pd.read_parquet(TABLE).query("hour==9")
    nbm = nbm.set_index(pd.to_datetime(nbm.date)).reindex(days)
    ready = published(nbm,"published",issues)&published(nbm,"run",issues)&nbm.status.eq("ok")
    for k in ("nbm_high","nbm_peak_hours","nbm_spread","nbm_cloud"):
        f[k] = nbm[k].where(ready)
    for name,stamp,mask in [("nbm_time",nbm.published,ready),("hrrr_time",ext.cycle_available,nwp),
                            ("cloud_time",ext.cloud_available,cloud),("sat_time",ext.goes_available,sat)]:
        f[name] = pd.to_datetime(stamp,utc=True,format="mixed",errors="coerce").where(mask)
    f["hrrr_09_bias"] = f.LGA_temp-(f.latest_high-f.latest_warm)
    f["nbm_minus_hrrr"] = f.nbm_high-f.latest_high
    for station in ("JFK","EWR","TEB"):
        f[f"{station}_minus_lga"] = f[f"{station}_temp"]-f.LGA_temp
    for date,row in ext.loc[nwp].iterrows():
        run = pd.Timestamp(row.cycle_run)
        path = ROOT.parent/"nowcast/cache/cycles"/f"{run:%Y%m%d%H}-TMP.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        at = issues.loc[date]
        if pd.Timestamp(record["available"])>at or pd.Timestamp(record["run"])!=run:
            continue
        values = np.array(record["values"])
        steps = pd.date_range(run+pd.Timedelta(hours=1),periods=len(values),freq="h")
        future = values[(steps>=at)&(steps<at.normalize()+pd.DateOffset(days=1))]
        if len(future) >= 7:
            f.loc[date,["hrrr_noon_warm","hrrr_15_warm","hrrr_peak_hours","hrrr_plateau"]] = [
                future[3]-future[0],future[6]-future[0],np.argmax(future),np.sum(future>=future.max()-1)]
    # Forecast weather exists for all calendar dates irrespective of label coverage.
    f.to_parquet(SNAPSHOT)
    return f


def market_layouts(raw,hour=9):
    """Return as-of contract definitions only. A partition needs both tails and no missing interval."""
    layouts = {}
    for date,group in raw.groupby("date"):
        issue = (pd.Timestamp(date)+pd.Timedelta(hours=hour)).tz_localize(TZ)
        group = group.loc[group.listed.notna() & (group.listed<=issue)].sort_values("low",na_position="first")
        if len(group)<3 or group.regime.nunique()!=1:
            continue
        if (group.closed.notna() & (group.closed<=issue)).all():
            continue
        bins = membership(group.low.to_numpy(float),group.high.to_numpy(float))
        if not (bins.sum(axis=0)==1).all():
            continue
        left = group.high.iloc[0]
        right = group.low.iloc[-1]
        if not (pd.isna(group.low.iloc[0]) and pd.isna(group.high.iloc[-1]) and left<right):
            continue
        layouts[date] = dict(regime=group.regime.iloc[0],ids=group.market_id.astype(str).tolist(),
            brackets=group.bracket.tolist(),bins=bins,center=float((left+right)/2),
            latest_listing=group.listed.max())
    return layouts


def settlements(raw):
    yes = raw.loc[raw.winner,["date","market_id"]].set_index("date")
    if not yes.index.is_unique:
        raise ValueError("nonunique settlement")
    return yes.assign(market_id=yes.market_id.astype(str),closed=raw.groupby("date").closed.max())


if __name__=="__main__":
    panel = build()
    print(f"{len(panel)} mornings, {panel.index.min().date()} through {panel.index.max().date()}")
