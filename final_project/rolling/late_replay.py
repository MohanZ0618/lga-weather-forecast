"""Separate METAR replay for complete dates after the NOAA LCD archive ends."""
import argparse
import hashlib
import json
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import requests

from data import HOURS, OUT, SNAPSHOT_END, STATION_IDS, STORE, TZ, coverage, station_inputs
from experiment import PROBS, metrics, season_features
from rolling_models import predict_route
from study import add_forecasts
from weather import daily_labels, hourly_rows

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpf","dwpf","relh","drct","sknt","vsby","skyc1","skyc2","skyc3","skyc4"]


def station_archive():
    path = STORE/"raw_tail"/"iem_metar.csv"
    params = dict(station=list(STATION_IDS),data=FIELDS,year1=2026,month1=8,day1=31,
                  year2=2026,month2=11,day2=23,tz="Etc/UTC",format="onlycomma",
                  latlon="no",missing="M",trace="T",direct="no",report_type=[3,4])
    url = requests.Request("GET",URL,params=params).prepare().url
    if path.exists():
        content = path.read_bytes()
    else:
        response = requests.get(url,timeout=90)
        response.raise_for_status()
        content = response.content
    df = pd.read_csv(BytesIO(content),na_values=["M"],usecols=["station","valid",*FIELDS])
    if set(df.station)!=set(STATION_IDS):
        raise ValueError("Incomplete station response from the historical METAR archive")
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    (OUT/"iem_source.json").write_text(json.dumps(dict(url=url,sha256=hashlib.sha256(content).hexdigest(),
                                                   reports=len(df),first=df.valid.min(),last=df.valid.max()),indent=2))
    tables = {}
    for station,g in df.groupby("station"):
        g = g.set_index(pd.to_datetime(g.valid,utc=True).dt.tz_convert(TZ)).sort_index(kind="stable")
        g = g.loc[~g.index.duplicated(keep="last")]
        out = pd.DataFrame(index=g.index)
        for field,low,high in [("tmpf",-103,140),("dwpf",-112,140),("drct",0,360),("sknt",0,230),
                               ("relh",0,100),("vsby",0,125)]:
            raw = pd.to_numeric(g[field],errors="coerce")
            g[field] = raw.where(raw.between(low,high))
        out["temp"],out["rh"] = g.tmpf,g.relh
        out["wind"] = g.sknt*.514444
        out["spread"] = g.tmpf-g.dwpf
        out["east"] = (-out.wind*np.sin(np.deg2rad(g.drct))).mask(out.wind.eq(0),0)
        out["north"] = (-out.wind*np.cos(np.deg2rad(g.drct))).mask(out.wind.eq(0),0)
        out["sky"] = g[["skyc1","skyc2","skyc3","skyc4"]].apply(lambda col:col.map(coverage)).max(axis=1)
        out["vis"] = g.vsby*1.609344
        tables[station] = out
    return tables


def replay(tables):
    labels = daily_labels(hourly_rows(tables["LGA"])).loc["2026-09-01":SNAPSHOT_END]
    lcd = pd.read_parquet(STORE/"stations.parquet",filters=[("hour","==",9),("delay","==",0)])
    overlap = lcd.set_index("date")[["target_max","target_category"]].join(labels,how="inner",rsuffix="_iem")
    overlap.to_csv(OUT/"iem_lcd_overlap.csv")
    tail = labels.loc[labels.index>lcd.date.max()]
    if tail.empty:
        print("No additional eligible METAR days")
        return
    first = tail.index.min().date()
    predictions,summary = [],[]
    for hour in HOURS:
        ground = pd.concat([station_inputs(table,tail.index,hour,0).add_prefix(s+"_")
                            for s,table in tables.items()],axis=1)
        case = add_forecasts(season_features(ground.join(tail)),hour)
        bundle = joblib.load(OUT/f"{hour:02}"/"model.joblib")
        for name,b in [("candidate",bundle),("reference",bundle["reference"])]:
            p,has = predict_route(case,b)
            summary.append(dict(hour=hour,model=name,first=str(first),last=str(tail.index.max().date()),
                                source_days=int(has.sum()),**metrics(tail.target_category.to_numpy(),p)))
            rows = pd.DataFrame(p,index=tail.index,columns=PROBS)
            predictions.append(rows.assign(hour=hour,model=name,source_available=has,target=tail.target_category))
    pd.concat(predictions).to_csv(OUT/"iem_late_predictions.csv.gz")
    pd.DataFrame(summary).to_csv(OUT/"iem_late_metrics.csv",index=False)
    print(len(overlap),"overlap days,",int(overlap.target_category.eq(overlap.target_category_iem).sum()),
          "matching classes,",len(tail),"additional days")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only",action="store_true")
    args = parser.parse_args()
    tables = station_archive()
    if not args.fetch_only:
        replay(tables)
