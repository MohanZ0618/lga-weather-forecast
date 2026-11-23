"""Run an as-of quarter-hour forecast and replay the saved inputs offline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests

from features import STATIONS, TZ, coverage, fusion, ground
from analogue import augment
from sources import CACHE, HERE, external, get, save

ARTIFACT = HERE/"artifacts"/"model.joblib"
SCHEMA = 1


def latest_slot(now=None):
    now = (pd.Timestamp.now(tz=TZ) if now is None else pd.Timestamp(now).tz_convert(TZ))
    at = now.floor("15min", ambiguous=False)
    # Local wall-clock construction also works on 23- and 25-hour days.
    wall_date = at.tz_localize(None).normalize()
    if at.hour < 6:
        wall_date -= pd.Timedelta(days=1)
    last = (wall_date + pd.Timedelta(hours=20)).tz_localize(TZ)
    return min(at,last)


def numeric(value):
    try:
        return float(value)
    except (ValueError,TypeError):
        return np.nan


def metar_tables(records):
    tables = {}
    for station in STATIONS:
        rows = []
        for r in records:
            if r.get("icaoId") != "K"+station or not r.get("receiptTime") or not r.get("obsTime"):
                continue
            cloud = coverage([r.get("cover"),*(layer.get("cover") for layer in r.get("clouds") or [])])
            rows.append({"time":pd.Timestamp(r["obsTime"],unit="s",tz="UTC").tz_convert(TZ),
                         "received":pd.Timestamp(r["receiptTime"]).tz_convert(TZ),
                         "temp":numeric(r.get("temp"))*1.8+32,"dew":numeric(r.get("dewp"))*1.8+32,
                         "speed":numeric(r.get("wspd"))*.514444,"direction":numeric(r.get("wdir")),
                         "cover":cloud})
        table = pd.DataFrame(rows,columns=["time","received","temp","dew","speed","direction","cover"])
        if table.empty:
            table["time"] = pd.DatetimeIndex([],tz=TZ)
            table["received"] = pd.DatetimeIndex([],tz=TZ)
        tables[station] = table.sort_values(["time","received"]).set_index("time")
    return tables


def select_tier(row, artifact):
    if np.isfinite(row["nwp_high"]):
        if np.isfinite(row["sat10"]):
            return artifact["routing"]["satellite"]
        return artifact["routing"]["nwp"]
    return "station"


def infer(at, payload, artifact):
    if artifact["schema"] != SCHEMA:
        raise ValueError("Retrain the incompatible model artifact")
    base, stamps = ground(metar_tables(payload["metars"]),at)
    if base is None or not np.isfinite(base["high_so_far"]):
        return {"status":"stale_lga","cutoff":at.isoformat(),"station_times":stamps}
    ext = payload["external"]
    row = augment(fusion(at,base,ext),at,artifact["reference"])
    tier = select_tier(row,artifact)
    model = artifact["models"][tier]
    frame = pd.DataFrame([row])
    p = model.predict(frame)[0]
    comparison = {"station":artifact["models"]["station"].predict(frame)[0].tolist()}
    if np.isfinite(row["nwp_high"]):
        comparison["latest"] = artifact["models"]["latest"].predict(frame)[0].tolist()
        if np.isfinite(row["sat10"]):
            for name in ("local_cloud","upwind_cloud"):
                comparison[name] = artifact["models"][name].predict(frame)[0].tolist()
    cycle = ext.get("forecast",{}).get("latest")
    scans = ext.get("scans") or []
    measurements = {key:float(row[key]) if np.isfinite(row[key]) else None
                    for key in ("LGA_temp","sat10","sat40","upwind","delta30","nwp_high","nwp_cloud")}
    return {"status":"ok","cutoff":at.isoformat(),"version":artifact["version"],"tier":tier,
            "inputs":measurements,"available_predictions":comparison,
            "probabilities":p.tolist(),"station_times":stamps,
            "station_receipts":{sta:max((r["receiptTime"] for r in payload["metars"]
                if r.get("icaoId")=="K"+sta and r.get("receiptTime") and stamps.get(sta)
                and r.get("obsTime")==pd.Timestamp(stamps[sta]).timestamp()
                and pd.Timestamp(r["receiptTime"])<=at),default=None) for sta in STATIONS},
            "hrrr_run":cycle["run"] if cycle else None,"hrrr_available":cycle["available"] if cycle else None,
            "satellite_scan":scans[0]["end"] if scans and scans[0] else None,
            "satellite_available":scans[0]["available"] if scans and scans[0] else None,
            "source_errors":ext.get("errors",{}),"high_so_far_f":float(base["high_so_far"]) if np.isfinite(base["high_so_far"]) else None}


def fetch_payload(at):
    response = get("https://aviationweather.gov/api/data/metar",
                   params={"ids":",".join("K"+s for s in STATIONS),"hours":48,"format":"json"})
    response.raise_for_status()
    records = [] if response.status_code == 204 else response.json()
    if not isinstance(records,list):
        raise ValueError("Malformed METAR response")
    return {"cutoff":at.isoformat(),"fetched":pd.Timestamp.now(tz="UTC").isoformat(),
            "metars":records,"external":external(at,include_fixed=False)}


def run(at=None, replay=None):
    artifact = joblib.load(ARTIFACT)
    if replay is not None:
        payload = json.loads(Path(replay).read_text())
        return infer(pd.Timestamp(payload["cutoff"]).tz_convert(TZ),payload,artifact)
    at = latest_slot() if at is None else pd.Timestamp(at).tz_convert(TZ)
    if at > pd.Timestamp.now(tz=TZ) or at.minute%15 or at.second or not (6*60 <= at.hour*60+at.minute <= 20*60):
        raise ValueError("Cutoff must be a past service quarter hour")
    folder = CACHE/"live"/artifact["version"]
    key = at.strftime("%Y%m%d-%H%M")
    path = folder/(key+".json")
    if path.exists():
        return json.loads(path.read_text())
    try:
        payload = fetch_payload(at)
        result = infer(at,payload,artifact)
    except (requests.RequestException,ValueError) as e:
        result = {"status":"source_error","cutoff":at.isoformat(),"reason":str(e)}
    if result["status"] == "ok":
        save(folder/(key+".inputs.json"),payload)
        save(path,result)
    else:
        previous = sorted(p for p in folder.glob("*.json") if not p.name.endswith(".inputs.json")
                          and p.stem < key)
        if previous:
            result["last_success"] = json.loads(previous[-1].read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    print(json.dumps(run(args.cutoff,args.replay),indent=2))

