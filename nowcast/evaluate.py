"""Summarize the frozen system on paired dates, including source coverage."""
import json
import numpy as np
import pandas as pd
from features import BOUNDS
from sources import CACHE, HERE

OUT = HERE / "results"
P = [f"p{i}" for i in range(5)]


def compare(predictions, first, second):
    a = predictions.loc[predictions.level.eq(first)].set_index("cutoff")
    b = predictions.loc[predictions.level.eq(second)].set_index("cutoff")
    ix = a.index.intersection(b.index)
    a, b = a.loc[ix], b.loc[ix]
    y = a.label.to_numpy()
    pa, pb = a[P].to_numpy(), b[P].to_numpy()
    assert np.all(b.label.to_numpy() == y)
    diff = pd.DataFrame({"day":a.day,"accuracy_a":pa.argmax(axis=1)==y,"accuracy_b":pb.argmax(axis=1)==y,
        "loss_a":-np.log(pa[np.arange(len(y)),y].clip(1e-15)),
        "loss_b":-np.log(pb[np.arange(len(y)),y].clip(1e-15))})
    h = pd.to_datetime(ix,utc=True).tz_convert("America/New_York").hour
    rows = []
    for window,mask in {"all":np.ones(len(y),bool),"06-10":h<10,"10-15":(h>=10)&(h<15),"15-20":h>=15,
                        "within_2F":np.abs(a.actual.to_numpy()[:,None]-BOUNDS).min(axis=1)<=2}.items():
        if not np.any(mask):
            continue
        daily = diff.loc[mask].groupby("day").mean()
        gain = (daily.accuracy_b-daily.accuracy_a).to_numpy()
        rng = np.random.default_rng(4260)
        lo,hi = np.quantile(rng.choice(gain,(2000,len(gain)),replace=True).mean(axis=1),[.025,.975])
        rows.append({"before":first,"after":second,"window":window,"n":int(mask.sum()),"days":len(gain),
                     **daily.mean().to_dict(),"accuracy_delta":gain.mean(),"ci_low":lo,"ci_high":hi})
    return rows


def main():
    frame = pd.read_parquet(OUT/"predictions.parquet",filters=[("delay","==",0)])
    rows = []
    for period,group in frame.groupby("period"):
        for first,second in [("latest","upwind_cloud"),("station","deployed"),("local_cloud","upwind_cloud")]:
            rows.extend({"period":period,**r} for r in compare(group,first,second))
    pd.DataFrame(rows).to_csv(OUT/"paired_systems.csv",index=False)
    samples = pd.read_parquet(OUT/"sampled.parquet",filters=[("delay","==",0)])
    metadata = []
    for path in (CACHE/"snapshots").glob("*.json"):
        row=json.loads(path.read_text())
        scans=row.get("scans",[])
        meta={"cutoff":row["cutoff"],"eligible_satellite":bool(scans and scans[0] and scans[0]["fraction"][0] is not None),
              "eligible_hrrr":bool(row["forecast"].get("latest")),"source_errors":len(row["errors"])}
        metadata.append(meta)
    coverage=pd.DataFrame(metadata)
    coverage["station_and_label"]=coverage.cutoff.isin(samples.cutoff)
    coverage.sort_values("cutoff").to_csv(OUT/"coverage.csv",index=False)


if __name__ == "__main__":
    main()

