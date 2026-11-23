"""Build historical snapshots, train probability models, freeze, then score held-out dates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from backfill import schedule
from analogue import augment, build_reference
from features import BOUNDS, NWP, TZ, archive_stations, fusion, ground, labels
from models import LEVELS, Family, Bundle, choose, fit_level, predictors, score
from runtime import ARTIFACT, SCHEMA, select_tier
from sources import CACHE, HERE, save

OUT = HERE/"results"


def build(development_only=False):
    stations = archive_stations()
    truth = labels(stations["LGA"])
    reference = build_reference(stations["LGA"])
    joblib.dump(reference,OUT/"reference.joblib",compress=3)
    paths = sorted((CACHE/"snapshots").glob("*.json"))
    required = {at.strftime("%Y%m%d-%H%M") for at in schedule()}
    if development_only:
        required = {key for key in required if key < "202509"}
    paths = [p for p in paths if p.stem in required]
    absent = required - {p.stem for p in paths}
    if absent:
        raise ValueError(f"Backfill incomplete: {len(absent)} scheduled snapshots missing")
    rows = []
    for p in paths:
        snapshot = json.loads(p.read_text())
        at = pd.Timestamp(snapshot["cutoff"]).tz_convert(TZ)
        if at.date() not in truth.index:
            continue
        for lag in (0,15,30):
            base,_ = ground(stations,at,lag)
            if base is None or not np.isfinite(base["high_so_far"]):
                continue
            actual = truth.loc[at.date()]
            if base["high_so_far"] > actual+1e-6:
                raise ValueError(f"Observed lower bound exceeded daily target at {at}")
            row = augment(fusion(at,base,snapshot),at,reference)
            fixed = fusion(at,base,snapshot,True)
            row.update({"fixed_"+c:fixed[c] for c in NWP})
            rows.append({**row,"day":str(at.date()),"cutoff":at.isoformat(),"delay":lag,
                         "label":int(np.searchsorted(BOUNDS,actual,side="right")),"actual":actual})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT/"sampled.parquet",index=False)
    return df


def eligible(frame,level):
    if level == "fixed":
        subset = frame.copy()
        for c in NWP:
            subset[c] = subset["fixed_"+c]
        return subset.loc[subset.nwp_high.notna()]
    if level == "latest":
        return frame.loc[frame.nwp_high.notna()]
    if level in LEVELS[-2:]:
        return frame.loc[frame.nwp_high.notna() & frame.sat10.notna()]
    return frame


def compare_routes(frame,models,levels):
    scores = {level:score(frame,models[level].predict(frame)) for level in levels}
    return choose(scores,scores[levels[0]]["log_loss"])


def freeze(frame):
    dev = frame.loc[(frame.day<"2025-09") & frame.delay.eq(0)]
    models, selections, grids = {},{},[]
    for level in LEVELS:
        subset = eligible(dev,level)
        model, selected, results = fit_level(subset,level)
        models[level], selections[level] = model,selected
        grids.extend({"level":level,"candidate":key,**metrics} for key,metrics in results.items())
        print(level,len(subset),selected,flush=True)
    val = dev.loc[dev.day>="2025-01"]
    nwp = compare_routes(eligible(val,"latest"),models,["station","latest"])
    sat = compare_routes(eligible(val,"local_cloud"),models,[nwp,"local_cloud","upwind_cloud"])
    routing = {"nwp":nwp,"satellite":sat}
    controls = {}
    for a,b in zip(LEVELS,LEVELS[1:]):
        train = dev.loc[dev.day < "2025-01"]
        index = eligible(train,a).index.intersection(eligible(train,b).index)
        for level in (a,b):
            matched = eligible(train.loc[index],level)
            columns = predictors(matched,level)
            if a == "fixed":
                columns = [c for c in columns if c != "nwp_cloud"]
            logit = Family("logistic").fit(matched[columns],matched)
            controls[(a,b,level)] = Bundle(columns,{"logistic":logit},{"logistic":1.},1.)
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(dev,index=False).values.tobytes())
    for name in ("features.py","sources.py","models.py","analogue.py","study.py"):
        fingerprint.update((HERE/name).read_bytes())
    artifact = {"schema":SCHEMA,"version":fingerprint.hexdigest()[:12],"models":models,
                "control":controls,"routing":routing,"selected":selections,
                "reference":joblib.load(OUT/"reference.joblib"),
                "selection_end":"2025-08-31","training_end":"2024-12-31","timezone":TZ}
    ARTIFACT.parent.mkdir(exist_ok=True)
    temporary = ARTIFACT.with_suffix(".tmp")
    joblib.dump(artifact,temporary,compress=3)
    temporary.replace(ARTIFACT)
    pd.DataFrame(grids).to_csv(OUT/"validation.csv",index=False)
    save(OUT/"selection.json",{k:v for k,v in artifact.items() if k not in ("models","control","reference")})
    return artifact


def bootstrap(frame,p,q):
    y = frame.label.to_numpy()
    diff = (p.argmax(axis=1)==y).astype(float)-(q.argmax(axis=1)==y)
    means = pd.Series(diff,index=frame.day).groupby(level=0).mean().to_numpy()
    rng = np.random.default_rng(4260)
    samples = rng.choice(means,(2000,len(means)),replace=True).mean(axis=1)
    return np.quantile(samples,[.025,.975])


def assess(frame,artifact):
    metrics, pairs, predictions = [],[],[]
    for tag,start,end in [("prior_holdout","2025-09","2026-09"),("new_holdout","2026-09","2027")]:
        for lag in (0,15,30):
            held = frame.loc[frame.day.between(start,end,inclusive="left") & frame.delay.eq(lag)]
            if held.empty:
                continue
            for level in [*LEVELS,"deployed"]:
                subset = eligible(held,level)
                if subset.empty:
                    continue
                if level == "deployed":
                    routes = subset.apply(lambda row:select_tier(row,artifact),axis=1).to_numpy()
                    p = np.zeros((len(subset),5))
                    for tier in np.unique(routes):
                        mask = routes == tier
                        p[mask] = artifact["models"][tier].predict(subset.loc[mask])
                else:
                    p = artifact["models"][level].predict(subset)
                    routes = np.repeat(level,len(subset))
                windows = {"all":np.ones(len(subset),dtype=bool),"06-10":subset.hour<10,
                           "10-15":subset.hour.between(10,15,inclusive="left"),"15-20":subset.hour>=15}
                if level == "deployed":
                    distance = np.abs(subset.actual.to_numpy()[:,None]-BOUNDS).min(axis=1)
                    windows["within_2F"] = distance <= 2
                    windows["outside_2F"] = distance > 2
                    for clock in sorted(subset.hour.unique()):
                        windows[f"{int(clock):02}:{round(60*(clock%1)):02}"] = subset.hour.eq(clock)
                for window,mask in windows.items():
                    if np.any(mask):
                        metrics.append({"period":tag,"delay":lag,"level":level,"window":window,
                                        **score(subset.loc[mask],p[np.asarray(mask)])})
                preds = subset[["cutoff","day","label","actual"]].copy()
                preds["period"],preds["delay"],preds["level"],preds["route"] = tag,lag,level,routes
                preds[[f"p{i}" for i in range(5)]] = p
                predictions.append(preds)
            if lag:
                continue
            for a,b in zip(LEVELS,LEVELS[1:]):
                # Intersect before replacing the fresh HRRR columns for the old fixed cycle.
                ix = eligible(held,a).index.intersection(eligible(held,b).index)
                match = held.loc[ix]
                if match.empty:
                    continue
                for method in ["models","control"]:
                    before = a if method=="models" else (a,b,a)
                    after = b if method=="models" else (a,b,b)
                    first = artifact[method][before].predict(eligible(match,a))
                    second = artifact[method][after].predict(eligible(match,b))
                    lower,upper = bootstrap(match,second,first)
                    m1,m2 = score(match,first),score(match,second)
                    pairs.append({"period":tag,"estimator":method,"before":a,"after":b,"days":m1["days"],
                        "rows":m1["rows"],"accuracy_before":m1["accuracy"],"accuracy_after":m2["accuracy"],
                        "accuracy_delta":m2["accuracy"]-m1["accuracy"],"ci_low":lower,"ci_high":upper,
                        "loss_before":m1["log_loss"],"loss_after":m2["log_loss"]})
    pd.DataFrame(metrics).to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"ablation.csv",index=False)
    pd.concat(predictions).to_parquet(OUT/"predictions.parquet",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-samples",action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--freeze-only",action="store_true")
    phase.add_argument("--evaluate-frozen",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    frame=pd.read_parquet(OUT/"sampled.parquet") if args.reuse_samples else build(args.freeze_only)
    with threadpool_limits(limits=2):
        artifact=joblib.load(ARTIFACT) if args.evaluate_frozen else freeze(frame)
        if not args.freeze_only:
            assess(frame,artifact)
    print("Finished",artifact["version"],flush=True)

