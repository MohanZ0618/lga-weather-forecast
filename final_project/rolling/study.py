"""Fit and evaluate chronological fixed-cutoff forecasts. Run from weather-project."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data import DATA, HOURS, OUT, STORE, SNAPSHOT_END
from experiment import BOUNDS, FORECAST_FEATURES, PROBS, metrics, season_features
from rolling_models import BASE, GROUPS, NWP, Candidate, available, controls, predict_route, search_grid

FOLDS = [("2023-01-01","2024-01-01"),("2024-01-01","2025-01-01"),("2025-01-01","2025-09-01")]
FREEZE = "2025-09-01"
TIE = .001


def splits(frame):
    for start,end in FOLDS:
        yield start[:4],frame.loc[frame.index < start],frame.loc[(frame.index >= start)&(frame.index < end)]


def inputs(hour, delay=0, external=True):
    ground = pd.read_parquet(STORE/"stations.parquet",filters=[("hour","==",hour),("delay","==",delay)])
    data = season_features(ground.set_index("date").sort_index())
    return add_forecasts(data,hour,external)


def add_forecasts(data,hour,external=True):
    fixed = pd.concat([pd.read_csv(path,parse_dates=["date"]) for path in
                      (DATA/"hrrr_features.csv.gz",STORE/"autumn_fixed.csv")],ignore_index=True)
    fixed = fixed.loc[fixed.hour.eq(hour)].set_index("date")[FORECAST_FEATURES]
    data = data.join(fixed,validate="one_to_one")
    if external:
        ext = pd.read_parquet(STORE/"external.parquet")
        ext["date"] = pd.to_datetime(ext.date)
        data = data.join(ext.loc[ext.hour.eq(hour)].drop(columns=["hour","errors"]).set_index("date"),
                         validate="one_to_one")
    return data


def spec(group,tier,year,configs):
    return dict(group=group,tier=tier,year=year,configs=configs)


def complexity(item):
    return ({"station":1,"fixed":2,"latest":2,"cloud":2,"sat":3,"fusion":3}[item["tier"]],
            len(item["configs"]),len(GROUPS[item["group"]])+len(NWP[item["tier"]]),item["loss"])


def choose(options):
    lowest = min(o["loss"] for o in options)
    return min((o for o in options if o["loss"] <= lowest+TIE),key=complexity)


def eligible(frame,s):
    return frame.loc[(frame.index.year>=s["year"]) & available(frame,s["tier"])]


def fit_members(frame,s):
    train = eligible(frame,s)
    return [Candidate(c,s["group"],s["tier"]).fit(train) for c in s["configs"]]


def probs(members,frame):
    return np.mean([m.predict(frame) for m in members],axis=0)


def cross_validate(frame,s,fallback=None,match=None):
    predictions,records = [],[]
    for fold,train,val in splits(frame):
        if match:
            train = train.loc[available(train,match)]
            val = val.loc[available(val,match)]
        train = eligible(train,s)
        # All configurations see the same evaluation dates, with a shared station fallback.
        val = val.loc[val.index.year>=2021]
        has = available(val,s["tier"])
        p = pd.DataFrame(index=val.index,columns=PROBS,dtype=float)
        start = time.perf_counter()
        members = fit_members(train,s)
        if has.any():
            p.loc[has] = probs(members,val.loc[has])
        if fallback is not None:
            p.loc[~has] = fallback.loc[val.index[~has],PROBS]
        elapsed = time.perf_counter()-start
        valid = p.notna().all(axis=1)
        p = p.loc[valid]
        scores = metrics(val.loc[p.index,"target_category"].to_numpy(),p.to_numpy())
        records.append(dict(fold=fold,train_n=len(train),train_start=str(train.index.min().date()),
                            train_end=str(train.index.max().date()),source_days=int(has.sum()),seconds=elapsed,
                            **scores))
        predictions.append(p.assign(fold=fold))
    oof = pd.concat(predictions)
    result = dict(**s,loss=float(np.mean([r["log_loss"] for r in records])))
    return result,oof,records


def grid_search(frame,template,fallback,folder):
    folder.mkdir(parents=True,exist_ok=True)
    choices,all_probs,records = [],{},[]
    for config in search_grid():
        s = {**template,"configs":[config]}
        option,p,folds = cross_validate(frame,s,fallback)
        choices.append(option)
        all_probs[config] = p
        records.extend(dict(**r,config=config) for r in folds)
    # One champion per family; only these two family champions may be averaged.
    champions = {}
    for option in sorted(choices,key=lambda o:o["loss"]):
        family = json.loads(option["configs"][0])[0]
        champions.setdefault(family,option)
    pair = [o["configs"][0] for o in list(champions.values())[:2]]
    merged = (all_probs[pair[0]][PROBS]+all_probs[pair[1]][PROBS])/2
    merged["fold"] = all_probs[pair[0]].fold
    ensemble_losses = []
    for fold,p in merged.groupby("fold"):
        m = metrics(frame.loc[p.index,"target_category"].to_numpy(),p[PROBS].to_numpy())
        ensemble_losses.append(m["log_loss"])
        group = [r for r in records if r["fold"]==fold and r["config"] in pair]
        records.append({**group[0],**m,"seconds":sum(r["seconds"] for r in group),"config":json.dumps(pair)})
    ensemble = {**template,"configs":pair,"loss":float(np.mean(ensemble_losses))}
    choices.append(ensemble)
    best = choose(choices)
    pbest = merged if len(best["configs"])==2 else all_probs[best["configs"][0]]
    pd.DataFrame(records).assign(tier=template["tier"],group=template["group"],year=template["year"],
                                 seed=4260,features=json.dumps(BASE+GROUPS[template["group"]]+NWP[template["tier"]])
                                 ).to_csv(folder/"grid.csv",index=False)
    (folder/"ranking.json").write_text(json.dumps(sorted(choices,key=lambda o:o["loss"]),indent=2))
    (folder/"selected.json").write_text(json.dumps(best,indent=2))
    pbest.to_parquet(folder/"selected_oof.parquet")
    return best,pbest


def station_search(hour):
    frame = inputs(hour,external=False)
    folder = OUT/f"{hour:02}"
    folder.mkdir(parents=True,exist_ok=True)
    controls_log,options = [],[]
    for year in (2018,2021):
        for group in GROUPS:
            item = spec(group,"station",year,controls(hour))
            option,_,records = cross_validate(frame,item)
            controls_log.extend(dict(**r,year=year,group=group) for r in records)
            options.append(option)
    pd.DataFrame(controls_log).to_csv(folder/"airport_ablation.csv",index=False)
    best = choose(options)
    grid_search(frame,{k:v for k,v in best.items() if k!="loss"},None,folder/"station")
    print(f"{hour:02} station search complete",flush=True)


def fixed_search(hour):
    folder = OUT/f"{hour:02}"
    station = json.loads((folder/"station/selected.json").read_text())
    fallback = pd.read_parquet(folder/"station/selected_oof.parquet")
    grid_search(inputs(hour,external=False),spec(station["group"],"fixed",2021,[]),fallback,folder/"fixed")
    print(f"{hour:02} fixed cycle search complete",flush=True)


def replay_stats(frame,p):
    target = frame.target_category.to_numpy()
    hit = p.argmax(axis=1)==target
    loss = -np.log(p[np.arange(len(target)),target].clip(1e-15))
    return hit,loss


def compare_week(frame,proposed,reference):
    h1,l1 = replay_stats(frame,proposed)
    h0,l0 = replay_stats(frame,reference)
    delta = pd.DataFrame(dict(net=h1.astype(int)-h0.astype(int),loss=l1-l0,n=1),index=frame.index)
    weekly = delta.groupby(pd.Grouper(freq="7D")).sum().to_numpy()
    rng = np.random.default_rng(4260)
    draws = weekly[rng.integers(len(weekly),size=(5000,len(weekly)))].sum(axis=1)
    draws = draws[draws[:,2]>0]
    intervals = np.quantile(draws[:,:2]/draws[:,2,None],[.025,.975],axis=0)
    return dict(net_correct=int(delta.net.sum()),delta_loss=delta.loss.mean(),acc_low=intervals[0,0],
                acc_high=intervals[1,0],loss_low=intervals[0,1],loss_high=intervals[1,1])


def source_controls(frame,group,hour,fallback,folder):
    options,records = [],[]
    for tier in ("fixed","latest","cloud","sat","fusion"):
        item = spec(group,tier,2021,controls(hour))
        option,_,folds = cross_validate(frame,item,fallback)
        options.append(option)
        records.extend(dict(**r,tier=tier,cohort="all dates",pair="") for r in folds)
    # Match both training and evaluation availability on each side of each comparison.
    for a,b in [("station","fixed"),("fixed","latest"),("latest","cloud"),("latest","sat"),("cloud","fusion")]:
        common = frame.loc[available(frame,a)&available(frame,b)]
        for tier in (a,b):
            _,_,folds = cross_validate(common,spec(group,tier,2021,controls(hour)))
            records.extend(dict(**r,tier=tier,cohort="matched",pair=f"{a}/{b}") for r in folds)
    pd.DataFrame(records).to_csv(folder/"source_ablation.csv",index=False)
    return options


def heldout_controls(frame,hour,group,folder):
    """Post-selection paired replays with the original hour-specific ensemble fixed."""
    configs = controls(hour)
    train = frame.loc[frame.index < FREEZE]
    test = frame.loc[frame.index >= FREEZE]
    station = spec("base","station",2018,configs)
    station_models = fit_members(train,station)
    pairs = [("training years",spec("base","station",2021,configs),station)]
    pairs += [(g,station,spec(g,"station",2018,configs)) for g in ("thermo","sky","both")]
    for a,b in [("station","fixed"),("fixed","latest"),("latest","cloud"),("latest","sat"),("cloud","fusion")]:
        pairs.append((f"{a}/{b}",spec(group,a,2021,configs),spec(group,b,2021,configs)))
    records = []
    for label,left,right in pairs:
        for cohort in ("all","matched"):
            both = available(test,left["tier"]) & available(test,right["tier"])
            matched_train = available(train,left["tier"]) & available(train,right["tier"])
            history = train.loc[matched_train] if cohort=="matched" else train
            cases = test.loc[both] if cohort=="matched" else test
            predictions = []
            for item in (left,right):
                bundle = dict(spec=item,members=fit_members(history,item),fallback=station_models)
                predictions.append(predict_route(cases,bundle)[0])
            for period,mask in [("historical",cases.index<"2026-09-01"),("autumn",cases.index>="2026-09-01")]:
                if not mask.any():
                    continue
                base,new = [p[mask] for p in predictions]
                y = cases.target_category[mask].to_numpy()
                records.append(dict(pair=label,cohort=cohort,period=period,n=len(y),
                    left_train_n=len(eligible(history,left)),right_train_n=len(eligible(history,right)),
                    right_coverage=available(cases.loc[mask],right["tier"]).mean(),
                    **{f"left_{k}":v for k,v in metrics(y,base).items() if k!="n"},
                    **{f"right_{k}":v for k,v in metrics(y,new).items() if k!="n"},
                    **compare_week(cases.loc[mask],new,base)))
    pd.DataFrame(records).to_csv(folder/"paired_ablation_replay.csv",index=False)


def run_hour(hour):
    folder = OUT/f"{hour:02}"
    station = json.loads((folder/"station/selected.json").read_text())
    fallback = pd.read_parquet(folder/"station/selected_oof.parquet")
    frame = inputs(hour)
    options = source_controls(frame,station["group"],hour,fallback,folder)
    advanced = choose([o for o in options if o["tier"]!="fixed"])
    tier = advanced["tier"]
    winner,_ = grid_search(frame,spec(station["group"],tier,2021,[]),fallback,folder/tier)
    finalists = [station,json.loads((folder/"fixed/selected.json").read_text()),winner]
    chosen = choose(finalists)
    train = frame.loc[frame.index < FREEZE]
    bundle = dict(schema=1,spec=chosen,members=fit_members(train,chosen),fallback=fit_members(train,station),
                  station_spec=station,trained_through="2025-08-31",frozen=pd.Timestamp.now(tz="UTC").isoformat())
    (folder/"selection.json").write_text(json.dumps(dict(selected=chosen,finalists=finalists,tie=TIE,
        frozen=bundle["frozen"]),indent=2))
    # Compare against the same historical ensemble, also refitted to the same end date.
    original = spec("base","fixed",2021,controls(hour))
    old_station = spec("base","station",2018,controls(hour))
    ref = dict(spec=original,members=fit_members(train,original),fallback=fit_members(train,old_station))
    bundle["reference"] = ref
    joblib.dump(bundle,folder/"model.joblib",compress=3)
    rows,summary,reliability = [],[],[]
    for delay in (0,15,30):
        replay = inputs(hour,delay) if delay else frame
        for period,start,end in [("historical",FREEZE,"2026-09-01"),
                                 ("autumn","2026-09-01",pd.Timestamp(SNAPSHOT_END)+pd.Timedelta(days=1))]:
            test = replay.loc[(replay.index>=start)&(replay.index<end)]
            p,has = predict_route(test,bundle)
            baseline,old_has = predict_route(test,ref)
            near = np.abs(test.target_max.to_numpy()[:,None]-BOUNDS).min(axis=1)<=2
            for name,mask in [("all",np.ones(len(test),bool)),("matched",has & old_has),("near_boundary",near)]:
                if np.sum(mask)==0:
                    continue
                for model,values in [("candidate",p),("reference",baseline)]:
                    m = metrics(test.target_category[mask].to_numpy(),values[mask])
                    summary.append(dict(period=period,delay=delay,subset=name,model=model,
                        coverage=float((has if model=="candidate" else old_has)[mask].mean()),**m,
                        **(compare_week(test.loc[mask],p[mask],baseline[mask]) if model=="candidate" else {})))
            for model,values,source in [("candidate",p,has),("reference",baseline,old_has)]:
                record = pd.DataFrame(values,index=test.index,columns=PROBS)
                rows.append(record.assign(period=period,delay=delay,model=model,source_available=source,
                                           actual=test.target_category,target_max=test.target_max))
                if delay==0:
                    hit,_ = replay_stats(test,values)
                    top = values.max(axis=1)
                    for low in np.arange(10)/10:
                        mask = (top>=low)&((top<round(low+.1,1)) if low<.9 else (top<=1.0000001))
                        if mask.any():
                            reliability.append(dict(period=period,model=model,bin_low=low,n=int(mask.sum()),
                                                    confidence=top[mask].mean(),accuracy=hit[mask].mean()))
    pd.concat(rows).to_csv(folder/"replay.csv.gz")
    pd.DataFrame(summary).to_csv(folder/"metrics.csv",index=False)
    pd.DataFrame(reliability).to_csv(folder/"reliability.csv",index=False)
    heldout_controls(frame,hour,station["group"],folder)
    print(hour,chosen,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"stations":station_search,"fixed":fixed_search,"search":run_hour}
    parser.add_argument("stage",choices=stages)
    parser.add_argument("--hour",type=int,choices=HOURS)
    args = parser.parse_args()
    with threadpool_limits(2):
        for hour in [args.hour] if args.hour else HOURS:
            stages[args.stage](hour)
