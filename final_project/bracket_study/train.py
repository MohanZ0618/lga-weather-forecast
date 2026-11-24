"""Select families on winter/spring settlements, tune on later settlements, and freeze."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from estimators import DEFAULTS, FAMILIES, SEED, TIERS, Distribution, Routed, defaults, present
from panels import FREEZE, OUT, SCREEN, TUNE, TZ, Brackets, events, ground, history, target


def write_json(path,obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
    tmp.replace(path)


@lru_cache(maxsize=48)
def fold(hour,start,end):
    frame = target(ground(hour,development=True),hour,"all_f")
    all_events = events(hour)
    ev = all_events.loc[(all_events.index>=start)&(all_events.index<end)&
                        (all_events.closed<pd.Timestamp(end,tz=TZ))&all_events.index.isin(frame.index)]
    if ev.empty or ev.regime.eq("routine_metar").any():
        raise ValueError("Expected historical all-report proxy during model selection")
    train = history(frame,start,"all_f")
    fb = {tier:Distribution("threshold",tier,defaults("threshold")).fit(train.loc[present(train,tier)])
          for tier in TIERS[:2]}
    return train,frame.loc[ev.index],Brackets(ev),fb


def trials(family):
    base = defaults(family)
    yield base
    if family=="threshold":
        for c in np.geomspace(.01,100,33):
            if not np.isclose(c,base["C"]):
                yield dict(base,C=float(c))
    else:
        rng = np.random.default_rng(SEED)
        for _ in range(64):
            p = dict(base,depth=int(rng.integers(2,7)),trees=int(rng.integers(150,1001)),
                     rate=float(np.exp(rng.uniform(np.log(.015),np.log(.12)))),
                     l2=float(np.exp(rng.uniform(0,np.log(30)))))
            if family in ("hurdle","blend"):
                p["C"] = float(10**rng.uniform(-2,2))
            yield p


def evaluate(hour,family,tier,params,splits):
    rows = []
    for start,end in splits:
        train,case,brackets,fb = fold(hour,start,end)
        tick = time.perf_counter()
        model = Routed(family,tier,params).fit(train,fb)
        out,route = model.predict(case,brackets.cuts)
        scores = brackets.score(brackets.unpack(out))
        rows.append(dict(hour=hour,family=family,tier=tier,params=json.dumps(params,sort_keys=True),
            start=start,end=end,train_end=str(train.index.max().date()),train_days=len(train),
            rich_train_days=int(present(train,tier).sum()),days=len(case),rich_days=int((route==tier).sum()),
            log_loss=scores.log_loss.mean(),accuracy=scores.correct.mean(),brier=scores.brier.mean(),
            seconds=time.perf_counter()-tick))
    return rows


def select_hour(hour,stage,workers):
    folder = OUT/f"{hour:02}"
    folder.mkdir(parents=True,exist_ok=True)
    screen = stage=="screen"
    if screen:
        settings = [(f,t,defaults(f)) for t in TIERS for f in FAMILIES]
        splits = SCREEN
    else:
        lock = json.loads((folder/"family.json").read_text())
        f,t = lock["family"],lock["tier"]
        settings = [(f,t,p) for p in trials(f)]
        splits = TUNE
    for start,end in splits:
        fold(hour,start,end)
    def run(item):
        f,t,p = item
        return evaluate(hour,f,t,p,splits)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(run,settings))
    rows = pd.DataFrame([r for group in groups for r in group])
    rows.to_csv(folder/f"{stage}.csv",index=False)
    ranking = rows.groupby(["family","tier","params"],as_index=False,sort=False).agg(
        loss=("log_loss","mean"),accuracy=("accuracy","mean"),days=("days","sum"))
    best = ranking.loss.min()
    if screen:
        eligible = ranking.loc[ranking.loss<=best+.001].copy()
        eligible["sources"] = eligible.tier.map(TIERS.index)
        eligible["complexity"] = eligible.family.map(FAMILIES.index)
        win = eligible.sort_values(["sources","complexity","loss"]).iloc[0]
    else:
        win = ranking.loc[ranking.loss.idxmin()]
    spec = dict(family=win.family,tier=win.tier,params=json.loads(win.params),
                validation_log_loss=float(win.loss),days=int(win.days),end=splits[-1][1])
    write_json(folder/("family.json" if screen else "parameters.json"),spec)
    print(f"{stage} {hour:02}: {spec}",flush=True)


def refit(hour):
    folder = OUT/f"{hour:02}"
    spec = json.loads((folder/"parameters.json").read_text())
    bundle = dict(hour=hour,train_before=FREEZE,spec=spec,models={})
    for kind in ("all_f","routine_f","course"):
        frame = target(ground(hour,development=True),hour,kind)
        train = history(frame,FREEZE,kind)
        bundle["models"][kind] = Routed(spec["family"],spec["tier"],spec["params"]).fit(train)
    joblib.dump(bundle,folder/"frozen.joblib",compress=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["screen","tune","freeze","all"])
    parser.add_argument("--hour",type=int,choices=[9,12,15])
    parser.add_argument("--workers",type=int,default=3)
    args = parser.parse_args()
    for stage in (("screen","tune","freeze") if args.stage=="all" else (args.stage,)):
        for hour in ([args.hour] if args.hour else [9,12,15]):
            if stage=="freeze":
                refit(hour)
            else:
                select_hour(hour,stage,args.workers)


if __name__=="__main__":
    with threadpool_limits(1,user_api="blas"):
        main()
