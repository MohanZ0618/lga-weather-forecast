"""Retain the frozen 09:00 family; select small probability adjustments before a later replay."""
import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from estimators import ADDITIONS, BASE, band_probs, present
from morning import labels, tail_reference
from morning_data import SNAPSHOT, TAIL, market_layouts, settlements
from panels import FREEZE, OUT, TABLE, TZ, contracts, ground
from replay import interval
from market_study import DEGREES, frame_for
from late_replay import station_archive
from train import write_json

DEST = OUT/"old09"
BUNDLE = OUT/"09"/"frozen.joblib"
LOCK = "2026-03-01"
MIDDLE = "2025-12-01"
END = "2026-11-22"


@dataclass(frozen=True)
class Adjustment:
    threshold_weight: float = .5
    bias_f: float = 0.
    digit_power: float = 0.
    early_floor: bool = False


DEFAULT = Adjustment()


def old_inputs(start,end):
    columns = BASE+ADDITIONS["nbm"]+["LGA_max_so_far","routine_floor_9"]
    f = ground(9)[columns]
    if pd.Timestamp(end)>TAIL:
        cache = DEST/"tail_inputs.parquet"
        if cache.exists():
            tail = pd.read_parquet(cache)
        else:
            tail = frame_for(9,station_archive(cached_only=True),labels()).loc[TAIL:]
            nbm = pd.read_parquet(TABLE).query("hour==9")
            nbm = nbm.set_index(pd.to_datetime(nbm.date))
            tail = tail.join(nbm[["nbm_high","nbm_peak_hours","nbm_spread","nbm_cloud"]])
            tail["nbm_disagreement"] = tail.nbm_high-tail.latest_high
            tail = tail[columns]
            tail.to_parquet(cache)
        f = pd.concat([f.loc[f.index<TAIL],tail])
    return f.loc[(f.index>=start)&(f.index<end)]


def digit_weights(y,kind):
    # Fit the frequency of each 9°F remainder on a fixed pre-model window only.
    ok = (y.index>=pd.Timestamp(FREEZE)-pd.Timedelta(days=737)) & (y.index<pd.Timestamp(FREEZE)-pd.Timedelta(days=7))
    values = y.loc[ok & y.hours.ge(20) & y.blocks.eq(4),kind].dropna().to_numpy(int)
    if not len(values):
        raise ValueError("missing historical temperatures for rounding adjustment")
    counts = np.bincount(values%9,minlength=9)+20.
    return counts/counts.mean()


def component_inputs(start,end,raw):
    layouts = market_layouts(raw.drop(columns="winner"))
    panel = old_inputs(start,end)
    dates = panel.index.intersection(pd.DatetimeIndex(layouts)).sort_values()
    if dates.empty:
        raise ValueError("No eligible archived market/weather panel for this interval")
    f = panel.loc[dates]
    morning = pd.read_parquet(SNAPSHOT,columns=["LGA_early_high","LGA_observed"]).loc[dates]
    issue = pd.Series((dates+pd.Timedelta(hours=9)).tz_localize(TZ),index=dates)
    if (morning.LGA_observed>issue-pd.Timedelta(minutes=5)).any():
        raise ValueError("late reading for supplementary all-report floor")
    bundle = joblib.load(BUNDLE)
    if bundle["train_before"]!=FREEZE or dates.min()<pd.Timestamp(FREEZE):
        raise ValueError("model would predate its training cutoff")
    cuts = np.broadcast_to(DEGREES[1:]-.5,(len(dates),len(DEGREES)-1))
    first = np.empty((len(dates),len(DEGREES)))
    second = np.empty_like(first)
    floor = np.empty(len(dates))
    richer_floor = np.empty_like(floor)
    digit = np.empty_like(first)
    obs = labels()
    for kind in ("all_f","routine_f"):
        use = np.array([layouts[d]["regime"]=="routine_metar" for d in dates])==(kind=="routine_f")
        if not use.any():
            continue
        sub = f.loc[use]
        known = (sub.routine_floor_9 if kind=="routine_f" else np.floor(sub.LGA_max_so_far+.5)).to_numpy()
        sub = sub.assign(floor_f=known)
        floor[use] = known
        richer_floor[use] = (known if kind=="routine_f" else np.fmax(known,
                             np.floor(morning.loc[sub.index,"LGA_early_high"].to_numpy()+.5)))
        digit[use] = digit_weights(obs,kind)[DEGREES%9]
        routed = bundle["models"][kind]
        tiers = np.full(len(sub),"station",dtype=object)
        for tier in routed.models:
            tiers[present(sub,tier)] = tier
        for tier,model in routed.models.items():
            which = tiers==tier
            if not which.any():
                continue
            ix = np.flatnonzero(use)[which]
            members = model.members if model.family=="blend" else [model,model]
            for dest,member in zip((first,second),members):
                dest[ix] = band_probs(member.survival(sub.loc[which],cuts[ix]))
    return dates,layouts,(first,second,floor,richer_floor,digit)


def adjust(inputs,setting):
    first,second,floor,richer_floor,digit = inputs
    q = setting.threshold_weight*first+(1-setting.threshold_weight)*second
    if setting.bias_f:
        shifted = np.zeros_like(q)
        if setting.bias_f>0:
            shifted[:,1:] = q[:,:-1]
            shifted[:,-1] += q[:,-1]
        else:
            shifted[:,:-1] = q[:,1:]
            shifted[:,0] += q[:,0]
        amount = abs(setting.bias_f)
        q = (1-amount)*q+amount*shifted
    if setting.digit_power:
        q *= digit**setting.digit_power
    limit = richer_floor if setting.early_floor else floor
    indices = np.clip(np.nan_to_num(limit,nan=DEGREES[0])-DEGREES[0],0,len(DEGREES)-1).astype(int)
    below = np.arange(len(DEGREES))[None,:]<indices[:,None]
    mass = np.where(below,q,0).sum(axis=1)
    q[below] = 0.
    q[np.arange(len(q)),indices] += mass
    return q/q.sum(axis=1,keepdims=True)


def bins_for(data,setting):
    dates,layouts,inputs = data
    mass = adjust(inputs,setting)
    return {d:.999*(layouts[d]["bins"]@p)+.001/len(layouts[d]["ids"]) for d,p in zip(dates,mass)}


def scoring(dates,layouts,prob,truth):
    rows = []
    for d in dates:
        p = prob[d]
        winner = layouts[d]["ids"].index(truth.loc[d,"market_id"])
        top = int(p.argmax())
        rows.append(dict(date=d,hit=int(top==winner),loss=-np.log(p[winner]),pick=layouts[d]["brackets"][top]))
    return pd.DataFrame(rows).set_index("date")


def validation_truth(raw,lock):
    data = settlements(raw.loc[raw.date<pd.Timestamp(lock)])
    return data.loc[data.closed+pd.Timedelta(days=2)<pd.Timestamp(lock,tz=TZ)]


def base_reference():
    base = pd.read_csv(OUT/"bracket_probabilities.csv.gz",parse_dates=["date"],dtype={"market_id":str})
    base = base.loc[base.hour.eq(9)&base.delay.eq(0),["date","market_id","p"]]
    return pd.concat([base,tail_reference()]).set_index(["date","market_id"]).p


def check_reference(data,ref):
    dates,layouts,_ = data
    matching = dates.intersection(ref.index.get_level_values("date").unique())
    defaults = bins_for(data,DEFAULT)
    for d in matching:
        saved = ref.loc[d].reindex(layouts[d]["ids"])
        np.testing.assert_allclose(defaults[d],saved,atol=1e-8,rtol=0)
    return matching


def select():
    raw = contracts()
    early = raw.loc[raw.date<LOCK]
    data = component_inputs(FREEZE,LOCK,early)
    truth = validation_truth(early,LOCK)
    dates = data[0].intersection(truth.index)
    rows = []
    for w,b,h,f in product((.25,.5,.75),(-.5,0.,.5),(0.,.5),(False,True)):
        setting = Adjustment(w,b,h,f)
        scores = scoring(dates,data[1],bins_for(data,setting),truth)
        before = scores.index<MIDDLE
        rows.append(dict(**asdict(setting),days=len(scores),autumn_hits=int(scores.hit[before].sum()),
                         winter_hits=int(scores.hit[~before].sum()),hits=int(scores.hit.sum()),loss=scores.loss.mean(),
                         changes=sum(v!=asdict(DEFAULT)[k] for k,v in asdict(setting).items())))
    grid = pd.DataFrame(rows)
    base = grid.loc[grid.changes.eq(0)].iloc[0]
    safe = grid.loc[(grid.autumn_hits>=base.autumn_hits)&(grid.winter_hits>=base.winter_hits)&
                    (grid.loss<=base.loss+.02)]
    best = safe.sort_values(["hits","loss","changes"],ascending=[False,True,True],kind="stable").iloc[0]
    spec = Adjustment(float(best.threshold_weight),float(best.bias_f),float(best.digit_power),bool(best.early_floor))
    grid.to_csv(DEST/"validation.csv",index=False)
    write_json(DEST/"candidate.json",dict(setting=asdict(spec),locked_before=LOCK,days=len(dates),
        latest_settlement_used=truth.loc[dates,"closed"].max().isoformat(),bundle_sha256=sha256(BUNDLE.read_bytes()).hexdigest()))
    print("Locked",spec,len(dates),"dates",flush=True)


def evaluate():
    cfg = json.loads((DEST/"candidate.json").read_text())
    if cfg["locked_before"]!=LOCK or cfg["bundle_sha256"]!=sha256(BUNDLE.read_bytes()).hexdigest():
        raise ValueError("candidate lock does not match the base artifact")
    raw = contracts()
    data = component_inputs(LOCK,END,raw)
    matched = check_reference(data,base_reference())
    left = bins_for(data,DEFAULT)
    right = bins_for(data,Adjustment(**cfg["setting"]))
    forecasts = [dict(date=d,market_id=i,bracket=name,base=p,candidate=q)
                 for d in matched for i,name,p,q in zip(data[1][d]["ids"],data[1][d]["brackets"],left[d],right[d])]
    pd.DataFrame(forecasts).to_csv(DEST/"probabilities.csv.gz",index=False)
    truth = settlements(raw)
    daily = scoring(matched,data[1],left,truth).add_prefix("base_").join(
        scoring(matched,data[1],right,truth).add_prefix("candidate_"))
    daily["period"] = np.select([daily.index<"2026-09-01",daily.index<TAIL],["spring_summer","autumn"],default="tail")
    daily.to_csv(DEST/"daily.csv")
    rows = []
    for period,g in [*daily.groupby("period",sort=False),("all",daily)]:
        change = g.candidate_hit-g.base_hit
        low,high = interval(change)
        rows.append(dict(period=period,days=len(g),base_hits=int(g.base_hit.sum()),candidate_hits=int(g.candidate_hit.sum()),
            fixed=int((change>0).sum()),lost=int((change<0).sum()),gain=change.mean(),low=low,high=high,
            base_loss=g.base_loss.mean(),candidate_loss=g.candidate_loss.mean()))
    summary = pd.DataFrame(rows)
    summary.to_csv(DEST/"metrics.csv",index=False)
    # The user selected the previous model. Calibration candidates stay separately identified.
    write_json(DEST/"active.json",dict(model="../09/frozen.joblib",sha256=cfg["bundle_sha256"],
        hour=9,family="threshold + hurdle",setting=asdict(DEFAULT)))
    print(summary.to_string(index=False))


def forecast(day):
    d = pd.Timestamp(day)
    active = json.loads((DEST/"active.json").read_text())
    if active["sha256"]!=sha256(BUNDLE.read_bytes()).hexdigest():
        raise ValueError("selected model artifact has changed")
    data = component_inputs(d,d+pd.Timedelta(days=1),contracts())
    p = bins_for(data,Adjustment(**active["setting"]))[d]
    print(pd.DataFrame(dict(bracket=data[1][d]["brackets"],p=p)).sort_values("p",ascending=False).to_string(index=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["select","evaluate","forecast"])
    parser.add_argument("--date")
    args = parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(2):
        if args.stage=="forecast":
            if args.date is None:
                parser.error("forecast requires --date")
            forecast(args.date)
        elif args.stage=="select":
            select()
        else:
            evaluate()
