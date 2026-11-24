"""Paired retrospective evaluation of the locked models and controls on identical dates."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from estimators import Routed, defaults, present
from panels import (OUT,ROOT,FREEZE,MARKET,PERIODS,TZ,Brackets,climate,contracts,
                    delayed_routine_floor,events,ground,history,listing_only,season_weights,target)
from experiment import BOUNDS,PROBS,metrics,stabilize
from market_study import combined
from study import OUT as ROLLING


def interval(delta):
    week = delta.to_frame("delta").assign(n=1).resample("7D").sum().to_numpy()
    rng = np.random.default_rng(4260)
    draws = week[rng.integers(len(week),size=(5000,len(week)))].sum(axis=1)
    valid = draws[:,1]>0
    return np.quantile(draws[valid,0]/draws[valid,1],[.025,.975])


def plot_scores(summary):
    from matplotlib import pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(9,4),sharey=True,layout="constrained")
    styles = [("season_only","Seasonal climatology","#9c7045","--"),
              ("listings_only","Listed brackets only","#74787d","--"),
              ("incumbent","Previous market model","#557cab","-"),
              ("new","New frozen model","#007963","-")]
    for axis,(period,title) in zip(axes,[("historical","Sep 2025 – Aug 2026"),("autumn","Sep – Nov 2026")]):
        panel = summary.loc[(summary.period==period)&summary.delay.eq(0)].set_index("model")
        for key,label,color,style in styles:
            rows = panel.loc[[key]].sort_values("hour")
            axis.plot(rows.hour,rows.log_loss,color=color,ls=style,marker="o",lw=2,label=label)
        axis.set(xticks=[9,12,15],xticklabels=["09:00","12:00","15:00"],title=title,
                 xlabel="Issue time in New York")
        axis.grid(axis="y",alpha=.2)
        axis.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Market bracket log loss (lower is better)")
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/"market_log_loss.png",dpi=170)
    plt.close(fig)


def model_probabilities(models,frame,hour,brackets):
    routine = brackets.events.regime.eq("routine_metar").to_numpy()
    raw = np.empty((len(frame),brackets.cuts.shape[1]+1))
    routes = np.empty(len(frame),dtype=object)
    for kind,mask in (("routine_f",routine),("all_f",~routine)):
        if mask.any():
            x = target(frame.loc[mask],hour,kind)
            raw[mask],routes[mask] = models[kind].predict(x,brackets.cuts[mask])
    return brackets.unpack(raw),routes


def controls(hour,spec):
    result = {}
    configs = {"logistic_hrrr":("threshold","hrrr"),"logistic_nbm":("threshold","nbm"),
               "structure_control":("threshold",spec["tier"]),"untuned":(spec["family"],spec["tier"])}
    cache = {}
    for name,(family,tier) in configs.items():
        if (family,tier) not in cache:
            fits = {}
            for kind in ("all_f","routine_f"):
                train = history(target(ground(hour),hour,kind),FREEZE,kind)
                fits[kind] = Routed(family,tier,defaults(family)).fit(train)
            cache[family,tier] = fits
        result[name] = cache[family,tier]
    # This control holds both training and evaluation dates fixed when isolating NBM.
    fits = {}
    for kind in ("all_f","routine_f"):
        train = history(target(ground(hour),hour,kind),FREEZE,kind)
        fits[kind] = Routed("threshold","hrrr",defaults("threshold")).fit(train.loc[present(train,"nbm")])
    result["logistic_hrrr_matched"] = fits
    return result


def evaluate_hour(hour):
    folder = OUT/f"{hour:02}"
    bundle = joblib.load(folder/"frozen.joblib")
    extra = controls(hour,bundle["spec"])
    archived = pd.read_csv(MARKET/"market_predictions.csv.gz",parse_dates=["date"])
    archived = archived.loc[archived.hour.eq(hour)].set_index("date")
    old = joblib.load(MARKET/f"one_degree_{hour:02}.joblib")
    old_course = pd.read_csv(ROLLING/f"{hour:02}"/"replay.csv.gz",parse_dates=["date"])
    all_events = events(hour)
    past = all_events.loc[(all_events.index<FREEZE)&(all_events.closed<pd.Timestamp(FREEZE,tz=TZ))]
    tickers = contracts().set_index(["date","bracket"]).market_id
    daily,bins,course,paired = [],[],[],[]
    course_train = history(target(ground(hour),hour,"course"),FREEZE,"course")
    counts = pd.crosstab(course_train.index.month,course_train.target_category).reindex(
        index=range(1,13),columns=range(5),fill_value=0).fillna(0).to_numpy()+1
    for delay in (0,15,30):
        weather = ground(hour,delay)
        for period,start,end in PERIODS:
            panel = weather.loc[(weather.index>=start)&(weather.index<end)]
            c = target(panel,hour,"course")
            probs,_ = bundle["models"]["course"].predict(c,np.tile(BOUNDS,(len(c),1)))
            priors = counts[c.index.month-1]
            probs = stabilize(probs,priors/priors.sum(axis=1,keepdims=True),c.floor_f.to_numpy())
            archive = old_course.loc[old_course.delay.eq(delay)&old_course.model.eq("candidate")].set_index("date")
            season = np.stack([np.bincount(course_train.target_category,weights=season_weights(course_train.index,d),
                                          minlength=5)+.01 for d in c.index])
            for name,p in (("new",probs),("incumbent",archive.loc[c.index,PROBS].to_numpy()),
                           ("season_only",season/season.sum(axis=1,keepdims=True))):
                for subset,mask in (("all",np.ones(len(c),bool)),
                                    ("same_market_dates",c.index.isin(all_events.index)),
                                    ("near_boundary",np.abs(c.label_f.to_numpy()[:,None]-BOUNDS).min(axis=1)<=2)):
                    course.append(dict(hour=hour,delay=delay,period=period,model=name,subset=subset,
                                       **metrics(c.target_category.to_numpy()[mask],p[mask])))
            ev = all_events.loc[all_events.index.intersection(panel.index)].sort_index()
            frame = panel.loc[ev.index]
            bb = Brackets(ev)
            probabilities,routes = model_probabilities(bundle["models"],frame,hour,bb)
            results = {"new":probabilities}
            if delay==0:
                for name in ("selected","raw_HRRR"):
                    g = archived.loc[archived.model.eq(name)].loc[ev.index]
                    if not all(a==b for a,b in zip(g.brackets.map(json.loads),ev.brackets)):
                        raise ValueError("archived contract order mismatch")
                    results["incumbent" if name=="selected" else name] = g.probabilities.map(json.loads).map(np.array).tolist()
                for name,models in extra.items():
                    results[name] = model_probabilities(models,frame,hour,bb)[0]
                climate_probs = [None]*len(ev)
                for kind,routine in (("routine_f",True),("all_f",False)):
                    loc = np.flatnonzero(ev.regime.eq("routine_metar").to_numpy()==routine)
                    if len(loc):
                        train = history(target(weather,hour,kind),FREEZE,kind)
                        vals = climate(train,Brackets(ev.iloc[loc]))
                        for i,p in zip(loc,vals):
                            climate_probs[i]=p
                results["season_only"] = climate_probs
                results["listings_only"] = listing_only(past,ev)
            else:
                degree = combined(old["fits"],frame,hour,old["scale"],ev)
                results["incumbent"] = [e.bins@p for e,p in zip(ev.itertuples(),degree)]
            routine = ev.regime.eq("routine_metar")
            floor = np.where(routine,frame[f"routine_floor_{hour}"],np.floor(frame.LGA_max_so_far+.5))
            y = np.where(routine,frame.routine_f,frame.all_f)
            scored = {}
            for name,p in results.items():
                scores = bb.score(p)
                scored[name] = scores
                daily.append(scores.assign(model=name,period=period,hour=hour,delay=delay,
                    route=routes if name=="new" else "",nbm_present=present(frame,"nbm"),
                    rise=np.select([y>floor,y==floor,y<floor],["rose","at_floor","below_proxy_floor"],default="unknown")))
            for d,e,pr,route in zip(ev.index,ev.itertuples(),probabilities,routes):
                for label,prob in zip(e.brackets,pr):
                    bins.append(dict(date=d,hour=hour,delay=delay,event=e.slug,market_id=tickers.loc[(d,label)],
                                     bracket=label,p=prob,winner=label==e.brackets[e.winner],route=route))
            pairs = [("overall","incumbent","new",np.ones(len(ev),bool))]
            if delay==0:
                pairs += [("add_nbm_routed","logistic_hrrr","logistic_nbm",np.ones(len(ev),bool)),
                          ("add_nbm_matched","logistic_hrrr_matched","logistic_nbm",present(frame,"nbm")),
                          ("change_family","structure_control","untuned",np.ones(len(ev),bool)),
                          ("tuning","untuned","new",np.ones(len(ev),bool))]
            for comparison,left,right,mask in pairs:
                delta = (scored[right].log_loss-scored[left].log_loss).loc[mask]
                if len(delta):
                    lo,hi = interval(delta)
                    paired.append(dict(hour=hour,delay=delay,period=period,comparison=comparison,
                        left=left,right=right,days=len(delta),delta_loss=delta.mean(),low=lo,high=hi))
    return pd.concat(daily).reset_index(),pd.DataFrame(bins),pd.DataFrame(course),pd.DataFrame(paired)


def main():
    compact = ROOT/"data"/"bracket_delay_floors.parquet"
    if not compact.exists():
        pd.DataFrame({str(h):delayed_routine_floor(h) for h in (9,12,15)}).to_parquet(compact)
    runs = [evaluate_hour(h) for h in (9,12,15)]
    daily,bins,course,paired = [pd.concat(parts,ignore_index=True) for parts in zip(*runs)]
    keys = ["date","hour","delay"]
    sets = bins.groupby(keys)
    np.testing.assert_allclose(sets.p.sum(),1,atol=1e-10)
    if bins.duplicated(keys+["market_id"]).any() or not sets.winner.sum().eq(1).all():
        raise ValueError("inconsistent contract partition in export")
    truth = contracts().set_index(["date","market_id"]).winner
    expected = truth.reindex(pd.MultiIndex.from_frame(bins[["date","market_id"]]))
    np.testing.assert_array_equal(bins.winner.to_numpy(),expected.to_numpy())
    exported = bins.loc[bins.winner].set_index(keys).sort_index()
    scored = daily.loc[daily.model.eq("new")].set_index(keys).sort_index()
    pd.testing.assert_index_equal(exported.index,scored.index)
    np.testing.assert_allclose(-np.log(exported.p),scored.log_loss,atol=1e-12)
    daily.to_csv(OUT/"daily.csv.gz",index=False)
    bins.to_csv(OUT/"bracket_probabilities.csv.gz",index=False)
    course.to_csv(OUT/"course.csv",index=False)
    paired.to_csv(OUT/"paired.csv",index=False)
    summary = daily.groupby(["hour","period","delay","model"]).agg(days=("date","size"),
        log_loss=("log_loss","mean"),accuracy=("correct","mean"),brier=("brier","mean")).reset_index()
    summary.to_csv(OUT/"market_metrics.csv",index=False)
    plot_scores(summary)
    timely = daily.loc[daily.delay.eq(0)].copy()
    timely["season"] = timely.date.dt.month.map(lambda m:("winter","spring","summer","autumn")[(m%12)//3])
    slices = []
    for field in ("season","regime","rise","nbm_present","route"):
        g = timely.loc[timely.model.isin(["new","incumbent","season_only","listings_only"])]
        if field=="route":
            g = g.loc[g.model.eq("new")]
        agg = g.groupby(["hour","model",field]).agg(days=("date","size"),log_loss=("log_loss","mean"),
            accuracy=("correct","mean"),brier=("brier","mean")).reset_index().rename(columns={field:"value"})
        slices.append(agg.assign(subset=field))
    pd.concat(slices).to_csv(OUT/"subgroups.csv",index=False)
    paths = [ROOT/"data"/"bracket_nbm.parquet",compact,ROOT/"data"/"rolling"/"stations.parquet",
             ROOT/"data"/"rolling"/"external.parquet",MARKET/"integer_labels.csv",MARKET/"contracts.parquet",
             MARKET/"market_predictions.csv.gz"]
    (OUT/"inputs.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths},indent=2))
    print(summary.loc[(summary.delay==0)&summary.model.isin(["new","incumbent"])].to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(1,user_api="blas"):
        main()
