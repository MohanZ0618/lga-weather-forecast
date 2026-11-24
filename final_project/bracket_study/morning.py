"""Morning bracket selection, walk-forward prediction, and independent settlement scoring."""
import argparse
from itertools import product
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from morning_data import SNAPSHOT, TAIL, market_layouts, settlements
from panels import MARKET, OUT, TZ, contracts
from market_study import DEGREES
from replay import interval

DEST = OUT/"morning"
FREEZE_DATE = pd.Timestamp("2025-09-01")
CANDIDATES = ("ridge","boost","blend")
BASE = dict(market_weight=0.,heaping=0.,scale=1.)
COLS = ["season_cos","season_sin","nbm_peak_hours","nbm_spread","nbm_cloud","nbm_minus_hrrr",
        "latest_warm","model_cloud","sat10","sat40","hrrr_09_bias","hrrr_peak_hours","hrrr_plateau",
        "hrrr_noon_warm","hrrr_15_warm","LGA_trend_1h","LGA_trend_3h","LGA_spread","LGA_sky",
        "LGA_east","LGA_north","JFK_minus_lga","EWR_minus_lga","TEB_minus_lga"]


def design(f):
    anchor = f.nbm_high.fillna(f.latest_high).fillna(f.LGA_temp+5).fillna(f.LGA_previous_max)
    # These fallback constants are fixed, including for the first archive day with no prior observations.
    anchor = anchor.fillna(55.)
    x = f[COLS].assign(anchor=anchor,already_observed=f.LGA_early_high-anchor,
        current_gap=f.LGA_temp-anchor,previous_gap=f.LGA_previous_max-anchor,
        missing_nbm=f.nbm_high.isna(),missing_hrrr=f.latest_high.isna())
    return x,anchor


def regressors(x,y):
    ridge = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                          StandardScaler(),Ridge(alpha=40))
    boost = HistGradientBoostingRegressor(max_leaf_nodes=7,min_samples_leaf=35,max_iter=140,
        learning_rate=.045,l2_regularization=5,early_stopping=False,random_state=4260)
    return {"ridge":ridge.fit(x,y),"boost":boost.fit(x,y)}


def centers(models,x,anchor):
    result = {name:anchor.to_numpy()+model.predict(x) for name,model in models.items()}
    result["blend"] = (result["ridge"]+result["boost"])/2
    return result


def labels():
    return pd.read_csv(MARKET/"integer_labels.csv",parse_dates=["date"],index_col="date")


def month_inputs(panel,observed,dates,month,kind):
    # At the first local midnight of each forecast month, purge seven full local dates.
    cutoff = pd.Timestamp(month)-pd.Timedelta(days=7)
    valid = observed.loc[observed.index<cutoff]
    valid = valid.loc[valid.hours.ge(20)&valid.blocks.eq(4)&valid[kind].notna()]
    train_dates = panel.index.intersection(valid.index)
    if not len(train_dates) or train_dates.max()>=cutoff or min(dates)<month:
        raise ValueError("overlapping forecast and training periods")
    split = cutoff-pd.Timedelta(days=270)
    early = train_dates[train_dates<split]
    cal = train_dates[train_dates>=split]
    if len(cal)<120 or len(early)<365:
        raise ValueError("insufficient chronological calibration history")
    x,anchor = design(panel.loc[train_dates])
    y = valid.loc[train_dates,kind]
    pilot = regressors(x.loc[early],(y-anchor).loc[early])
    cal_mu = centers(pilot,x.loc[cal],anchor.loc[cal])
    recent = y.loc[y.index>=cutoff-pd.Timedelta(days=730)].to_numpy(int)
    counts = np.bincount(recent%9,minlength=9)+20.
    digit_weight = counts/counts.mean()
    fitted = regressors(x,y-anchor)
    current,base = design(panel.loc[dates])
    estimates = centers(fitted,current,base)
    rows = {}
    for date in dates:
        case = panel.loc[date]
        phase = 2*np.pi*(cal.dayofyear.to_numpy()-date.dayofyear)/365.25
        weights = np.exp((np.cos(phase)-1)*.8)
        for col,s in (("nbm_spread",2.),("nbm_minus_hrrr",4.)):
            if np.isfinite(case[col]):
                diff = panel.loc[cal,col].to_numpy()-case[col]
                weights = weights*np.where(np.isfinite(diff),.2+.8*np.exp(-.5*(diff/s)**2),.5)
        weights = weights/weights.sum()
        rows[date] = dict(train_end=train_dates.max(),cal_start=cal.min(),cal_end=cal.max(),
                          trained_before=cutoff,kind=kind,digit_weight=digit_weight,
                          floor=case.LGA_early_high if kind=="all_f" else np.nan)
        for family in CANDIDATES:
            residual = y.loc[cal].to_numpy()-cal_mu[family]
            bias = np.sum(weights*residual)
            sd = max(1.,np.sqrt(np.sum(weights*(residual-bias)**2)))
            rows[date][family] = (float(estimates[family][current.index.get_loc(date)]+bias),float(sd))
    return rows


def forecast(state,layout,family,market_weight,heaping,scale):
    mu,sd = state[family]
    mu += market_weight*np.clip(layout["center"]-mu,-8,8)
    cdf = ndtr((DEGREES+.5-mu)/(sd*scale))
    degree = np.diff(cdf,prepend=0.)
    degree[-1] += 1.-cdf[-1]
    degree *= state["digit_weight"][DEGREES%9]**heaping
    if np.isfinite(state["floor"]):
        floor = int(np.clip(np.floor(state["floor"]+.5),DEGREES[0],DEGREES[-1])-DEGREES[0])
        degree[floor] += degree[:floor].sum()
        degree[:floor] = 0
    prob = layout["bins"]@(degree/degree.sum())
    return .999*prob+.001/len(prob)


def states(panel,observed,layouts,start,end,*,frozen=False):
    days = sorted(d for d in layouts if pd.Timestamp(start)<=d<pd.Timestamp(end) and d in panel.index)
    index = pd.DatetimeIndex(days)
    result = {}
    fits = [(pd.Timestamp(start),index)] if frozen else [
        (month.start_time,index[index.to_period("M")==month]) for month in index.to_period("M").unique()]
    for origin,monthly in fits:
        for kind in ("all_f","routine_f"):
            select = [d for d in monthly if ("routine_f" if layouts[d]["regime"]=="routine_metar" else "all_f")==kind]
            if select:
                result.update(month_inputs(panel,observed,select,origin,kind))
    return result


def stage_scores(stage,states_,layouts,results,configs,lock):
    # Settlement times in the archive are closing proxies. Allow two extra days before use for selection.
    available = results.loc[results.closed+pd.Timedelta(days=2)<pd.Timestamp(lock,tz=TZ)]
    rows = []
    for family,params in configs:
        hits,loss = [],[]
        for date,state in states_.items():
            if date not in available.index:
                continue
            layout = layouts[date]
            p = forecast(state,layout,family,**params)
            y = layout["ids"].index(available.loc[date,"market_id"])
            hits.append(p.argmax()==y)
            loss.append(-np.log(p[y]))
        if not hits:
            raise ValueError("no resolved selection dates")
        rows.append(dict(stage=stage,family=family,**params,days=len(hits),correct=sum(hits),
                         accuracy=np.mean(hits),log_loss=np.mean(loss)))
    return pd.DataFrame(rows).sort_values(["correct","log_loss"],ascending=[False,True],kind="stable")


def select():
    raw = contracts()
    layouts = market_layouts(raw.drop(columns="winner"))
    results = settlements(raw.loc[raw.date<FREEZE_DATE])
    panel = pd.read_parquet(SNAPSHOT).loc[lambda f:f.index<FREEZE_DATE]
    observed = labels().loc[lambda f:f.index<FREEZE_DATE]
    screen = states(panel,observed,layouts,"2025-01-01","2025-05-01")
    first = stage_scores("family",screen,layouts,results,[(f,BASE) for f in CANDIDATES],"2025-05-01")
    winner = first.iloc[0].family
    validation = states(panel,observed,layouts,"2025-05-01",FREEZE_DATE)
    options = [(winner,dict(market_weight=w,heaping=h,scale=s)) for w,h,s in product(
        (0.,.15,.30,.45),(0.,.5,1.),(.85,1.,1.15))]
    tune = stage_scores("tune",validation,layouts,results,options,FREEZE_DATE)
    pd.concat([first,tune]).to_csv(DEST/"validation.csv",index=False)
    chosen = tune.iloc[0]
    spec = dict(family=winner,params={k:float(chosen[k]) for k in BASE},locked_before=str(FREEZE_DATE.date()),
                training_gap_days=7,settlement_margin_days=2)
    (DEST/"spec.json").write_text(json.dumps(spec,indent=2))
    print("Selected on earlier contracts only",spec,flush=True)


def predict():
    spec = json.loads((DEST/"spec.json").read_text())
    layouts = market_layouts(contracts().drop(columns="winner"))
    panel = pd.read_parquet(SNAPSHOT)
    historical_labels = labels()
    estimates = states(panel,historical_labels,layouts,FREEZE_DATE,"2026-11-22")
    frozen = states(panel,historical_labels.loc[historical_labels.index<FREEZE_DATE],layouts,
                    FREEZE_DATE,"2026-11-22",frozen=True)
    rows = []
    for date,state in sorted(estimates.items()):
        layout = layouts[date]
        p = forecast(state,layout,spec["family"],**spec["params"])
        fixed = forecast(frozen[date],layout,spec["family"],**spec["params"])
        for vec in (p,fixed):
            if not np.isfinite(vec).all() or vec.min()<0 or abs(vec.sum()-1)>1e-9:
                raise ValueError(f"invalid distribution on {date}")
        for id_,title,prob,p_fixed in zip(layout["ids"],layout["brackets"],p,fixed):
            rows.append(dict(date=date,market_id=id_,bracket=title,p=prob,p_frozen=p_fixed,kind=state["kind"],
                frozen_train_end=frozen[date]["train_end"],
                train_end=state["train_end"],cal_start=state["cal_start"],cal_end=state["cal_end"],
                trained_before=state["trained_before"],latest_listing=layout["latest_listing"]))
    pd.DataFrame(rows).to_csv(DEST/"predictions.csv.gz",index=False)
    print("Written forecasts",len(estimates),flush=True)


def tail_reference():
    # Re-use the previously frozen 09 model for the later archive, with its original 0-minute station delay.
    cached = DEST/"reference_tail.csv.gz"
    if cached.exists():
        return pd.read_csv(cached,parse_dates=["date"],dtype={"market_id":str})
    import joblib
    from late_replay import station_archive
    from market_study import frame_for
    from panels import Brackets,TABLE
    from replay import model_probabilities
    panel = frame_for(9,station_archive(cached_only=True),labels()).loc[TAIL:]
    b = pd.read_parquet(TABLE).query("hour==9")
    b = b.set_index(pd.to_datetime(b.date))
    panel = panel.join(b[["nbm_high","nbm_peak_hours","nbm_spread","nbm_cloud"]])
    panel["nbm_disagreement"] = panel.nbm_high-panel.latest_high
    layout = market_layouts(contracts().drop(columns="winner"))
    ev = pd.DataFrame.from_dict({d:v for d,v in layout.items() if d in panel.index},orient="index").sort_index()
    base = joblib.load(OUT/"09"/"frozen.joblib")
    probabilities,_ = model_probabilities(base["models"],panel.loc[ev.index],9,Brackets(ev))
    forecasts = pd.DataFrame([dict(date=d,market_id=i,p=p) for (d,e),vec in zip(ev.iterrows(),probabilities)
                              for i,p in zip(e.ids,vec)])
    forecasts.to_csv(cached,index=False)
    return forecasts


def results_by_day(forecasts,truth):
    rows = []
    for date,g in forecasts.groupby("date"):
        p = g.p.to_numpy()
        idx = np.flatnonzero(g.market_id.eq(truth.loc[date,"market_id"]).to_numpy())
        if len(idx)!=1 or abs(p.sum()-1)>1e-7:
            raise ValueError(f"inconsistent settlement at {date}")
        win = idx.item()
        best = p.argmax()
        rows.append(dict(date=date,hit=int(best==win),confidence=p[best],correct_p=p[win],log_loss=-np.log(p[win]),
                         top2=int(win in np.argsort(-p,kind="stable")[:2]),
                         pick=g.bracket.iloc[best] if "bracket" in g else g.market_id.iloc[best]))
    return pd.DataFrame(rows).set_index("date")


def score():
    archived = contracts()
    truth = settlements(archived)
    predictions = pd.read_csv(DEST/"predictions.csv.gz",parse_dates=["date"],dtype={"market_id":str})
    unique = predictions.drop_duplicates("date").set_index("date")
    issue = pd.Series((unique.index+pd.Timedelta(hours=9)).tz_localize(TZ),index=unique.index)
    cutoff = unique.index.to_period("M").to_timestamp()-pd.Timedelta(days=7)
    if (pd.to_datetime(unique.train_end)>=cutoff).any() or (pd.to_datetime(unique.cal_end)>=cutoff).any():
        raise ValueError("a training or calibration label crosses its monthly embargo")
    if (pd.to_datetime(unique.frozen_train_end)>=FREEZE_DATE-pd.Timedelta(days=7)).any():
        raise ValueError("static fit crosses its original embargo")
    if (pd.to_datetime(unique.latest_listing,utc=True,format="mixed")>issue).any():
        raise ValueError("late market listing")
    panel = pd.read_parquet(SNAPSHOT).loc[unique.index]
    for column in ("LGA_observed","EWR_observed","TEB_observed","JFK_observed"):
        if (panel[column]>issue-pd.Timedelta(minutes=5)).any():
            raise ValueError(f"late observation in {column}")
    for column in ("nbm_time","hrrr_time","cloud_time","sat_time"):
        if (panel[column]>issue).any():
            raise ValueError(f"late forecast or satellite input in {column}")
    audit = dict(forecast_days=len(unique),issued_at="09:00 America/New_York",station_cutoff="08:55",
                 last_static_label=unique.frozen_train_end.max(),first_monthly_label_end=unique.train_end.min(),
                 last_monthly_label_end=unique.train_end.max(),all_time_checks_passed=True)
    (DEST/"audit.json").write_text(json.dumps(audit,indent=2))
    new = results_by_day(predictions,truth)
    fixed = results_by_day(predictions.assign(p=predictions.p_frozen),truth)
    old = pd.read_csv(OUT/"bracket_probabilities.csv.gz",parse_dates=["date"],dtype={"market_id":str})
    old = old.loc[old.hour.eq(9)&old.delay.eq(0),["date","market_id","bracket","p"]]
    tail = tail_reference()
    tail["bracket"] = tail.market_id.map(archived.astype({"market_id":str}).set_index("market_id").bracket)
    old = results_by_day(pd.concat([old,tail]),truth)
    daily = new.join(old,rsuffix="_old",how="left").join(fixed.add_suffix("_frozen"),how="left")
    daily.index.name = "date"
    daily["period"] = np.select([daily.index<pd.Timestamp("2026-09-01"),daily.index<TAIL],
                                ["original_year","autumn"],default="later_15_days")
    daily.to_csv(DEST/"daily.csv")
    rows = []
    for period,group in [*daily.groupby("period",sort=False),("all",daily)]:
        paired = group.dropna(subset=["hit_old"])
        delta = paired.hit-paired.hit_old
        low,high = interval(delta)
        rows.append(dict(period=period,all_forecasts=len(group),paired=len(paired),old_correct=int(paired.hit_old.sum()),
            correct=int(paired.hit.sum()),old_accuracy=paired.hit_old.mean(),accuracy=paired.hit.mean(),
            frozen_correct=int(paired.hit_frozen.sum()),frozen_accuracy=paired.hit_frozen.mean(),
            frozen_loss=paired.log_loss_frozen.mean(),
            fixed=int((delta>0).sum()),lost=int((delta<0).sum()),delta_low=low,delta_high=high,
            old_loss=paired.log_loss_old.mean(),loss=paired.log_loss.mean(),top2=paired.top2.mean()))
    summary = pd.DataFrame(rows)
    summary.to_csv(DEST/"metrics.csv",index=False)
    confidence = []
    common = daily.dropna(subset=["hit_old"])
    contracts_by_name = archived.set_index(["date","bracket"])
    for period,g in [*common.groupby("period",sort=False),("all",common)]:
        for model,suffix in (("previous","_old"),("monthly","")):
            for threshold in (.50,.60,.70):
                selected = g.loc[g["confidence"+suffix]>=threshold]
                hits = selected["hit"+suffix]
                chosen = contracts_by_name.loc[pd.MultiIndex.from_arrays([selected.index,selected["pick"+suffix]])]
                confidence.append(dict(period=period,model=model,min_probability=threshold,
                    days=len(selected),total=len(g),coverage=len(selected)/len(g),
                    hits=int(hits.sum()),accuracy=hits.mean(),other_model_hits=int(
                        selected["hit" if suffix else "hit_old"].sum()),open_tail_picks=int(
                            (chosen.low.isna()|chosen.high.isna()).sum())))
    pd.DataFrame(confidence).to_csv(DEST/"confidence.csv",index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["select","predict","score","all"])
    args = parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(2):
        for step in (("select","predict","score") if args.stage=="all" else [args.stage]):
            {"select":select,"predict":predict,"score":score}[step]()
