"""Predict whole Fahrenheit degrees and score each day's listed NYC contracts."""
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data import HOURS, OUT, SNAPSHOT_END, STATION_IDS, TZ, station_inputs
from experiment import season_features
from late_replay import station_archive
from market_data import MARKET, whole_f
from study import FREEZE, add_forecasts, inputs

DEGREES = np.arange(-110,151)
P_COLS = [f"p_{d}F" for d in DEGREES]
BASE = ["season_sin","season_cos","LGA_temp","LGA_rh","LGA_wind","LGA_trend_1h","LGA_trend_3h",
        "LGA_spread","LGA_east","LGA_north","LGA_sky","JFK_temp","EWR_temp","TEB_temp","LGA_previous_max"]
WEATHER = ["hrrr_remaining_max","hrrr_warming","latest_high","latest_warm","model_cloud","sat10","sat40"]
YEARS = [("2023-01-01","2024-01-01"),("2024-01-01","2025-01-01"),("2025-01-01",FREEZE)]
SCALES = (.75,1.,1.25,1.5)


def degree_mass(location,sigma,floor,routine):
    # Discretize at half degrees. Fold probability below the qualifying observed high onto that high.
    cdf = ndtr((DEGREES[None,:]+.5-location[:,None])/sigma)
    p = np.diff(cdf,axis=1,prepend=0)
    p /= p.sum(axis=1,keepdims=True)
    limit = np.clip(np.nan_to_num(whole_f(floor),nan=DEGREES[0]),DEGREES[0],DEGREES[-1]).astype(int)
    below = DEGREES[None,:]<limit[:,None]
    bounded = np.where(below,0,p)
    bounded[np.arange(len(p)),limit-DEGREES[0]] += (p*below).sum(axis=1)
    # Other resolution sources can disagree with an archived station floor.
    p = bounded if routine else .95*bounded+.05*p
    # Allow for archive/settlement discrepancies including post-issue corrections.
    return .999*p+.001/len(DEGREES)


def frame_for(hour,tables,labels):
    frame = inputs(hour)
    later = labels.loc[(labels.index>frame.index.max())&(labels.index<=SNAPSHOT_END)].index
    if len(later):
        extra = pd.concat([station_inputs(t,later,hour,0).add_prefix(s+"_") for s,t in tables.items()],axis=1)
        frame = pd.concat([frame,add_forecasts(season_features(extra),hour)])
    return frame.join(labels[["all_f","routine_f","hours","blocks",f"routine_floor_{hour}"]],validate="one_to_one")


def curve(frame,hour,target,source):
    x = frame[BASE + (WEATHER if source=="weather" else [])].copy()
    floor = frame[f"routine_floor_{hour}" if target=="routine_f" else "LGA_max_so_far"]
    # On missing numerical cycles, the estimator still sees a station-based anchor and inputs.
    anchor = floor.combine_first(frame.LGA_temp).combine_first(frame.LGA_previous_max)
    if source=="weather":
        nwp = frame.latest_high.combine_first(frame.hrrr_remaining_max)
        anchor = pd.concat([anchor,nwp],axis=1).max(axis=1)
    x["floor"] = floor
    x["anchor"] = anchor
    return x,anchor,floor


def fitted(frame,hour,target,config):
    source,family,param = config
    x,anchor,_ = curve(frame,hour,target,source)
    target_delta = frame[target]-anchor
    ok = target_delta.notna() & frame.hours.ge(20) & frame.blocks.eq(4)
    if family=="raw":
        return None
    if family=="ridge":
        model = make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=param))
    else:
        model = HistGradientBoostingRegressor(max_leaf_nodes=param,min_samples_leaf=25,max_iter=120,
                                               learning_rate=.05,early_stopping=False,random_state=4260)
    model.fit(x.loc[ok],target_delta.loc[ok])
    return model


def centers(model,frame,hour,target,source):
    x,anchor,floor = curve(frame,hour,target,source)
    delta = model.predict(x) if model is not None else 0.
    return anchor.to_numpy()+delta,floor.to_numpy()


def fit_distribution(frame,start,hour,target,config):
    history = frame.loc[frame.index<pd.Timestamp(start)]
    boundary = pd.Timestamp(start)-pd.Timedelta(days=120)
    early = history.loc[history.index<boundary]
    cal = history.loc[(history.index>=boundary)&history[target].notna()&history.hours.ge(20)&history.blocks.eq(4)]
    pilot = fitted(early,hour,target,config)
    mu,_ = centers(pilot,cal,hour,target,config[0])
    error = cal[target].to_numpy()-mu
    offset = float(np.nanmedian(error))
    sigma = max(.65,float(np.nanmean(abs(error-offset)))*np.sqrt(np.pi/2))
    return dict(model=fitted(history,hour,target,config),offset=offset,sigma=sigma,target=target,config=config)


def forecast(fit,case,hour,scale):
    mu,floor = centers(fit["model"],case,hour,fit["target"],fit["config"][0])
    return degree_mass(mu+fit["offset"],fit["sigma"]*scale,floor,fit["target"]=="routine_f")


def membership(low,high):
    return (DEGREES>=np.where(np.isfinite(low),low,-np.inf)[:,None]) & (
            DEGREES<=np.where(np.isfinite(high),high,np.inf)[:,None])


def market_days(contracts,hour,frame):
    rows = []
    for date,g in contracts.groupby("date"):
        if date not in frame.index:
            continue
        at = (date+pd.Timedelta(hours=hour)).tz_localize(TZ)
        win = g.loc[g.winner].iloc[0]
        if g.listed.max()>at or pd.isna(win.closed) or win.closed<=at:
            continue
        bins = membership(g.low.to_numpy(float),g.high.to_numpy(float))
        if not (bins.sum(axis=0)==1).all():
            raise ValueError("Market does not partition integer temperatures")
        rows.append(dict(date=date,regime=win.regime,slug=win.slug,closed=win.closed,
                         brackets=g.bracket.tolist(),winner=int(np.argmax(g.winner)),bins=bins))
    return pd.DataFrame(rows).set_index("date").sort_index()


def scores(case,p,events):
    common = events.index.intersection(case.index)
    positions = case.index.get_indexer(common)
    result = []
    for pos,(date,e) in zip(positions,events.loc[common].iterrows()):
        prob = e.bins @ p[pos]
        np.testing.assert_allclose(prob.sum(),1,atol=1e-10)
        result.append(dict(date=date,slug=e.slug,regime=e.regime,winner=e.brackets[e.winner],
            predicted=e.brackets[int(prob.argmax())],correct=int(prob.argmax()==e.winner),n_bins=len(prob),
            loss=-np.log(prob[e.winner]),brier=np.sum((prob-np.eye(len(prob))[e.winner])**2),
            brackets=json.dumps(e.brackets,ensure_ascii=False),probabilities=json.dumps(prob.tolist())))
    return result


def combined(fits,case,hour,scale,events):
    p = forecast(fits["all_f"],case,hour,scale)
    use = case.index.isin(events.index[events.regime.eq("routine_metar")])
    if use.any():
        p[use] = forecast(fits["routine_f"],case.loc[use],hour,scale)
    return p


def run_hour(hour,frame,contracts):
    events = market_days(contracts,hour,frame)
    configs = [("weather","raw",0),("station","ridge",1)]
    configs += [("weather","ridge",a) for a in (.1,1,10)]
    configs += [("weather","boost",n) for n in (7,15)]
    grid = []
    for config in configs:
        for start,end in YEARS:
            # All contracts from the selection period must have resolved before the freeze.
            case = frame.loc[(frame.index>=start)&(frame.index<end)]
            eligible_events = events.loc[events.closed < pd.Timestamp(FREEZE,tz=TZ)]
            fit = fit_distribution(frame,start,hour,"all_f",config)
            for scale in SCALES:
                p = forecast(fit,case,hour,scale)
                valid = case.all_f.notna() & case.hours.ge(20) & case.blocks.eq(4)
                obs = case.loc[valid,"all_f"].to_numpy(int)-DEGREES[0]
                loss = -np.log(p[valid][np.arange(len(obs)),obs]).mean()
                days = scores(case,p,eligible_events)
                grid.append(dict(config=json.dumps(config),scale=scale,fold=start[:4],days=int(valid.sum()),
                                 degree_log_loss=loss,market_days=len(days),
                                 market_loss_sum=sum(d["loss"] for d in days)))
    grid = pd.DataFrame(grid)
    ranking = grid.groupby(["config","scale"]).agg(degree_loss=("degree_log_loss","mean"),
        n=("market_days","sum"),loss_sum=("market_loss_sum","sum")).reset_index()
    ranking["market_log_loss"] = ranking.loss_sum/ranking.n
    ranking = ranking.sort_values(["market_log_loss","degree_loss"])
    grid.to_csv(MARKET/f"validation_{hour:02}.csv",index=False)
    ranking.to_csv(MARKET/f"ranking_{hour:02}.csv",index=False)
    selected = ranking.iloc[0]
    base = ranking.loc[ranking.config.eq(json.dumps(configs[0]))].iloc[0]
    predictions,degree_results = [],[]
    for name,row in [("selected",selected),("raw_HRRR",base)]:
        config = json.loads(row.config)
        fits = {t:fit_distribution(frame,FREEZE,hour,t,config) for t in ("all_f","routine_f")}
        case = frame.loc[frame.index>=FREEZE]
        p = combined(fits,case,hour,row.scale,events)
        group = events.regime.reindex(case.index).eq("routine_metar")
        target = case.all_f.where(~group,case.routine_f)
        ok = target.notna() & case.hours.ge(20) & case.blocks.eq(4)
        actual = target.loc[ok].to_numpy(int)
        band = actual-DEGREES[0]
        predicted = DEGREES[p[ok].argmax(axis=1)]
        degree_results.append(dict(hour=hour,model=name,days=len(actual),
            exact_degree_accuracy=np.mean(actual==predicted),degree_log_loss=-np.log(p[ok][np.arange(len(actual)),band]).mean(),
            mean_absolute_error=np.mean(abs(actual-(p[ok]@DEGREES)))))
        predictions.extend(dict(**r,hour=hour,model=name) for r in scores(case,p,events))
        if name=="selected":
            wide = pd.DataFrame(p,index=case.index,columns=P_COLS)
            wide["target_proxy_f"] = target
            wide["exact_target_source"] = np.where(group,"routine archive reconstruction","all-report archive proxy")
            wide.to_parquet(MARKET/f"one_degree_{hour:02}.parquet")
            joblib.dump(dict(hour=hour,fits=fits,scale=row.scale,support=DEGREES,train_end="2025-08-31"),
                        MARKET/f"one_degree_{hour:02}.joblib",compress=3)
    return predictions,degree_results


def main():
    labels = pd.read_csv(MARKET/"integer_labels.csv",parse_dates=["date"],index_col="date")
    tables = station_archive()
    contracts = pd.read_parquet(MARKET/"contracts.parquet")
    contracts["date"] = pd.to_datetime(contracts.date)
    for col in ("listed","closed"):
        contracts[col] = pd.to_datetime(contracts[col],utc=True,format="mixed",errors="coerce")
    predictions,degree_results = [],[]
    for hour in HOURS:
        p,d = run_hour(hour,frame_for(hour,tables,labels),contracts)
        predictions.extend(p)
        degree_results.extend(d)
    rows = pd.DataFrame(predictions)
    rows.to_csv(MARKET/"market_predictions.csv.gz",index=False)
    groups = [rows.assign(period="all"),rows.assign(period=rows.regime),
              rows.assign(period=np.where(rows.date<"2026-09-01","historical","autumn"))]
    report = pd.concat(groups).groupby(["period","hour","model"]).agg(days=("date","size"),
        correct=("correct","sum"),accuracy=("correct","mean"),log_loss=("loss","mean"),brier=("brier","mean"))
    report.to_csv(MARKET/"market_metrics.csv")
    pd.DataFrame(degree_results).to_csv(MARKET/"degree_metrics.csv",index=False)
    print(report.loc["all"].to_string(),flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
