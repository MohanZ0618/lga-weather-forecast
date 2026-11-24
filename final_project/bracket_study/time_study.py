"""Compare shared versus separately fitted issue hours on the same settled market dates."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from estimators import Distribution, Routed
from morning_data import market_layouts, settlements
from panels import Brackets, FREEZE, OUT, contracts, ground, history, target
from replay import interval, model_probabilities

HOURS = (9,12,15)
KINDS = ("all_f","routine_f")
DEST = OUT/"old09"


class TimedDistribution(Distribution):
    def design(self,frame):
        x,anchor,floor = super().design(frame)
        hour = frame.issue_hour.to_numpy()
        return np.column_stack([x,hour==12,hour==15]),anchor,floor


class TimedRoute(Routed):
    candidate_type = TimedDistribution


def common_dates(archive):
    hits = archive.pivot(index="date",columns="hour",values="correct")
    return hits.reindex(columns=HOURS).dropna().sort_index().index


def training_rows(hour,kind):
    # Slice the development panel before using a daily weather label.
    frame = ground(hour,development=True).assign(issue_hour=hour)
    return history(target(frame,hour,kind),FREEZE,kind)


def train(tables,spec):
    return {kind:TimedRoute(spec["family"],spec["tier"],spec["params"]).fit(table)
            for kind,table in tables.items()}


def summarize(daily):
    by_hour = daily.assign(scope=daily.hour.astype(str))
    both = pd.concat([by_hour,daily.assign(scope="overall")])
    return both.groupby(["model","scope"],sort=False).agg(
        dates=("date","nunique"),forecasts=("correct","size"),hits=("correct","sum"),
        accuracy=("correct","mean"),log_loss=("log_loss","mean")).reset_index()


def main():
    spec = json.loads((OUT/"09"/"parameters.json").read_text())
    if spec["end"]!=FREEZE or spec["family"]!="blend":
        raise ValueError("unexpected 09:00 reference model")
    original = pd.read_csv(OUT/"daily.csv.gz",parse_dates=["date"])
    original = original.loc[original.model.eq("new") & original.delay.eq(0)]
    dates = common_dates(original)
    if dates.empty or dates.min()<pd.Timestamp(FREEZE):
        raise ValueError("invalid common evaluation period")
    original = original.loc[original.date.isin(dates)]
    archive = original[["date","hour","winner","predicted","correct","log_loss"]].copy()
    archive["model"] = "current"
    raw = contracts()
    listings = raw.loc[raw.date.isin(dates)].drop(columns="winner")
    per_hour = {h:{k:training_rows(h,k) for k in KINDS} for h in HOURS}
    shared = train({k:pd.concat([per_hour[h][k] for h in HOURS]) for k in KINDS},spec)
    forecasts = []
    for hour in HOURS:
        separate = train(per_hour[hour],spec)
        layouts = market_layouts(listings,hour)
        events = pd.DataFrame.from_dict(layouts,orient="index").loc[dates]
        bb = Brackets(events)
        # The replay adapter accepts label columns. Mask them for inference.
        panel = ground(hour).loc[dates].assign(issue_hour=hour,all_f=np.nan,routine_f=np.nan)
        for name,models in (("separate_blend",separate),("shared_blend",shared)):
            probs,_ = model_probabilities(models,panel,hour,bb)
            for d,p,ids,names in zip(dates,probs,events.ids,events.brackets):
                if not np.isfinite(p).all() or p.min()<0 or abs(p.sum()-1)>1e-8:
                    raise ValueError(f"invalid probabilities at {d} {hour} {name}")
                forecasts.extend(dict(date=d,hour=hour,model=name,market_id=i,bracket=b,p=v)
                                 for i,b,v in zip(ids,names,p))
    forecasts = pd.DataFrame(forecasts)
    DEST.mkdir(exist_ok=True,parents=True)
    forecasts.to_csv(DEST/"time_probabilities.csv.gz",index=False)
    truth = settlements(raw).market_id.rename("winning_id")
    forecast = forecasts.join(truth,on="date",validate="many_to_one")
    key = ["date","hour","model"]
    picks = forecast.loc[forecast.groupby(key).p.idxmax(),key+["bracket"]].rename(columns={"bracket":"predicted"})
    resolved = forecast.loc[forecast.market_id.eq(forecast.winning_id),key+["bracket","p"]]
    resolved = resolved.rename(columns={"bracket":"winner"})
    scored = picks.merge(resolved,on=key,validate="one_to_one")
    if len(scored)!=len(dates)*len(HOURS)*2:
        raise ValueError("missing settlement in issue-time partition")
    official = archive.set_index(["date","hour"]).winner
    expected = official.reindex(pd.MultiIndex.from_frame(scored[["date","hour"]]))
    np.testing.assert_array_equal(scored.winner,expected)
    scored["correct"] = scored.predicted.eq(scored.winner).astype(int)
    scored["log_loss"] = -np.log(scored.pop("p"))
    daily = pd.concat([archive,scored],ignore_index=True).sort_values(["date","hour","model"])
    daily.to_csv(DEST/"time_daily.csv.gz",index=False)
    metrics = summarize(daily)
    metrics.to_csv(DEST/"time_metrics.csv",index=False)
    print(metrics.to_string(index=False))
    wide = scored.pivot(index=["date","hour"],columns="model",values="correct")
    delta = wide.shared_blend-wide.separate_blend
    by_scope = [(h,delta.xs(h,level="hour")) for h in HOURS]
    by_scope.append(("overall",delta.groupby(level="date").mean()))
    rows = []
    for scope,series in by_scope:
        low,high = interval(series)
        rows.append(dict(scope=scope,dates=len(series),gain=series.mean(),low=low,high=high))
    pd.DataFrame(rows).to_csv(DEST/"time_comparison.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
