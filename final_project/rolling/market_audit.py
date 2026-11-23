"""Verify and export one-degree forecasts mapped to actual market contracts."""
import json

import numpy as np
import pandas as pd

from data import HOURS, OUT
from market_data import MARKET,whole_f
from market_study import DEGREES,P_COLS,membership


def main():
    contracts = pd.read_parquet(MARKET/"contracts.parquet")
    fields = ["date","event_id","slug","regime","market_id","bracket","low","high","winner","listed","closed"]
    contracts[fields].to_csv(MARKET/"historical_contract_bins.csv.gz",index=False)
    labels = pd.read_csv(MARKET/"integer_labels.csv")
    actual = contracts.loc[contracts.winner,fields].merge(labels,on="date",how="left",validate="one_to_one")
    actual["observed_integer_f"] = np.where(actual.regime.eq("routine_metar"),actual.routine_f,actual.all_f)
    actual["within_winning_bracket"] = ((actual.observed_integer_f>=actual.low.fillna(-np.inf))&
                                        (actual.observed_integer_f<=actual.high.fillna(np.inf))).astype("boolean")
    actual.loc[actual.observed_integer_f.isna(),"within_winning_bracket"] = pd.NA
    actual.to_csv(MARKET/"settlement_check.csv",index=False)
    picks = pd.read_csv(MARKET/"market_predictions.csv.gz")
    chosen = picks.loc[picks.model.eq("selected")]
    rows = []
    for hour in HOURS:
        wide = pd.read_parquet(MARKET/f"one_degree_{hour:02}.parquet")
        np.testing.assert_allclose(wide[P_COLS].sum(axis=1),1,atol=1e-12)
        if not np.isfinite(wide[P_COLS].to_numpy()).all() or wide[P_COLS].min().min()<0:
            raise ValueError("Invalid one-degree forecast")
        for r in chosen.loc[chosen.hour.eq(hour)].itertuples():
            bins = contracts.loc[contracts.date.eq(r.date)]
            p = wide.loc[pd.Timestamp(r.date),P_COLS].to_numpy(float)
            mapped = membership(bins.low.to_numpy(float),bins.high.to_numpy(float)) @ p
            np.testing.assert_allclose(mapped,json.loads(r.probabilities),atol=1e-12)
            assert bins.loc[bins.winner,"bracket"].item()==r.winner
            for b,probability in zip(bins.itertuples(),mapped):
                rows.append(dict(date=r.date,hour=hour,event=r.slug,regime=r.regime,market_id=b.market_id,
                                 bracket=b.bracket,low=b.low,high=b.high,p=probability,winner=b.winner))
    mapped = pd.DataFrame(rows)
    mapped.to_csv(MARKET/"contract_probabilities.csv.gz",index=False)
    comparisons = []
    for hour in HOURS:
        panel = picks.loc[picks.hour.eq(hour)].pivot(index="date",columns="model",values=["loss","correct"])
        delta = pd.DataFrame({"loss":panel.loss.selected-panel.loss.raw_HRRR,
                              "correct":panel.correct.selected-panel.correct.raw_HRRR,"days":1})
        delta.index = pd.to_datetime(delta.index)
        weeks = delta.groupby(pd.Grouper(freq="7D")).sum().to_numpy()
        rng = np.random.default_rng(4260)
        draws = weeks[rng.integers(len(weeks),size=(5000,len(weeks)))].sum(axis=1)
        draws = draws[draws[:,2]>0]
        ci = np.quantile(draws[:,:2]/draws[:,2,None],[.025,.975],axis=0)
        comparisons.append(dict(hour=hour,days=len(delta),net_correct=int(delta.correct.sum()),
            delta_accuracy=delta.correct.mean(),delta_loss=delta.loss.mean(),loss_low=ci[0,0],loss_high=ci[1,0],
            accuracy_low=ci[0,1],accuracy_high=ci[1,1]))
    pd.DataFrame(comparisons).to_csv(MARKET/"paired_results.csv",index=False)
    overlap = pd.read_csv(OUT/"iem_lcd_overlap.csv")
    summary = {"verified_events":int(contracts.date.nunique()),"scored_dates":int(chosen.date.nunique()),
               "mapped_forecasts":len(chosen),"mapped_bin_probabilities":len(mapped),
               "overlap_days":len(overlap),"overlap_integer_agreement":int(np.sum(
                   whole_f(overlap.target_max)==whole_f(overlap.target_max_iem))),
               "settlement_crosscheck":actual.groupby("regime").within_winning_bracket.agg(["count","sum"]).to_dict("index")}
    (MARKET/"audit.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=="__main__":
    main()
