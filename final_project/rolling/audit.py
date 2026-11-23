"""Check archive availability, baseline reproduction and saved forecast probabilities."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import DATA, HOURS, OUT, ROOT, STORE, TZ, SNAPSHOT_END, sources
from experiment import FORECAST_FEATURES, PROBS
from rolling_models import BASE, predict_route
from study import inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduction():
    previous = ROOT/"results"
    keys = ["date","hour","cohort","scope","model"]
    def by_day(path):
        return pd.read_csv(path/"daily_predictions.csv").set_index(keys).sort_index()
    old,new = by_day(previous),by_day(OUT/"reproduction")
    pd.testing.assert_index_equal(old.index,new.index)
    np.testing.assert_array_equal(old.target_category,new.target_category)
    np.testing.assert_array_equal(old.predicted,new.predicted)
    index = ["hour","cohort","scope","model","n"]
    a,b = [pd.read_csv(path/"main_table.csv").set_index(index).sort_index() for path in
           (previous,OUT/"reproduction")]
    diff = (a-b).abs().max().to_dict()
    if diff["log_loss"]>.001 or diff["accuracy"]>0:
        raise ValueError(f"Reproduction outside tolerance: {diff}")
    return dict(rows=len(a),daily_records=len(old),max_absolute_difference=diff)


def audit_sources(ext):
    stamps = (pd.to_datetime(ext.date)+pd.to_timedelta(ext.hour,unit="h")).dt.tz_localize(TZ).dt.tz_convert("UTC")
    manifests = []
    for field,col in [("cycle_url","cycle_available"),("cloud_url","cloud_available"),("goes_url","goes_available")]:
        rows = ext.loc[ext[field].notna()]
        published = pd.to_datetime(rows[col],utc=True,format="mixed")
        if not published.le(stamps.loc[rows.index]).all():
            raise ValueError(f"Post-cutoff {field}")
        if field=="goes_url":
            end = pd.to_datetime(rows.goes_end,utc=True,format="mixed")
            if not ((end<=published)&(end>=stamps.loc[rows.index]-pd.Timedelta(minutes=30))).all():
                raise ValueError("Invalid GOES scan window")
        else:
            if not pd.to_datetime(rows.cycle_run,utc=True).le(published).all():
                raise ValueError("HRRR cycle created after publication record")
        for _,r in rows.drop_duplicates(field).iterrows():
            if field=="goes_url":
                cache = sources.CACHE/"cloud"/(Path(r[field]).stem+".json")
            else:
                variable = "TMP" if field=="cycle_url" else "TCDC"
                cache = sources.CACHE/"cycles"/f"{pd.Timestamp(r.cycle_run):%Y%m%d%H}-{variable}.json"
            manifests.append(dict(variable=field,url=r[field],available=r[col],point_cache_sha256=digest(cache)))
    checks = pd.read_csv(OUT/"upstream_hashes.csv.gz")
    manifest = pd.DataFrame(manifests).merge(checks,on="url",how="left",validate="many_to_one")
    if manifest.digest.isna().any():
        raise ValueError("Incomplete upstream hash manifest")
    if (pd.to_datetime(manifest.modified,utc=True)>pd.to_datetime(manifest.available,utc=True,format="mixed")).any():
        raise ValueError("An archive object was replaced after its recorded availability")
    manifest.to_csv(OUT/"external_sources.csv.gz",index=False)
    flags = ext[["latest_high","model_cloud","sat10","sat40"]].notna()
    flags["error"] = ext.errors.map(lambda x: bool(json.loads(x)) if isinstance(x,str) else bool(x))
    flags.groupby([pd.to_datetime(ext.date).dt.year.rename("year"),ext.hour]).agg(["sum","size"]).to_csv(
        OUT/"external_coverage.csv")
    return len(manifests)


def audit():
    ext = pd.read_parquet(STORE/"external.parquet")
    if ext.duplicated(["date","hour"]).any() or len(ext)!=len(pd.date_range("2021-01-01",SNAPSHOT_END))*3:
        raise ValueError("Incomplete archive date schedule")
    if ext.errors.map(lambda x: bool(json.loads(x)) if isinstance(x,str) else bool(x)).any():
        raise ValueError("Archive requests need retry")
    result = dict(reproduction=reproduction(),archive_issues=len(ext),external_sources=audit_sources(ext))
    summaries,leaders,features = [],[],[]
    for hour in HOURS:
        root = OUT/f"{hour:02}"
        metrics = pd.read_csv(root/"metrics.csv").assign(hour=hour)
        summaries.append(metrics)
        select = json.loads((root/"selection.json").read_text())
        ranks = []
        for ranking in root.glob("*/ranking.json"):
            ranks.extend(json.loads(ranking.read_text()))
        ranked = sorted(ranks,key=lambda item:item["loss"])
        winner = select["selected"]
        runner = next(item for item in ranked if item!=winner)
        for position,item in [("selected",winner),("best alternative",runner)]:
            per_fold = pd.read_csv(root/item["tier"]/"grid.csv")
            config = item["configs"][0] if len(item["configs"])==1 else json.dumps(item["configs"])
            scores = per_fold.loc[per_fold.config.eq(config)].set_index("fold").log_loss
            leaders.append(dict(hour=hour,position=position,**item,**{f"loss_{year}":loss for year,loss in scores.items()}))
        replay = pd.read_csv(root/"replay.csv.gz",parse_dates=["date"])
        p = replay[PROBS].to_numpy()
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
        if not np.isfinite(p).all() or (p<0).any():
            raise ValueError("Invalid archived probabilities")
        frame = inputs(hour)
        original = pd.read_csv(DATA/f"samples_{hour:02}.csv.gz",parse_dates=["date"],index_col="date")
        np.testing.assert_allclose(frame.loc[original.index,"target_max"],original.target_max,atol=1e-9)
        np.testing.assert_array_equal(frame.loc[original.index,"target_category"],original.target_category)
        legacy_features = [c for c in BASE if c in original]
        np.testing.assert_allclose(frame.loc[original.index,legacy_features],original[legacy_features],
                                   rtol=0,atol=1e-9,equal_nan=True)
        saved = joblib.load(root/"model.joblib")
        check = replay.loc[replay.delay.eq(0)&replay.model.eq("candidate")].set_index("date")
        actual,_ = predict_route(frame.loc[check.index],saved)
        np.testing.assert_allclose(actual,check[PROBS].to_numpy(),atol=1e-10)
        for m in saved["members"]:
            features.append(dict(hour=hour,family=m.family,params=m.params,input_columns=m.used))
        # Fixed HRRR has a separately hashed upstream tile and an explicit nine o'clock deadline.
        fixed = pd.read_csv(DATA/"hrrr_cycles.csv.gz",parse_dates=["date"]).query("status=='ok'")
        check_fixed = frame.loc[frame.index.intersection(fixed.date),FORECAST_FEATURES]
        if len(check_fixed.dropna())==0:
            raise ValueError("Fixed-cycle baseline is empty")
    pd.concat(summaries).to_csv(OUT/"all_metrics.csv",index=False)
    pd.DataFrame(leaders).to_csv(OUT/"model_selection.csv",index=False)
    result.update(station_snapshot_sha256=digest(STORE/"stations.parquet"),
                  external_snapshot_sha256=digest(STORE/"external.parquet"),deployed_features=features)
    (OUT/"audit.json").write_text(json.dumps(result,indent=2))
    print(result["reproduction"],result["archive_issues"],result["external_sources"])


if __name__=="__main__":
    audit()
