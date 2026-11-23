"""Fittable candidates and inference for the fixed-cutoff study."""
from itertools import product
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import STATION_IDS, THERMO, SKY
from experiment import BOUNDS, CLASSES, stabilize
from buoy_study import crossing_to_bands

BASE = ["season_sin","season_cos",*[f"{s}_{k}" for s in STATION_IDS for k in
        ("temp","rh","wind","max_so_far","change_3h","previous_max")],
        *[f"{s}_trend_{lag}h" for s in STATION_IDS for lag in (1,3)]]
GROUPS = {"base": [], "thermo": THERMO, "sky": SKY, "both": THERMO+SKY}
NWP = {"station": [], "fixed": ["hrrr_remaining_max","hrrr_warming"],
       "latest": ["latest_high","latest_warm"],
       "cloud": ["latest_high","latest_warm","model_cloud"],
       "sat": ["latest_high","latest_warm","sat10","sat40"],
       "fusion": ["latest_high","latest_warm","model_cloud","sat10","sat40"]}


def key(family, **params):
    return json.dumps([family,params],sort_keys=True,separators=(",",":"))


def search_grid():
    for c in (0.1,1,10):
        for family in ("logistic","threshold"):
            yield key(family,C=c)
    for depth,leaf in product((5,10,None),(2,5,10)):
        yield key("forest",max_depth=depth,min_samples_leaf=leaf,n_estimators=300)
    for depth,(rate,trees) in product((2,3,5),((0.03,300),(0.06,150),(0.04,200))):
        yield key("boost",max_depth=depth,learning_rate=rate,n_estimators=trees)
    for k in (15,40,80):
        yield key("knn",n_neighbors=k)
    for v in (1e-9,0.01,0.1):
        yield key("gaussian",var_smoothing=v)


def controls(hour):
    tree = key("boost",max_depth=3 if hour == 9 else 2,learning_rate=0.04,n_estimators=200)
    first = key("logistic",C=10) if hour == 9 else key("forest",max_depth=10,min_samples_leaf=4,n_estimators=200)
    return [first,tree]


def estimator(family, params):
    if family == "boost":
        return XGBClassifier(random_state=4260,n_jobs=2,subsample=.9,colsample_bytree=.9,
                             objective="multi:softprob",**params)
    if family == "forest":
        return make_pipeline(SimpleImputer(),RandomForestClassifier(random_state=4260,n_jobs=2,**params))
    if family == "knn":
        model = KNeighborsClassifier(weights="distance",**params)
    elif family == "gaussian":
        model = GaussianNB(**params)
    else:
        model = LogisticRegression(max_iter=1000,**params)
    return make_pipeline(SimpleImputer(),StandardScaler(),model)


def available(frame,tier):
    if tier == "station":
        return pd.Series(True,index=frame.index)
    return frame[NWP[tier]].notna().all(axis=1)


class Candidate:
    def __init__(self,config,group,tier):
        self.family,self.params = json.loads(config)
        self.columns = BASE+GROUPS[group]+NWP[tier]
        self.tier = tier
        self.model = estimator(self.family,self.params)

    def threshold_rows(self,x):
        blocks = []
        for limit in BOUNDS:
            b = x.copy()
            for c in ["LGA_temp","LGA_max_so_far","JFK_temp","EWR_temp","TEB_temp",
                      *NWP[self.tier][:1]]:
                b[c] = b[c]-limit
            blocks.append(b)
        return pd.concat(blocks,ignore_index=True)

    def fit(self,data):
        # Only dates from the training fold enter priors, imputers and model fitting.
        counts = pd.crosstab(data.index.month,data.target_category).reindex(
            index=range(1,13),columns=CLASSES,fill_value=0).to_numpy()+1
        self.priors = counts/counts.sum(axis=1,keepdims=True)
        self.used = [c for c in self.columns if data[c].notna().any()]
        x = data[self.used]
        if self.family == "threshold":
            blocks = self.threshold_rows(data[self.columns])
            y = np.concatenate([data.target_max.ge(limit) for limit in BOUNDS])
            uncertain = ~blocks.LGA_max_so_far.ge(0)
            self.model.fit(blocks.loc[uncertain,self.used],y[uncertain])
        else:
            self.model.fit(x,data.target_category)
        return self

    def predict(self,data):
        high = data.LGA_max_so_far.to_numpy()
        priors = self.priors[data.index.month-1]
        if self.family == "threshold":
            x = self.threshold_rows(data[self.columns])[self.used]
            cross = self.model.predict_proba(x)[:,1].reshape(len(BOUNDS),len(data)).T
            return crossing_to_bands(cross,high,priors)
        return stabilize(self.model.predict_proba(data[self.used]),priors,high)


def predict_route(frame,bundle):
    spec = bundle["spec"]
    mask = available(frame,spec["tier"])
    p = np.empty((len(frame),5))
    for subset,members in [(mask,bundle["members"]),(~mask,bundle["fallback"])]:
        if subset.any():
            p[subset] = np.mean([m.predict(frame.loc[subset]) for m in members],axis=0)
    return p, mask
