"""Probability models and validation-only model selection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import BOUNDS, NWP, SAT_LOCAL, SAT_MOTION
from analogue import PRIOR

LEVELS = ("station", "fixed", "latest", "local_cloud", "upwind_cloud")
FAMILIES = ("analogue", "logistic", "forest", "boost", "warming", "threshold")
GRID = np.array([.02,.10,.25,.50,.75,.90,.98])


def predictors(frame, level):
    skip = {"cutoff","day","actual","label","direction","delay","hour_band","weight"}
    base = [c for c in frame if c not in skip and c not in NWP+SAT_LOCAL+SAT_MOTION
            and not c.startswith("fixed_")]
    if level != "station":
        base += NWP
    if level in LEVELS[-2:]:
        base += SAT_LOCAL
    if level == "upwind_cloud":
        base += SAT_MOTION
    return base


def physical(p, high, temp=1.0):
    p = np.maximum(p,1e-5)**(1/temp)
    floor = np.searchsorted(BOUNDS,np.nan_to_num(high,nan=-np.inf),side="right")
    p[np.arange(5)[None,:] < floor[:,None]] = 0
    return p/p.sum(axis=1,keepdims=True)


def date_weights(frame):
    w = 1/frame.groupby("day").day.transform("size").to_numpy()
    return w/w.mean()


def score(frame, p):
    y = frame.label.to_numpy(dtype=int)
    w = date_weights(frame)
    return {"accuracy":np.average(p.argmax(axis=1)==y,weights=w),
            "log_loss":np.average(-np.log(p[np.arange(len(y)),y].clip(1e-15)),weights=w),
            "brier":np.average(np.sum((p-np.eye(5)[y])**2,axis=1),weights=w),
            "days":frame.day.nunique(), "rows":len(y)}


class Family:
    def __init__(self, name):
        self.name = name

    def fit(self, X, frame):
        weights = date_weights(frame)
        y = frame.label.to_numpy()
        if self.name == "analogue":
            return self
        if self.name == "threshold":
            self.model = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                StandardScaler(),LogisticRegression(C=.15,max_iter=700))
            self.model.fit(self.threshold_inputs(X),np.concatenate([frame.actual>=b for b in BOUNDS]),
                           logisticregression__sample_weight=np.tile(weights,4))
        elif self.name == "warming":
            self.models = []
            rise = (frame.actual-frame.high_so_far).clip(lower=0)
            for q in GRID:
                reg = HistGradientBoostingRegressor(loss="quantile", quantile=q,max_iter=75,
                    max_leaf_nodes=9, min_samples_leaf=14, l2_regularization=2,random_state=4260)
                self.models.append(reg.fit(X,rise,sample_weight=weights))
        elif self.name == "logistic":
            self.model = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                StandardScaler(),LogisticRegression(C=0.4,max_iter=700))
            self.model.fit(X,y,logisticregression__sample_weight=weights)
        else:
            self.model = (RandomForestClassifier(n_estimators=160,min_samples_leaf=3,max_features=0.8,
                          random_state=4260,n_jobs=2) if self.name=="forest" else
                          XGBClassifier(n_estimators=130,max_depth=2,learning_rate=.06,n_jobs=2,
                          random_state=4260,objective="multi:softprob"))
            self.model.fit(X,y,sample_weight=weights)
        return self

    @staticmethod
    def threshold_inputs(X):
        rows = []
        for boundary in BOUNDS:
            cols = {"remaining_margin":boundary-X.high_so_far, "current_margin":boundary-X.LGA_temp,
                    "hour":X.hour,"sun":X.sun,"trend":X.LGA_trend, "cover":X.LGA_cover,
                    "headroom_time":(boundary-X.high_so_far)/(22-X.hour),
                    "dew_spread":X.LGA_temp-X.LGA_dew, "coastal_gradient":X.JFK_minus_LGA}
            if "nwp_high" in X:
                cols.update(forecast_margin=boundary-X.nwp_high, bias=X.nwp_bias, lead=X.peak_hours)
            for col in SAT_LOCAL+SAT_MOTION:
                if col in X:
                    cols[col] = X[col]
            rows.append(pd.DataFrame(cols))
        return pd.concat(rows)

    def predict(self, X, high):
        if self.name == "analogue":
            return X[PRIOR].to_numpy()
        if self.name == "threshold":
            exceed = self.model.predict_proba(self.threshold_inputs(X))[:,1].reshape(4,len(X)).T
            cdf = 1-np.minimum.accumulate(exceed,axis=1)
            p = np.diff(np.column_stack([np.zeros(len(X)),cdf,np.ones(len(X))]),axis=1)
            return physical(p,high)
        if self.name != "warming":
            p = np.zeros((len(X),5))
            p[:,self.model.classes_.astype(int)] = self.model.predict_proba(X)
            return physical(p,high)
        quantiles = np.maximum(0,np.sort(np.array([m.predict(X) for m in self.models]).T,axis=1))
        p = []
        for row,h in zip(quantiles,high):
            # Small temperature jitter handles ties in rounded ASOS values.
            values = h+row+np.arange(len(row))*1e-6
            cdf = np.interp(BOUNDS-1e-7,values,GRID,left=.005,right=.995)
            p.append(np.diff(np.r_[0,cdf,1]))
        return physical(np.asarray(p),high)


@dataclass
class Bundle:
    columns: list
    fitted: dict
    weights: dict
    temperature: float

    def predict(self, frame):
        high = frame.high_so_far.to_numpy()
        summed = sum(weight*self.fitted[name].predict(frame[self.columns],high)
                     for name,weight in self.weights.items())
        return physical(summed,high,self.temperature)


def temperature(p, frame):
    result = minimize_scalar(lambda t:score(frame,physical(p,frame.high_so_far.to_numpy(),t))["log_loss"],
                             bounds=(.7,1.6),method="bounded")
    return float(result.x)


def choose(scores, baseline_loss):
    eligible = [key for key,item in scores.items() if item["log_loss"] <= baseline_loss*1.01]
    if eligible:
        return min(eligible,key=lambda key:(-scores[key]["accuracy"], scores[key]["log_loss"]))
    return min(scores,key=lambda key:scores[key]["log_loss"])


def fit_level(frame, level, validation_start="2025-01", validation_end="2025-09"):
    if not "2025-01" <= validation_start < validation_end:
        raise ValueError("Validation must follow the 2023 and 2024 calibration folds")
    cols = predictors(frame,level)
    folds, predictions = [],{n:[] for n in FAMILIES}
    for start,end in [("2023-01","2024-01"),("2024-01","2025-01")]:
        train = frame.loc[frame.day < start]
        valid = frame.loc[(frame.day>=start)&(frame.day<end)]
        if len(train)<30 or valid.empty:
            raise ValueError(f"Insufficient chronological training data for {level}, {start}")
        for name in FAMILIES:
            model = Family(name).fit(train[cols],train)
            predictions[name].append(model.predict(valid[cols],valid.high_so_far.to_numpy()))
        folds.append(valid)
    oof = pd.concat(folds)
    p = {n:np.vstack(v) for n,v in predictions.items()}
    ranked = sorted(FAMILIES,key=lambda n:score(oof,p[n])["log_loss"])
    mixtures = {n:{n:1.} for n in FAMILIES}
    a,b = ranked[:2]
    for w in (.25,.5,.75):
        mixtures[f"{a}:{w}+{b}"] = {a:w,b:1-w}
    temperatures = {name:temperature(sum(p[n]*w for n,w in weights.items()),oof)
                    for name,weights in mixtures.items()}
    train = frame.loc[frame.day < validation_start]
    valid = frame.loc[(frame.day >= validation_start) & (frame.day < validation_end)]
    if valid.empty:
        raise ValueError(f"No validation dates for {level}")
    models = {n:Family(n).fit(train[cols],train) for n in FAMILIES}
    candidates, scores = {},{}
    for name,weights in mixtures.items():
        for t in [1.,temperatures[name]]:
            key = name if t == 1 else name+" calibrated"
            bundle = Bundle(cols,{n:models[n] for n in weights},weights,t)
            candidates[key] = bundle
            scores[key] = score(valid,bundle.predict(valid))
    selection = choose(scores,scores["logistic"]["log_loss"])
    return candidates[selection], selection, scores

