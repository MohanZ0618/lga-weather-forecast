"""Conditional temperature distributions shared by variable market bins and course bands."""
import numpy as np
from catboost import CatBoostRegressor
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 4260
FAMILIES = ("threshold", "monotone_xgb", "gaussian_cat", "hurdle", "blend")
TIERS = ("station", "hrrr", "nbm")
DEFAULTS = dict(C=1., depth=4, rate=.05, trees=400, l2=3.)
BASE = ["season_sin","season_cos","LGA_temp","LGA_rh","LGA_trend_1h","LGA_trend_3h",
        "LGA_spread","LGA_east","LGA_north","LGA_sky","JFK_temp","EWR_temp","TEB_temp",
        "LGA_previous_max"]
ADDITIONS = {"station": [], "hrrr": ["latest_high","latest_warm","model_cloud"],
             "nbm": ["latest_high","latest_warm","model_cloud","nbm_high","nbm_peak_hours",
                     "nbm_spread","nbm_cloud","nbm_disagreement"]}


def defaults(family):
    return dict(DEFAULTS,depth=3,trees=300,l2=1.) if family=="monotone_xgb" else DEFAULTS.copy()


def present(frame,tier):
    if tier=="station":
        return np.ones(len(frame),bool)
    ok = frame.latest_high.notna() & frame.latest_warm.notna()
    if tier=="nbm":
        ok &= frame.nbm_high.notna()
    return ok.to_numpy()


def features(frame,tier):
    floor = frame.floor_f.to_numpy(float)
    # The anchor is observable at issue time. Richer sources refine its expected remaining rise.
    anchor = frame.floor_f.fillna(frame.LGA_temp).fillna(frame.LGA_previous_max).fillna(55).to_numpy(float)
    if tier!="station":
        forecast = frame.latest_high.to_numpy()
        if tier=="nbm":
            forecast = (forecast+frame.nbm_high.to_numpy())/2
        anchor = np.fmax(anchor,forecast)
    x = frame[BASE+ADDITIONS[tier]].to_numpy(float)
    return np.column_stack([x, floor-anchor]),anchor,floor


def threshold_rows(x,anchor,cuts):
    # Only this single coordinate varies with a threshold. Its XGBoost constraint guarantees ordering.
    return np.column_stack([(anchor[:,None]-cuts).ravel(),np.repeat(x,cuts.shape[1],axis=0)])


def linear(c):
    return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),
                         LogisticRegression(C=c,max_iter=800))


class Distribution:
    def __init__(self,family,tier,params):
        self.family,self.tier,self.params = family,tier,params.copy()

    def design(self,frame):
        return features(frame,self.tier)

    def fit(self,frame):
        if self.family=="blend":
            self.members = [type(self)(f,self.tier,self.params).fit(frame) for f in ("threshold","hurdle")]
            return self
        x,anchor,floor = self.design(frame)
        y = frame.label_f.to_numpy(float)
        p = self.params
        if self.family in ("threshold","monotone_xgb"):
            # The forecast anchor defines training thresholds independently of the outcome.
            cuts = np.floor(anchor[:,None])+np.arange(-14,15,2)[None,:]+.5
            cuts = np.column_stack([cuts,np.tile([40.,55.,70.,85.],(len(y),1))])
            ok = ~(floor[:,None]>=cuts)
            stacked = threshold_rows(x,anchor,cuts)[ok.ravel()]
            target = (y[:,None]>=cuts)[ok]
            if self.family=="threshold":
                self.model = linear(p["C"])
            else:
                self.model = XGBClassifier(n_jobs=2,random_state=SEED,tree_method="hist",
                    max_depth=p["depth"],learning_rate=p["rate"],n_estimators=p["trees"],
                    reg_lambda=p["l2"],monotone_constraints=(1,)+(0,)*x.shape[1])
            self.model.fit(stacked,target)
        else:
            self.model = CatBoostRegressor(loss_function="RMSEWithUncertainty",depth=p["depth"],
                iterations=p["trees"],learning_rate=p["rate"],l2_leaf_reg=p["l2"],
                random_seed=SEED,thread_count=2,verbose=False,allow_writing_files=False)
            if self.family=="hurdle":
                valid = np.isfinite(floor)
                warm = y>floor+1e-6
                self.gate = linear(p["C"]).fit(x[valid],warm[valid])
                self.model.fit(x[valid & warm],np.log((y-floor)[valid & warm]))
            else:
                self.model.fit(x,y-anchor)
        return self

    def survival(self,frame,cuts):
        if self.family=="blend":
            return np.mean([m.survival(frame,cuts) for m in self.members],axis=0)
        x,anchor,floor = self.design(frame)
        if self.family in ("threshold","monotone_xgb"):
            prob = self.model.predict_proba(threshold_rows(x,anchor,cuts))[:,1].reshape(cuts.shape)
        else:
            raw = self.model.predict(x,prediction_type="RawFormulaVal",thread_count=2)
            mu = raw[:,0,None]
            sigma = np.exp(np.clip(raw[:,1,None],-10,5))
            if self.family=="gaussian_cat":
                prob = ndtr((mu+anchor[:,None]-cuts)/np.maximum(.65,sigma))
            else:
                base = np.where(np.isfinite(floor),floor,anchor)[:,None]
                distance = np.maximum(cuts-base,1e-8)
                prob = self.gate.predict_proba(x)[:,1,None]*ndtr((mu-np.log(distance))/np.maximum(.25,sigma))
        return np.where(floor[:,None]>=cuts,1.,np.minimum.accumulate(prob,axis=1))


def band_probs(survival):
    return np.diff(np.pad(1-survival,((0,0),(1,1)),constant_values=(0,1)),axis=1).clip(0,1)


class Routed:
    """Richer sources use the selected model. Missing sources use fixed lower-tier logistic fits."""
    candidate_type = Distribution
    def __init__(self,family,tier,params):
        self.family,self.tier,self.params = family,tier,params

    def fit(self,train,fallbacks=None):
        self.models = {}
        for tier in TIERS[:TIERS.index(self.tier)+1]:
            rich = tier==self.tier
            if not rich and fallbacks is not None:
                self.models[tier] = fallbacks[tier]
            else:
                ok = present(train,tier)
                self.models[tier] = self.candidate_type(self.family if rich else "threshold",tier,
                    self.params if rich else defaults("threshold")).fit(train.loc[ok])
        return self

    def predict(self,frame,cuts):
        routing = np.full(len(frame),"station",dtype=object)
        for tier in self.models:
            routing[present(frame,tier)] = tier
        out = np.empty((len(frame),cuts.shape[1]+1))
        for tier,model in self.models.items():
            ok = routing==tier
            if ok.any():
                out[ok] = band_probs(model.survival(frame.loc[ok],cuts[ok]))
        return out,routing
