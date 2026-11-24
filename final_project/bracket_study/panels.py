"""Chronological panels, exact settlement boundaries and weather-free baselines."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"rolling"))
from data import TZ, STORE
from study import inputs
from market_data import MARKET, whole_f
from market_study import DEGREES, market_days
from nbm_archive import TABLE

OUT = ROOT/"results"/"bracket_study"
SCREEN = (("2025-01-01","2025-03-01"),("2025-03-01","2025-05-01"))
TUNE = (("2025-05-01","2025-07-01"),("2025-07-01","2025-09-01"))
FREEZE = "2025-09-01"
PERIODS = (("historical",FREEZE,"2026-09-01"),("autumn","2026-09-01","2026-11-07"))


@lru_cache(maxsize=1)
def contracts():
    g = pd.read_parquet(MARKET/"contracts.parquet")
    g["date"] = pd.to_datetime(g.date)
    for c in ("listed","closed"):
        g[c] = pd.to_datetime(g[c],utc=True,format="mixed",errors="coerce")
    return g


@lru_cache(maxsize=3)
def events(hour):
    frame = pd.DataFrame(index=pd.DatetimeIndex(contracts().date.unique()))
    return market_days(contracts(),hour,frame)


@lru_cache(maxsize=12)
def ground(hour,delay=0,development=False):
    f = inputs(hour,delay)
    labels = pd.read_csv(MARKET/"integer_labels.csv",index_col="date",parse_dates=["date"])
    dev = ROOT/"build"/"nbm_development.parquet"
    nbm = pd.read_parquet(dev if development and not TABLE.exists() else TABLE).loc[lambda d:d.hour.eq(hour)].copy()
    if development:
        f = f.loc[f.index<FREEZE]
    nbm["date"] = pd.to_datetime(nbm.date)
    known = nbm.status.eq("ok")
    issue = (nbm.date+pd.Timedelta(hours=hour)).dt.tz_localize(TZ)
    if (pd.to_datetime(nbm.loc[known,"published"],utc=True)>issue[known]).any():
        raise ValueError("late NBH record")
    f = f.join(labels[["all_f","routine_f",f"routine_floor_{hour}","hours","blocks"]],validate="one_to_one")
    f = f.join(nbm.set_index("date")[["nbm_high","nbm_peak_hours","nbm_spread","nbm_cloud"]],validate="one_to_one")
    f["nbm_disagreement"] = f.nbm_high-f.latest_high
    if delay:
        # Saved routine-floor rows are for nominal cutoffs. Rebuild earlier closed-hour maxima for delay tests.
        f[f"routine_floor_{hour}"] = delayed_routine_floor(hour).reindex(f.index)
    return f


@lru_cache(maxsize=3)
def delayed_routine_floor(hour):
    compact = ROOT/"data"/"bracket_delay_floors.parquet"
    if compact.exists():
        return pd.read_parquet(compact,columns=[str(hour)])[str(hour)]
    from prepare import DATA,lcd_local_time
    pieces = []
    for path in sorted((DATA/"raw").glob("LCD_USW00014732_*.csv")):
        raw = pd.read_csv(path,usecols=["DATE","REPORT_TYPE","HourlyDryBulbTemperature"],dtype="string")
        raw = raw.loc[raw.REPORT_TYPE.eq("FM-15")]
        time = pd.DatetimeIndex(lcd_local_time(raw.DATE))
        temp = pd.to_numeric(raw.HourlyDryBulbTemperature,errors="coerce").to_numpy(float,na_value=np.nan)
        p = pd.Series(np.where((temp>=-75)&(temp<=60),temp*1.8+32,np.nan),index=time)
        pieces.append(p.loc[p.index.hour<hour-1])
    if not pieces:
        raise FileNotFoundError("Rebuild delay_floors.parquet from the NOAA LCD station archives")
    series = pd.concat(pieces).sort_index()
    res = series.groupby(series.index.tz_localize(None).normalize()).max()
    return pd.Series(whole_f(res),index=res.index)


def target(frame,hour,kind):
    f = frame.copy(deep=False)
    if kind=="course":
        return f.assign(label_f=f.target_max,floor_f=f.LGA_max_so_far)
    floor = f[f"routine_floor_{hour}"] if kind=="routine_f" else whole_f(f.LGA_max_so_far)
    return f.assign(label_f=f[kind],floor_f=floor)


def history(frame,start,kind):
    ok = frame.index<pd.Timestamp(start)
    if kind!="course":
        ok &= frame.hours.ge(20)&frame.blocks.eq(4)
    train = frame.loc[ok & frame.label_f.notna()]
    if train.empty or train.index.max()>=pd.Timestamp(start):
        raise ValueError("invalid chronological training interval")
    return train


class Brackets:
    def __init__(self,ev):
        self.events = ev
        self.counts = ev.brackets.str.len().to_numpy()
        size = self.counts.max()
        self.cuts = np.empty((len(ev),size-1))
        for i,membership in enumerate(ev.bins):
            # An integer range starting at T places its cut at T-0.5 before nearest-degree rounding.
            edges = np.array([DEGREES[np.flatnonzero(row)[0]]-.5 for row in membership[1:]])
            self.cuts[i] = np.pad(edges,(0,size-1-len(edges)),mode="edge")

    def unpack(self,padded):
        result = []
        for row,n in zip(padded,self.counts):
            p = np.r_[row[:n-1],row[n-1:].sum()]
            # A small market-specific allowance for settlement/archive discrepancies.
            result.append(.999*p+.001/n)
        return result

    def score(self,probabilities):
        rows = []
        for (date,e),p in zip(self.events.iterrows(),probabilities):
            if not np.isfinite(p).all() or p.min()<0 or abs(p.sum()-1)>1e-7:
                raise ValueError(f"invalid market probabilities at {date}")
            y = int(e.winner)
            rows.append(dict(date=date,regime=e.regime,n_bins=len(p),winner=e.brackets[y],
                             predicted=e.brackets[np.argmax(p)],correct=int(np.argmax(p)==y),
                             log_loss=-np.log(p[y]),brier=float(p@p-2*p[y]+1)))
        return pd.DataFrame(rows).set_index("date")


def season_weights(train_index,date):
    phase = 2*np.pi*(train_index.dayofyear.to_numpy()-date.dayofyear)/365.25
    return np.exp((np.cos(phase)-1)/.115)


def climate(train,brackets):
    # Same calendar season on earlier dates only. Kernel smooths integer-degree observations.
    y = train.label_f.to_numpy()
    result = []
    for (date,event) in brackets.events.iterrows():
        weight = season_weights(train.index,date)
        mass = np.bincount(np.clip(y.astype(int)-DEGREES[0],0,len(DEGREES)-1),
                           weights=weight,minlength=len(DEGREES))+.002
        p = event.bins@(mass/mass.sum())
        result.append(p)
    return result


def listing_only(past,ev):
    results = []
    for event in ev.itertuples():
        n = len(event.brackets)
        idx = np.arange(n)
        counts = np.exp(-.5*((idx-(n-1)/2)/1.6)**2)+.1
        for old in past.itertuples():
            relative = (old.winner+.5)/len(old.brackets)
            counts[min(n-1,int(relative*n))] += 1
        results.append(counts/counts.sum())
    return results
