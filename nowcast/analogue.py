"""Temperature-band prior from LGA days preceding all model-fitting dates."""
import numpy as np
import pandas as pd
from features import BOUNDS, labels

PRIOR = [f"analogue_p{i}" for i in range(5)]


def build_reference(station):
    table = station[station.index.year < 2021]
    targets = labels(table)
    groups = {}
    for day, obs in table.groupby(table.index.date):
        if day not in targets.index:
            continue
        hourly = obs.groupby(obs.index.tz_convert("UTC").floor("h")).tail(1)
        complete = hourly.temp.expanding().max().to_numpy()
        clock = hourly.index.tz_convert("UTC").floor("h")
        for slot in range(57):
            at = (pd.Timestamp(day)+pd.Timedelta(hours=6,minutes=slot*15)).tz_localize("America/New_York")
            i = clock.searchsorted((at-pd.Timedelta(minutes=15)).floor("h"),side="left")-1
            if i >= 0 and np.isfinite(complete[i]):
                groups.setdefault(slot,[]).append([day.month,complete[i],targets.loc[day]-complete[i]])
    return {slot:np.asarray(rows) for slot,rows in groups.items()}


def augment(row, at, reference):
    slot = (at.hour-6)*4+at.minute//15
    month,high,rise = reference[slot].T
    difference = np.abs(at.month-month)
    weights = np.exp(-.5*((np.minimum(difference,12-difference)/1.5)**2
                           +((row["high_so_far"]-high)/5)**2))
    classes = np.searchsorted(BOUNDS,row["high_so_far"]+rise,side="right")
    counts = np.bincount(classes,weights=weights,minlength=5)+.01
    counts[np.arange(5)<np.searchsorted(BOUNDS,row["high_so_far"],side="right")] = 0
    probabilities = counts/counts.sum()
    return {**row,**dict(zip(PRIOR,probabilities)), "expected_rise":np.average(rise,weights=weights)}

