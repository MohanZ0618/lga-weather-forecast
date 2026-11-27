"""Extend the observation samples back to 2010 with the paper's own LCD pipeline (prepare.py).

Downloads LCD v2 files for 2010-2018 (four stations), builds samples for 2010-01-02 to 2017-12-31 with
prepare.station_features and the same daily label rule, and rebuilds 2018 as a check: those rows must equal
the bundled samples_HH.csv.gz exactly. Writes data/history/samples_HH_2010_2017.csv.gz, the source hashes
and the overlap check. The bundled 2018-2026 samples are not modified.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from prepare import DATA, HOURS, STATION_IDS, daily_labels, fetch_station, hourly_rows, previous_day_maxima, station_features

YEARS = range(2010, 2019)
OUT = DATA / "history"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [(station, year) for station in STATION_IDS for year in YEARS]
    stations, sources = {station: [] for station in STATION_IDS}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for station, frame, meta in pool.map(fetch_station, keys):
            stations[station].append(frame)
            sources.append(meta)
            print(f"{station} {meta['source_year']} {len(frame)} reports", flush=True)
    pd.DataFrame(sources).to_csv(OUT / "lcd_sources.csv", index=False)
    stations = {s: pd.concat(t).sort_index() for s, t in stations.items()}
    stations = {s: t[~t.index.duplicated(keep="last")] for s, t in stations.items()}
    maxima = {s: previous_day_maxima(t) for s, t in stations.items()}
    labels = daily_labels(hourly_rows(stations["LGA"])).loc["2010-01-02":"2018-12-31"]
    checks = []
    for hour in HOURS:
        rows = [station_features(stations, maxima, day, hour) for day in labels.index]
        samples = pd.DataFrame(rows, index=labels.index).join(labels)
        samples.index.name = "date"
        samples.loc[:"2017-12-31"].to_csv(OUT / f"samples_{hour:02}_2010_2017.csv.gz",
                                          compression={"method": "gzip", "mtime": 0})
        bundled = pd.read_csv(DATA / f"samples_{hour:02}.csv.gz", parse_dates=["date"], index_col="date")
        mine = samples.loc["2018-01-02":"2018-12-31"]
        theirs = bundled.loc["2018-01-02":"2018-12-31"]
        same_dates = mine.index.equals(theirs.index)
        common = theirs.columns.intersection(mine.columns)
        values = np.isclose(mine.loc[theirs.index, common].astype("Float64").to_numpy(float, na_value=np.nan),
                            theirs[common].astype("Float64").to_numpy(float, na_value=np.nan),
                            rtol=0, atol=1e-9, equal_nan=True) if same_dates else np.array([[False]])
        checks.append({"hour": hour, "dates_2018": len(theirs), "same_dates": same_dates,
                       "columns_compared": len(common), "cells_equal": float(values.mean()),
                       "early_dates": int((samples.index <= "2017-12-31").sum())})
        print(checks[-1], flush=True)
    pd.DataFrame(checks).to_csv(OUT / "overlap_check_2018.csv", index=False)


if __name__ == "__main__":
    main()
