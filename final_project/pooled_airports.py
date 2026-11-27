"""Does training on nearby airports help the LGA forecast? JFK, EWR and TEB as extra training targets.

The selected model (censored Gaussian of the remaining rise, guidance_study.py) is bias-limited: its 2024
validation loss is flat from 400 to 1,040 training days. This study asks whether rows from three nearby
airports help where more LGA rows did not. Every target airport A gets the same treatment as LGA:
  labels    A's own final hourly-observation maximum (prepare.daily_labels on prepare.hourly_rows)
  features  prepare.station_features on the four-station dict, re-centered on A: "own_*" are A's columns,
            "n1_*", "n2_*", "n3_*" the other three airports ordered by distance from A (nearest first),
            plus season terms and A's own station guidance (guidance.build run for A's ICAO code)
For LGA the re-centered columns are a permutation of the CURRENT-without-HRRR inputs, so the LGA-only arm
is that model. HRRR is left out (its cache is LGA-only); CURRENT itself is read from its saved test file.

Steps (python final_project/pooled_airports.py STEP):
  fetch  LCD v2 2020-2026 for the four stations with prepare.fetch_station, one file at a time, and the IEM
         MOS LAV and NBS archives for KLGA, KJFK, KEWR and KTEB, 2020-2026 (guidance.py's endpoint: one
         request per model-year, sequential, 3 s apart, retries), cached under data/raw/ (gitignored).
  build  fetch (cached files are only hashed), then the compact tables under data/pooled/ and two checks:
         rebuilt LGA rows against whatever data/samples_HH.csv.gz holds, rebuilt KLGA guidance against
         data/guidance_features.csv.gz.
  run    (default; offline, about a minute on 2 threads) fits the arms and writes results/pooled_airports/.
  all    build, then run.

Protocol: the LGA cohort is guidance_study.py's (bundled LGA dates with HRRR and complete LAMP/NBM), so the
test days are CURRENT's 359. Pooled rows are the other airports on the same training dates. Training ends
2023-12-31; L2, the LGA weight and the reported selection use 2024 LGA validation only; test scores LGA only.
The 1% climatology mix uses the LGA training months in every arm, so arms differ only in the fitted model.
"""

import argparse
import contextlib
import gzip
import hashlib
import io
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from threadpoolctl import threadpool_limits

import guidance
import shash
from distribution import GUIDANCE, band_cdf_edges, design, to_bands
from experiment import BOUNDS, PROBS, month_prior, season_features, split_dates, stabilize
from prepare import (DATA, END, HOURS, daily_labels, fetch_station, hourly_rows, previous_day_maxima,
                     station_features)
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
POOLED = DATA / "pooled"
OUT = ROOT / "results" / "pooled_airports"
GZIP = {"method": "gzip", "mtime": 0}
AIRPORTS = ("LGA", "JFK", "EWR", "TEB")
ICAO = {airport: f"K{airport}" for airport in AIRPORTS}
LCD_YEARS = range(2020, 2027)
MOS_YEARS = range(2020, 2027)
FIRST_DATE = "2021-01-01"  # earliest training date; 2020 files only supply lags and run lookback
# Airport reference points (FAA), used only to order the neighbour blocks.
COORDS = {"LGA": (40.7769, -73.8740), "JFK": (40.6398, -73.7789),
          "EWR": (40.6925, -74.1687), "TEB": (40.8501, -74.0608)}
FEATURES = ("temp", "rh", "wind", "max_so_far", "change_3h", "previous_max", "trend_1h", "trend_3h")
REQUIRED = ["lamp_remaining_max", "lamp_now", "lamp_cloud", "lamp_u", "lamp_v", "nbm_max", "nbm_max_sd"]
BLOCKS = ("own", "n1", "n2", "n3")
COLUMNS = ["season_sin", "season_cos", *(f"{b}_{f}" for b in BLOCKS for f in FEATURES), "rel_now", *GUIDANCE]
INDICATORS = [f"is_{airport}" for airport in AIRPORTS[1:]]
L2 = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
LGA_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
SEEDS = range(5)
HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)
CURRENT = "Censored Gaussian, HRRR + LAMP/NBM"
SAVED_NO_HRRR = "Censored Gaussian, LAMP/NBM"
ARMS = {"a": "LGA only", "b": "Pooled", "c": "Pooled + airport shifts", "d": "Pooled + airport shifts + LGA weight"}


def distance_km(a, b):
    (la1, lo1), (la2, lo2) = np.radians(COORDS[a]), np.radians(COORDS[b])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def neighbours(airport):
    return sorted((other for other in AIRPORTS if other != airport), key=lambda other: distance_km(airport, other))


# Fetch ---------------------------------------------------------------------------------------

def load_stations():
    """prepare.main's loading: fetch_station per file (downloads if missing), concatenate, drop duplicates."""
    stations, records = {airport: [] for airport in AIRPORTS}, []
    for airport in AIRPORTS:
        for year in LCD_YEARS:
            station, frame, meta = fetch_station((airport, year))  # one file at a time
            stations[station].append(frame)
            records.append(meta)
    stations = {s: pd.concat(tables).sort_index() for s, tables in stations.items()}
    stations = {s: table[~table.index.duplicated(keep="last")] for s, table in stations.items()}
    return stations, pd.DataFrame(records)


def fetch_mos(end="2026-09-01T00:00Z"):
    """guidance.fetch for any station: same endpoint, one request per model-year, sequential, 3 s apart."""
    records = []
    last = int(end[:4])
    for airport in AIRPORTS:
        station = ICAO[airport]
        for model in ("LAV", "NBS"):
            for year in MOS_YEARS:
                sts, ets = f"{year}-01-01T00:00Z", (f"{year + 1}-01-01T00:00Z" if year < last else end)
                query = urllib.parse.urlencode(dict(station=station, model=model, sts=sts, ets=ets, format="csv"))
                url = f"{guidance.URL}?{query}"
                path = guidance.RAW / model / f"{station}-{year}.csv"
                if not path.exists():
                    body = None
                    for attempt in range(5):
                        try:
                            body = urllib.request.urlopen(url, timeout=300).read()
                            break
                        except OSError:
                            time.sleep(15 * (attempt + 1))
                    if body is None:
                        raise RuntimeError(f"IEM request failed: {url}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.with_suffix(".part").write_bytes(body)
                    path.with_suffix(".part").replace(path)
                    time.sleep(3)
                body = path.read_bytes()
                rows = pd.read_csv(io.BytesIO(body), usecols=["runtime", "station"])
                records.append({"station": station, "model": model, "year": year, "url": url, "rows": len(rows),
                                "stations_in_file": ";".join(sorted(rows.station.dropna().unique())),
                                "first_run": rows.runtime.min() if len(rows) else "",
                                "last_run": rows.runtime.max() if len(rows) else "",
                                "sha256": hashlib.sha256(body).hexdigest()})
                print(f"MOS {station} {model} {year} {len(rows)} rows", flush=True)
    return pd.DataFrame(records)


def fetch():
    POOLED.mkdir(parents=True, exist_ok=True)
    stations, lcd = load_stations()
    bundled = pd.read_csv(DATA / "station_sources.csv")
    bundled = bundled.set_index(["station", "source_year"])["sha256"]
    lcd["matches_station_sources"] = [bundled.get((s, y)) == h for s, y, h in zip(lcd.station, lcd.source_year, lcd.sha256)]
    lcd.to_csv(POOLED / "lcd_sources.csv", index=False)
    print(f"LCD: {len(lcd)} files, {int(lcd.matches_station_sources.sum())} identical to station_sources.csv", flush=True)
    fetch_mos().to_csv(POOLED / "mos_sources.csv", index=False)
    return stations


# Build ---------------------------------------------------------------------------------------

@contextlib.contextmanager
def guidance_station(code):
    """guidance.py reads one station (module constant STATION); point it at another for one build."""
    saved = guidance.STATION
    guidance.STATION = code
    try:
        yield
    finally:
        guidance.STATION = saved


def cells_equal(mine, theirs):
    common = theirs.columns.intersection(mine.columns)
    same = mine.index.equals(theirs.index)
    if not same:
        return same, len(common), 0.0
    values = np.isclose(mine[common].astype("Float64").to_numpy(float, na_value=np.nan),
                        theirs[common].astype("Float64").to_numpy(float, na_value=np.nan),
                        rtol=0, atol=1e-9, equal_nan=True)
    return same, len(common), float(values.mean())


def build_guidance():
    """guidance.build for each airport's ICAO code; KLGA is compared with the bundled feature table."""
    frames, check = [], {}
    with tempfile.TemporaryDirectory() as folder:
        for airport in AIRPORTS:
            path = Path(folder) / f"{airport}.csv.gz"
            with guidance_station(ICAO[airport]), contextlib.redirect_stdout(io.StringIO()):
                guidance.build(0, path)
            table = pd.read_csv(path, parse_dates=["date"])
            if airport == "LGA":
                bundled = DATA / "guidance_features.csv.gz"
                check["klga_text_identical"] = gzip.decompress(path.read_bytes()) == gzip.decompress(bundled.read_bytes())
                theirs = pd.read_csv(bundled, parse_dates=["date"]).set_index(["date", "hour"])
                numeric = theirs.columns.difference(["lamp_run", "nbm_run"])
                _, check["klga_columns"], check["klga_cells_equal"] = cells_equal(
                    table.set_index(["date", "hour"])[numeric], theirs[numeric])
            frames.append(table.loc[table.date >= FIRST_DATE].assign(airport=airport))
            print(f"guidance {ICAO[airport]}: {int(table.lamp_remaining_max.notna().sum())} LAMP and "
                  f"{int(table.nbm_max.notna().sum())} NBM cutoffs", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "airport", table.pop("airport"))
    return table, check


def build():
    stations = fetch()
    maxima = {s: previous_day_maxima(table) for s, table in stations.items()}
    labels = {a: daily_labels(hourly_rows(stations[a])).loc[FIRST_DATE:END] for a in AIRPORTS}
    long = pd.concat([table.assign(airport=a) for a, table in labels.items()]).reset_index()
    long[["airport", "date", "valid_hours", "covered_periods", "target_max", "target_category"]].to_csv(
        POOLED / "labels.csv.gz", index=False, compression=GZIP)
    dates = pd.DatetimeIndex(sorted(set().union(*(table.index for table in labels.values()))), name="date")
    checks = []
    for hour in HOURS:
        table = pd.DataFrame([station_features(stations, maxima, day, hour) for day in dates], index=dates)
        table.to_csv(POOLED / f"stations_{hour:02}.csv.gz", compression=GZIP)
        # The LGA rows rebuilt here against whatever the bundled samples hold now.
        mine = table.loc[labels["LGA"].index].join(labels["LGA"])
        theirs = pd.read_csv(DATA / f"samples_{hour:02}.csv.gz", parse_dates=["date"], index_col="date")
        theirs = theirs.loc[FIRST_DATE:END]
        same, columns, equal = cells_equal(mine, theirs)
        checks.append({"hour": hour, "dates": len(theirs), "same_dates": same, "same_columns":
                       set(mine.columns) == set(theirs.columns), "columns_compared": columns, "cells_equal": equal})
        print(checks[-1], flush=True)
    table, guidance_check = build_guidance()
    table.to_csv(POOLED / "guidance.csv.gz", index=False, compression=GZIP)
    checks = pd.DataFrame(checks).assign(**guidance_check)
    checks.to_csv(POOLED / "overlap_check_lga.csv", index=False)
    print(checks.to_string(index=False), flush=True)


# Model ---------------------------------------------------------------------------------------

def load_pooled(folder=POOLED):
    stations = {h: pd.read_csv(folder / f"stations_{h:02}.csv.gz", parse_dates=["date"], index_col="date")
                for h in HOURS}
    labels = pd.read_csv(folder / "labels.csv.gz", parse_dates=["date"])
    table = pd.read_csv(folder / "guidance.csv.gz", parse_dates=["date"])
    return stations, labels, table


def target_frame(airport, hour, stations, labels, guidance_table):
    """One row per date with a label at `airport`: its own columns, its neighbours', its own guidance."""
    label = labels.loc[labels.airport.eq(airport)].set_index("date")[["target_max", "target_category"]]
    table = stations[hour].loc[label.index]
    blocks = {"own": airport, **{f"n{i}": other for i, other in enumerate(neighbours(airport), 1)}}
    frame = pd.DataFrame({f"{b}_{f}": table[f"{s}_{f}"] for b, s in blocks.items() for f in FEATURES},
                         index=label.index).join(label)
    mos = guidance_table.loc[guidance_table.airport.eq(airport) & guidance_table.hour.eq(hour)]
    frame = season_features(frame.join(mos.set_index("date")[REQUIRED], how="left"))
    # distribution.add_relative, with the target airport in place of LGA.
    return frame.assign(rel_now=frame.own_temp - frame.own_max_so_far,
                        lamp_now_error=frame.own_temp - frame.lamp_now,
                        rel_lamp=frame.lamp_remaining_max - frame.own_max_so_far,
                        rel_nbm=frame.nbm_max - frame.own_max_so_far,
                        airport=airport, **{f"is_{a}": float(a == airport) for a in AIRPORTS[1:]})


def assemble(hour, stations, labels, guidance_table, hrrr_dates):
    """LGA rows of guidance_study's cohort, plus the other airports on the same training dates."""
    lga = target_frame("LGA", hour, stations, labels, guidance_table)
    lga = lga.loc[lga.index.isin(hrrr_dates) & lga[REQUIRED].notna().all(axis=1)]
    splits = split_dates(lga.index)
    role = np.select([splits["train"], splits["validation"], splits["test"]], ["train", "validation", "test"], "")
    parts, coverage = [lga.assign(role=role).loc[role != ""]], []
    train_dates = lga.index[splits["train"]]
    coverage.append({"hour": hour, "airport": "LGA", "train": int(splits["train"].sum()),
                     "validation": int(splits["validation"].sum()), "test": int(splits["test"].sum())})
    for airport in AIRPORTS[1:]:
        frame = target_frame(airport, hour, stations, labels, guidance_table)
        on_dates = frame.index.isin(train_dates)
        usable = on_dates & frame[REQUIRED].notna().all(axis=1) & frame.own_max_so_far.notna()
        parts.append(frame.loc[usable].assign(role="train"))
        coverage.append({"hour": hour, "airport": airport, "train": int(usable.sum()),
                         "dropped_on_train_dates": int(len(train_dates) - usable.sum())})
    return pd.concat(parts), coverage


def weighted_loss_and_grad(theta, x, z, rise, l2, weight):
    """Censored Gaussian negative log likelihood with row weights (mean 1); shash.py's loss at weight 1."""
    n, k = x.shape
    beta = theta[1:k + 1]
    mu = theta[0] + x @ beta
    log_sigma = z @ theta[k + 1:]
    sigma = np.exp(log_sigma)
    zero = rise <= 1e-9
    t, s = (rise - mu) / sigma, -mu / sigma
    log_cdf = norm.logcdf(s)
    ll = np.where(zero, log_cdf, -0.5 * t * t - log_sigma - HALF_LOG_2PI)
    ratio = np.exp(norm.logpdf(s) - log_cdf)
    d_mu = weight * np.where(zero, -ratio / sigma, t / sigma)
    d_log_sigma = weight * np.where(zero, -ratio * s, t * t - 1)
    loss = -(weight * ll).sum() / n + l2 * (beta @ beta) / n
    grad = -np.concatenate([[d_mu.sum()], x.T @ d_mu, z.T @ d_log_sigma]) / n
    grad[1:k + 1] += 2 * l2 * beta / n
    return loss, grad


def weighted_fit(x, z, rise, l2, weight):
    """Returns parameters in shash.py's layout (skew 0, tail 1) so shash.cdf applies."""
    k, m = x.shape[1], z.shape[1]
    weight = np.asarray(weight, float) / np.mean(weight)
    start = np.zeros(k + 1 + m)
    start[0], start[k + 1] = rise.mean(), np.log(rise.std() + 1e-3)
    result = minimize(weighted_loss_and_grad, start, args=(x, z, rise, l2, weight), jac=True,
                      method="L-BFGS-B", options={"maxiter": 2000})
    return np.r_[result.x, np.zeros(m + 1)]


class Cohort:
    """Arrays for one cutoff: features are standardized per arm on that arm's training rows."""

    def __init__(self, rows):
        self.rows = rows
        self.z = np.column_stack([np.ones(len(rows)), rows.season_sin, rows.season_cos])
        self.rise = (rows.target_max - rows.own_max_so_far).to_numpy()
        self.edges = band_cdf_edges(rows.rename(columns={"own_max_so_far": "LGA_max_so_far"}))
        self.floor = rows.own_max_so_far.to_numpy()
        self.target = rows.target_category.to_numpy()
        self.lga = rows.airport.eq("LGA").to_numpy()
        self.role = rows.role.to_numpy()
        self.validation, self.test = self.role == "validation", self.role == "test"
        self.prior = month_prior(rows.loc[self.lga & (self.role == "train")], rows)
        self.dates = rows.index[self.test]

    def probabilities(self, theta, x, mask):
        cdf = shash.cdf(theta, x[mask], self.z[mask], self.edges[mask])
        return stabilize(to_bands(cdf), self.prior[mask], self.floor[mask])

    def validation_loss(self, theta, x):
        return daily_scores(self.probabilities(theta, x, self.validation), self.target[self.validation])["log_loss"].mean()

    def fit(self, train, columns, log, label, weights=(None,), **extra):
        """Choose L2 (and the LGA weight) on 2024 LGA validation; return the chosen fit."""
        x = design(self.rows, columns, train)
        best = None
        for weight in weights:
            for l2 in L2:
                if weight is None:
                    theta = shash.fit(x[train], self.z[train], self.rise[train], l2, shape=False)
                else:
                    w = np.where(self.lga[train], weight, 1.0)
                    theta = weighted_fit(x[train], self.z[train], self.rise[train], l2, w)
                loss = self.validation_loss(theta, x)
                log.append({"system": label, "l2": l2, "lga_weight": weight, "n_train": int(train.sum()),
                            "validation_log_loss": loss, **extra})
                if best is None or loss < best["validation_log_loss"]:
                    best = {"validation_log_loss": loss, "l2": l2, "lga_weight": weight, "theta": theta, "x": x,
                            "n_train": int(train.sum())}
        best["test"] = self.probabilities(best["theta"], best["x"], self.test)
        return best


def airport_shifts(cohort, columns, best):
    """Intercept shifts in degrees F of the latent rise, other features held equal (relative to LGA)."""
    train = cohort.role == "train"
    raw = cohort.rows[columns].astype(float)
    sd = raw[train].fillna(raw[train].median()).std().replace(0, 1)
    k = len(columns)
    beta = pd.Series(best["theta"][1:k + 1], index=columns)
    return {name.replace("is_", "shift_") + "_f": float(beta[name] / sd[name]) for name in INDICATORS}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    stations, labels, guidance_table = load_pooled()
    hrrr = pd.read_csv(DATA / "hrrr_features.csv.gz", parse_dates=["date"])
    saved = pd.read_csv(ROOT / "results" / "guidance" / "test_probabilities.csv.gz", parse_dates=["date"])
    saved_selection = pd.read_csv(ROOT / "results" / "guidance" / "selection.csv").set_index("hour")
    log, scores_rows, paired, selection, coverage, curve, shifts, checks, frames = [], [], [], [], [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            rows, cover = assemble(hour, stations, labels, guidance_table, pd.DatetimeIndex(hrrr.loc[hrrr.hour.eq(hour), "date"]))
            coverage.extend(cover)
            cohort = Cohort(rows)
            train = cohort.role == "train"
            hour_log = []
            fits = {
                "a": cohort.fit(train & cohort.lga, COLUMNS, hour_log, ARMS["a"]),
                "b": cohort.fit(train, COLUMNS, hour_log, ARMS["b"]),
                "c": cohort.fit(train, COLUMNS + INDICATORS, hour_log, ARMS["c"]),
                "d": cohort.fit(train, COLUMNS + INDICATORS, hour_log, ARMS["d"], weights=LGA_WEIGHTS),
            }
            shifts.append({"hour": hour, **airport_shifts(cohort, COLUMNS + INDICATORS, fits["c"]),
                           "l2": fits["c"]["l2"]})
            # Learning-curve points: pooled rows subsampled to the LGA count (and twice it), five draws each.
            n_lga = int((train & cohort.lga).sum())
            pool = np.flatnonzero(train)
            for key, columns in (("b", COLUMNS), ("c", COLUMNS + INDICATORS)):
                for size in (n_lga, 2 * n_lga):
                    for seed in SEEDS:
                        pick = np.zeros(len(rows), bool)
                        pick[np.random.default_rng(seed).choice(pool, size, replace=False)] = True
                        best = cohort.fit(pick, columns, hour_log, f"{ARMS[key]}, {size} rows", seed=seed)
                        test = daily_scores(best["test"], cohort.target[cohort.test]).mean()
                        curve.append({"hour": hour, "system": ARMS[key], "n_train": size, "seed": seed,
                                      "lga_rows": int((pick & cohort.lga).sum()), "l2": best["l2"],
                                      "validation_log_loss": best["validation_log_loss"], **test.add_prefix("test_")})
                full = fits[key]
                curve.append({"hour": hour, "system": ARMS[key], "n_train": full["n_train"], "seed": -1,
                              "lga_rows": n_lga, "l2": full["l2"], "validation_log_loss": full["validation_log_loss"],
                              **daily_scores(full["test"], cohort.target[cohort.test]).mean().add_prefix("test_")})
            curve.append({"hour": hour, "system": ARMS["a"], "n_train": n_lga, "seed": -1, "lga_rows": n_lga,
                          "l2": fits["a"]["l2"], "validation_log_loss": fits["a"]["validation_log_loss"],
                          **daily_scores(fits["a"]["test"], cohort.target[cohort.test]).mean().add_prefix("test_")})
            log.extend({"hour": hour, **entry} for entry in hour_log)

            # Saved systems from guidance_study.py on the same dates.
            dates, target = cohort.dates, cohort.target[cohort.test]
            reference = saved.loc[saved.hour.eq(hour)].set_index(["system", "date"])
            systems = {f"CURRENT ({CURRENT})": reference.loc[CURRENT].loc[dates, PROBS].to_numpy(),
                       f"Saved LGA only without HRRR ({SAVED_NO_HRRR})": reference.loc[SAVED_NO_HRRR].loc[dates, PROBS].to_numpy()}
            current_dates = pd.DatetimeIndex(reference.loc[CURRENT].index).sort_values()
            checks.append({"hour": hour, "test_dates": len(dates), "same_test_dates_as_current": current_dates.equals(dates),
                           "current_targets_match": bool((reference.loc[CURRENT].loc[dates, "target_category"].to_numpy() == target).all()),
                           "lga_train_dates": n_lga, "lga_train_dates_guidance_study": None,
                           "refit_vs_saved_no_hrrr_max_abs_prob": float(np.abs(fits["a"]["test"] - systems[
                               f"Saved LGA only without HRRR ({SAVED_NO_HRRR})"]).max()),
                           "weighted_fit_at_weight_1_vs_c_val_gap": float(min(
                               e["validation_log_loss"] for e in hour_log if e["system"] == ARMS["d"] and e["lga_weight"] == 1.0
                               and e["l2"] == fits["c"]["l2"]) - fits["c"]["validation_log_loss"])})
            validation = {f"CURRENT ({CURRENT})": saved_selection.loc[hour, f"val_{CURRENT}"],
                          f"Saved LGA only without HRRR ({SAVED_NO_HRRR})": saved_selection.loc[hour, f"val_{SAVED_NO_HRRR}"]}
            settings = {}
            for key, fit in fits.items():
                systems[ARMS[key]] = fit["test"]
                validation[ARMS[key]] = fit["validation_log_loss"]
                settings[ARMS[key]] = {"n_train": fit["n_train"], "l2": fit["l2"], "lga_weight": fit["lga_weight"]}
            no_hrrr = min(ARMS.values(), key=validation.get)
            overall = min([f"CURRENT ({CURRENT})", *ARMS.values()], key=validation.get)
            selection.append({"hour": hour, "selected_without_hrrr": no_hrrr, "selected_overall": overall,
                              **{f"val_{name}": validation[name] for name in [f"CURRENT ({CURRENT})", *ARMS.values()]}})
            scores = {name: daily_scores(p, target) for name, p in systems.items()}
            for name in systems:
                scores_rows.append({"hour": hour, "system": name, **settings.get(name, {}),
                                    "validation_log_loss": validation[name], "n_test": len(target),
                                    **scores[name].mean().rename({"correct": "accuracy"}).to_dict()})
                frames.append(pd.DataFrame(systems[name], columns=PROBS).assign(
                    system=name, hour=hour, date=dates, target_category=target))
            comparisons = [(f"CURRENT ({CURRENT})", ARMS[k]) for k in "abcd"]
            comparisons += [(ARMS["a"], ARMS[k]) for k in "bcd"]
            for baseline, candidate in comparisons:
                for metric in ("log_loss", "rps", "correct"):
                    diff = (scores[candidate][metric] - scores[baseline][metric]).to_numpy()
                    mean, low, high = block_bootstrap(dates, diff)
                    paired.append({"hour": hour, "baseline": baseline, "candidate": candidate,
                                   "metric": "accuracy" if metric == "correct" else metric, "delta": mean,
                                   "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
            print(f"{hour:02}:00 done; validation selects {no_hrrr} without HRRR, {overall} overall", flush=True)

    coverage = pd.DataFrame(coverage)
    guidance_coverage = pd.read_csv(ROOT / "results" / "guidance" / "coverage.csv").set_index("hour")
    checks = pd.DataFrame(checks)
    checks["lga_train_dates_guidance_study"] = checks.hour.map(guidance_coverage.train_dates)
    curve = pd.DataFrame(curve)
    tables = {"coverage": coverage, "validation_grid": pd.DataFrame(log), "test_scores": pd.DataFrame(scores_rows),
              "paired": pd.DataFrame(paired), "selection": pd.DataFrame(selection), "learning_curve": curve,
              "airport_shifts": pd.DataFrame(shifts), "checks": checks}
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "test_probabilities.csv.gz", index=False, compression=GZIP)
    write_readme(tables)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format, "display.max_rows", 200,
                           "display.max_colwidth", 60):
        print(coverage.to_string(index=False))
        print(tables["test_scores"].to_string(index=False))
        print(tables["paired"].loc[lambda t: t.metric.eq("log_loss")].to_string(index=False))
        print(curve.groupby(["hour", "system", "n_train"])[["validation_log_loss", "test_log_loss"]].agg(["mean", "std"]).to_string())
        print(tables["airport_shifts"].to_string(index=False))
        print(checks.to_string(index=False))


def write_readme(tables):
    """README_SECTION.md from the result tables, so a rerun after a data fix keeps every number current."""
    scores = tables["test_scores"].set_index(["hour", "system"])
    paired = tables["paired"].loc[lambda t: t.metric.eq("log_loss")].set_index(["hour", "baseline", "candidate"])
    coverage = tables["coverage"]
    selection = tables["selection"].set_index("hour")
    shifts = tables["airport_shifts"].set_index("hour")
    current, a, b, c, d = f"CURRENT ({CURRENT})", ARMS["a"], ARMS["b"], ARMS["c"], ARMS["d"]
    lga_rows = coverage.loc[coverage.airport.eq("LGA")].set_index("hour").train
    pooled_rows = coverage.groupby("hour").train.sum()
    curve = tables["learning_curve"]
    small = curve.loc[curve.seed.ge(0) & curve.system.eq(b)].groupby(["hour", "n_train"])[
        ["validation_log_loss", "test_log_loss"]].mean()

    def per_hour(values, digits=3, sign=False):
        return " / ".join(f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}" for v in values)

    def column(system, name="log_loss"):
        return per_hour(scores.loc[(h, system), name] for h in HOURS)

    def interval(baseline, candidate, h):
        row = paired.loc[(h, baseline, candidate)]
        return f"{row.delta:+.3f} [{row.ci95_low:+.3f}, {row.ci95_high:+.3f}]"

    def intervals(baseline, candidate):
        return "; ".join(interval(baseline, candidate, h) for h in HOURS)

    chosen = {h: selection.loc[h, "selected_without_hrrr"] for h in HOURS}
    chosen_vs_a = "; ".join(f"{h:02}:00 {chosen[h]} {interval(a, chosen[h], h)}" if chosen[h] != a
                            else f"{h:02}:00 LGA only itself" for h in HOURS)
    excluded = [f"{h:02}:00" for h in HOURS if chosen[h] != a and paired.loc[(h, a, chosen[h]), "excludes_zero"]]
    overall = [selection.loc[h, "selected_overall"] for h in HOURS]
    overall_text = ("CURRENT at every cutoff" if all(name == current for name in overall)
                    else ", ".join(f"{h:02}:00 {name.split(' (')[0]}" for h, name in zip(HOURS, overall)))
    same_count_val = per_hour(small.loc[(h, lga_rows[h]), "validation_log_loss"] for h in HOURS)
    same_count_test = per_hour(small.loc[(h, lga_rows[h]), "test_log_loss"] for h in HOURS)
    more_val = per_hour((scores.loc[(h, b), "validation_log_loss"] - small.loc[(h, lga_rows[h]), "validation_log_loss"]
                         for h in HOURS), sign=True)
    more_test = per_hour((scores.loc[(h, b), "log_loss"] - small.loc[(h, lga_rows[h]), "test_log_loss"]
                          for h in HOURS), sign=True)
    pairs = [distance_km(x, y) / 1.609344 for i, x in enumerate(AIRPORTS) for y in AIRPORTS[i + 1:]]
    miles = (min(pairs), max(pairs))
    shift_columns = [f"shift_{airport}_f" for airport in AIRPORTS[1:]]
    largest = float(shifts[shift_columns].abs().to_numpy().max())
    overlap = pd.read_csv(POOLED / "overlap_check_lga.csv")
    rows_same = bool(overlap.same_dates.all() and overlap.same_columns.all() and (overlap.cells_equal == 1).all())
    checks = ("rebuilt LGA labels and features equal the bundled samples in every cell" if rows_same else
              f"rebuilt LGA rows do NOT match the bundled samples (lowest share of equal cells {overlap.cells_equal.min():.4f})")
    checks += ("; rebuilt KLGA guidance equals the bundled table" if overlap.klga_text_identical.all() else
               "; rebuilt KLGA guidance does NOT match the bundled table")
    lines = [
        "## Pooling nearby airports as training targets",
        "",
        f"`python final_project/pooled_airports.py all` (build about 1 minute from cached raw files, run under 20 seconds) adds JFK, EWR and TEB as training targets for the censored Gaussian without HRRR: each airport's own label (prepare.py's rule), its own observations and LAMP/NBM guidance as the \"own\" inputs, the other three airports by distance as neighbour inputs. Training uses the same 2021-2023 dates as CURRENT ({lga_rows.iloc[0]:,} LGA rows, {pooled_rows.iloc[0]:,} pooled at 09:00); L2 and an LGA weight are chosen on 2024 LGA validation; test is the same 359 LGA days.",
        "",
        f"- Test log loss (09:00 / 12:00 / 15:00): LGA only {column(a)}, pooled {column(b)}, pooled with airport intercept shifts {column(c)}, the same with an LGA weight {column(d)}; CURRENT with HRRR {column(current)}. Validation: LGA only {column(a, 'validation_log_loss')}, pooled {column(b, 'validation_log_loss')}, CURRENT {column(current, 'validation_log_loss')}.",
        f"- Pooled minus LGA only: {intervals(a, b)} (95% 7-day block intervals). Pooled minus CURRENT: {intervals(current, b)}.",
        f"- Validation picks, without HRRR: {chosen_vs_a} against LGA only; the interval excludes zero at {', '.join(excluded) if excluded else 'no cutoff'}. With CURRENT among the candidates, validation selects {overall_text}.",
        f"- Same number of rows, different rows: the pool subsampled to the LGA count (five draws, about a quarter LGA) scores validation {same_count_val} and test {same_count_test}. Using all pooled rows instead changes validation by {more_val} and test by {more_test}.",
        f"- Airport intercept shifts are small (largest {largest:.1f} F of rise at equal inputs), so each airport's own observations and guidance already carry most of the difference between airports.",
        "",
        f"The extra rows are the same days at airports {miles[0]:.0f} to {miles[1]:.0f} miles apart, so they repeat LGA's weather and are far from independent samples. Checks (`test_pooled_airports.py`): {checks}; no LAMP or NBM run is used before its assumed release at any airport; the LGA-only refit matches guidance_study.py's saved no-HRRR model to {tables['checks'].refit_vs_saved_no_hrrr_max_abs_prob.max():.3f} in any band probability.",
    ]
    (OUT / "README_SECTION.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", default="run", choices=["fetch", "build", "run", "all"])
    args = parser.parse_args()
    if args.step == "fetch":
        fetch()
    if args.step in ("build", "all"):
        build()
    if args.step in ("run", "all"):
        run()


if __name__ == "__main__":
    main()
