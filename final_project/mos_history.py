"""Does more training history help once the guidance exists over the whole window? Run after mos.py build.

history_study.py found that 2010-2023 training hurt because LAMP, NBM and HRRR had to be imputed before
2018-2021. GFS and NAM MOS (data/mos_features.csv.gz) exist for every year, so here nothing is imputed in
training: a row enters a window only if the MOS inputs it uses are present.

(a) Stations + trends + MOS, censored Gaussian, trained 2010-2023 vs 2018-2023 vs 2021-2023, for two MOS
    sets that exist from 2008 (GFS alone; GFS + NAM). Context: stations + trends alone on 2010-2023.
    MEX starts only in July 2020 in the archive, so it is used in (b) and nowhere in the long windows.
(b) CURRENT's inputs (four stations + trends + HRRR + LAMP/NBM) plus MOS (GFS; GFS + NAM + MEX), 2021-2023
    complete rows.
(c) Two ways to use the long record with the short-window model:
    Two-stage: the 2010-2023 MOS model's predicted mean and log sigma of the remaining warming become two
      more inputs to CURRENT's model. Stage-2 training rows (2021-2023) get out-of-fold stage-1 values from a
      fit that leaves their own calendar year out; 2024 and the test year get the fit on all of 2010-2023.
    Weighted average: w * long MOS model + (1 - w) * CURRENT, w on a 0.1 grid chosen on 2024.
Every model is the censored Gaussian of shash.py (shape=False) with L2 chosen on 2024 validation. CURRENT is
refit here for validation (it reproduces the saved test probabilities) and the saved CURRENT probabilities are
the paired baseline. Validation and test are the `complete` rows of history_study.load: 359 test days per
cutoff, the same days as every other result. Writes results/mos_history/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import shash
from distribution import band_cdf_edges, design, relative_columns, to_bands
from experiment import HOURS, PROBS, month_prior, split_dates, stabilize
from history_study import GROUPS, SCOPE, load
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
OBS = "Four stations + trends"
L2 = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)
WINDOWS = {"2010-2023": 2010, "2018-2023": 2018, "2021-2023": 2021}
CURRENT = "Censored Gaussian, HRRR + LAMP/NBM"
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
PRODUCT = {
    "gfs": ["gfs_max", "rel_gfs", "gfs_remaining_max", "rel_gfs_rest", "gfs_now_error", "gfs_cloud", "gfs_u", "gfs_v"],
    "nam": ["nam_max", "rel_nam", "nam_remaining_max", "rel_nam_rest", "nam_now_error", "nam_cloud", "nam_u", "nam_v"],
    "mex": ["mex_max", "rel_mex"],
}
RAW = {"gfs": ["gfs_max", "gfs_remaining_max", "gfs_now"], "nam": ["nam_max", "nam_remaining_max", "nam_now"],
       "mex": ["mex_max"]}
LONG_SETS = {"GFS MOS": ("gfs",), "GFS + NAM MOS": ("gfs", "nam")}  # archive covers 2008-2026
SHORT_SETS = {"GFS MOS": ("gfs",), "GFS + NAM + MEX MOS": ("gfs", "nam", "mex")}  # MEX from 2020-07


def add_mos(samples, mos, hour):
    """Join one cutoff's MOS features and the floor-relative versions used by the models."""
    columns = [c for c in mos if c.split("_")[0] in PRODUCT and not c.endswith("_run")]
    out = samples.join(mos.loc[mos.hour == hour].set_index("date")[columns], how="left")
    for p in ("gfs", "nam"):
        out[f"rel_{p}"] = out[f"{p}_max"] - out.LGA_max_so_far
        out[f"rel_{p}_rest"] = out[f"{p}_remaining_max"] - out.LGA_max_so_far
        out[f"{p}_now_error"] = out.LGA_temp - out[f"{p}_now"]
    out["rel_mex"] = out.mex_max - out.LGA_max_so_far
    return out


def mos_columns(products):
    return [c for p in products for c in PRODUCT[p]]


def mos_present(samples, products):
    return samples[[c for p in products for c in RAW[p]]].notna().all(axis=1).to_numpy()


class Problem:
    """Arrays shared by every fit at one cutoff."""

    def __init__(self, samples, evaluate):
        self.samples = samples
        self.z = np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos])
        self.floor = samples.LGA_max_so_far.to_numpy()
        self.rise = (samples.target_max - samples.LGA_max_so_far).to_numpy()
        self.edges = band_cdf_edges(samples)
        self.target = samples.target_category.to_numpy()
        self.evaluate = evaluate

    def loss(self, p, split="validation"):
        mask = self.evaluate[split]
        return float(daily_scores(p[mask], self.target[mask])["log_loss"].mean())

    def fit(self, columns, train, grid=L2, log=None, tag=None):
        """Censored Gaussian over the L2 grid; the setting with the lowest 2024 log loss is kept."""
        x = design(self.samples, columns, train)
        prior = month_prior(self.samples.loc[train], self.samples)
        best = None
        for l2 in grid:
            theta = shash.fit(x[train], self.z[train], self.rise[train], l2, shape=False)
            p = stabilize(to_bands(shash.cdf(theta, x, self.z, self.edges)), prior, self.floor)
            loss = self.loss(p)
            if log is not None:
                log.append({**tag, "l2": l2, "validation_log_loss": loss})
            if best is None or loss < best["validation_log_loss"]:
                mu, log_sigma, _, _ = shash.parameters(theta, x, self.z)
                best = {"p": p, "l2": l2, "validation_log_loss": loss, "mu": mu, "log_sigma": log_sigma}
        return best


def out_of_fold(problem, columns, window_train, stage2_train, l2):
    """Stage-1 mean and log sigma for every row without leakage into stage-2 training rows.

    Rows of a stage-2 training year get a stage-1 fit that leaves that whole calendar year out; all other rows
    (2024 validation, the test year) get the fit on the full window. Returns (mu, log_sigma, held-out years).
    """
    full = problem.fit(columns, window_train, grid=(l2,))
    mu, log_sigma = full["mu"].copy(), full["log_sigma"].copy()
    years = problem.samples.index.year.to_numpy()
    held = sorted(int(y) for y in set(years[stage2_train]))
    for year in held:
        fold = problem.fit(columns, window_train & (years != year), grid=(l2,))
        rows = years == year
        mu[rows], log_sigma[rows] = fold["mu"][rows], fold["log_sigma"][rows]
    return mu, log_sigma, held


def run_hour(hour, samples, complete, saved, log):
    splits = split_dates(samples.index)
    evaluate = {"validation": splits["validation"] & complete, "test": splits["test"] & complete}
    problem = Problem(samples, evaluate)
    test = evaluate["test"]
    dates = samples.index[test]
    years = samples.index.year.to_numpy()
    obs = relative_columns(samples, OBS)
    current_columns = relative_columns(samples, SCOPE) + list(GROUPS)
    short = splits["train"] & complete & (years >= 2021)
    systems, coverage = {}, []

    def record(name, fit, part, train, note=""):
        systems[name] = {"p": fit["p"], "part": part, "l2": fit.get("l2"), "train_rows": int(train.sum()),
                         "validation_log_loss": fit["validation_log_loss"], "note": note}

    current = problem.fit(current_columns, short, log=log, tag={"hour": hour, "system": "CURRENT refit"})
    record("CURRENT refit", current, "baseline", short)
    long_fits = {}
    for set_name, products in LONG_SETS.items():
        present = mos_present(samples, products)
        for window, start in WINDOWS.items():
            train = splits["train"] & (years >= start) & present
            name = f"Stations + trends + {set_name}, {window}"
            fit = problem.fit(obs + mos_columns(products), train, log=log, tag={"hour": hour, "system": name})
            record(name, fit, "a", train)
            long_fits[(set_name, window)] = (fit, train)
            coverage.append({"hour": hour, "mos_set": set_name, "window": window, "train_rows": int(train.sum()),
                             "window_rows": int((splits["train"] & (years >= start)).sum()),
                             "validation_rows_missing_mos": int((evaluate["validation"] & ~present).sum()),
                             "test_rows_missing_mos": int((test & ~present).sum())})
    train = splits["train"] & (years >= 2010)
    record("Stations + trends only, 2010-2023", problem.fit(obs, train, log=log, tag={"hour": hour,
           "system": "Stations + trends only, 2010-2023"}), "a", train)
    for set_name, products in SHORT_SETS.items():
        name = f"CURRENT + {set_name}, 2021-2023"
        record(name, problem.fit(current_columns + mos_columns(products), short, log=log,
                                 tag={"hour": hour, "system": name}), "b", short)
        present = mos_present(samples, products)  # the few gaps are filled with the training median
        coverage.append({"hour": hour, "mos_set": f"CURRENT + {set_name}", "window": "2021-2023 complete",
                         "train_rows": int(short.sum()), "train_rows_missing_mos": int((short & ~present).sum()),
                         "validation_rows_missing_mos": int((evaluate["validation"] & ~present).sum()),
                         "test_rows_missing_mos": int((test & ~present).sum())})

    # (c) The long-history model is the validation-better MOS set at 2010-2023.
    set_name = min(LONG_SETS, key=lambda s: long_fits[(s, "2010-2023")][0]["validation_log_loss"])
    long_fit, long_train = long_fits[(set_name, "2010-2023")]
    columns = obs + mos_columns(LONG_SETS[set_name])
    mu, log_sigma, held = out_of_fold(problem, columns, long_train, short, long_fit["l2"])
    staged = samples.assign(stage1_mu=mu, stage1_log_sigma=log_sigma)
    stage2 = Problem(staged, evaluate)
    name = "Two-stage: long MOS model feeds CURRENT"
    record(name, stage2.fit(current_columns + ["stage1_mu", "stage1_log_sigma"], short, log=log,
                            tag={"hour": hour, "system": name}), "c", short,
           f"stage 1 = {set_name} 2010-2023, l2={long_fit['l2']}, out-of-fold years {held}")
    blends = {w: w * long_fit["p"] + (1 - w) * current["p"] for w in WEIGHTS}
    for w, p in blends.items():
        log.append({"hour": hour, "system": "Weighted average", "weight_long": w,
                    "validation_log_loss": problem.loss(p)})
    w = min(WEIGHTS, key=lambda w: problem.loss(blends[w]))
    record("Weighted average: long MOS model and CURRENT", {"p": blends[w], "validation_log_loss": problem.loss(blends[w])},
           "c", short, f"weight on {set_name} 2010-2023 = {w}")

    target = problem.target[test]
    base = saved.loc[saved.hour.eq(hour)].set_index("date").loc[dates, PROBS].to_numpy()
    reproduction = float(np.abs(systems["CURRENT refit"]["p"][test] - base).max())
    scores = {"CURRENT (saved)": daily_scores(base, target)}
    rows = [{"hour": hour, "system": "CURRENT (saved)", "part": "baseline", "n": len(target),
             "validation_log_loss": current["validation_log_loss"], **scores["CURRENT (saved)"].mean().to_dict()}]
    paired, frames = [], []
    for name, s in systems.items():
        scores[name] = daily_scores(s["p"][test], target)
        rows.append({"hour": hour, "system": name, "part": s["part"], "n": len(target),
                     "validation_log_loss": s["validation_log_loss"], **scores[name].mean().to_dict(),
                     "l2": s["l2"], "train_rows": s["train_rows"], "note": s["note"]})
        frames.append(pd.DataFrame(s["p"][test], columns=PROBS).assign(system=name, hour=hour, date=dates,
                                                                       target_category=target))
    chosen = min(systems, key=lambda name: systems[name]["validation_log_loss"])
    comparisons = [(name, "CURRENT (saved)") for name in systems if name != "CURRENT refit"]
    comparisons += [(f"Stations + trends + {s}, 2010-2023", f"Stations + trends + {s}, {w}")
                    for s in LONG_SETS for w in ("2018-2023", "2021-2023")]
    for candidate, baseline in comparisons:
        for metric in ("log_loss", "rps"):
            mean, low, high = block_bootstrap(dates, (scores[candidate][metric] - scores[baseline][metric]).to_numpy())
            paired.append({"hour": hour, "candidate": candidate, "baseline": baseline, "metric": metric,
                           "delta": mean, "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
    selection = {"hour": hour, "selected": chosen, "validation_log_loss": systems[chosen]["validation_log_loss"],
                 "test_log_loss": float(scores[chosen]["log_loss"].mean()),
                 "current_validation_log_loss": current["validation_log_loss"],
                 "current_refit_max_abs_prob_diff_vs_saved": reproduction, "long_model_mos_set": set_name}
    return rows, paired, coverage, selection, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "mos_history")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    mos = pd.read_csv(args.data / "mos_features.csv.gz", parse_dates=["date"])
    saved = pd.read_csv(ROOT / "results" / "guidance" / "test_probabilities.csv.gz", parse_dates=["date"])
    saved = saved.loc[saved.system.eq(CURRENT)]
    log, rows, paired, coverage, selections, frames = [], [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples, complete = load(args.data, hour, hrrr, guidance)
            out = run_hour(hour, add_mos(samples, mos, hour), complete, saved, log)
            for table, part in zip((rows, paired, coverage, frames), (out[0], out[1], out[2], out[4])):
                table.extend(part)
            selections.append(out[3])
            print(f"{hour:02}:00 done, validation selects {out[3]['selected']} ({time.time() - started:.0f} s)",
                  flush=True)
    for name, table in (("validation_grid", log), ("test_scores", rows), ("paired_vs_current", paired),
                        ("train_coverage", coverage), ("selection", selections)):
        pd.DataFrame(table).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(args.out / "test_probabilities.csv.gz", index=False,
                                                compression={"method": "gzip", "mtime": 0})
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 300,
                           "display.max_colwidth", 60):
        print(pd.DataFrame(rows).drop(columns=["note"]).to_string(index=False))
        print(pd.DataFrame(paired).to_string(index=False))
        print(pd.DataFrame(selections).to_string(index=False))
    print(f"total {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
