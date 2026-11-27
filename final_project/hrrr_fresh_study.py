"""Does the newest HRRR run released by each cutoff beat the fixed 10 UTC run? Run after hrrr_fresh.py build.

CURRENT is the selected system: censored Gaussian on four stations + trends + HRRR + LAMP/NBM, trained on
2021-2023 complete rows (history_study.load), L2 chosen on 2024 validation, scored on the 359 test days.
Candidates, fit the same way on the same inputs:
  (a) replace: the fresh run's remaining max and warming take the place of the 10 UTC ones
  (b) add: the fresh remaining max, warming and floor-relative max are added next to the 10 UTC ones
  control: CURRENT's inputs refit on the training rows that have a fresh run (isolates the row change)
(a), (b) and the control train only on rows with a fresh run and choose L2 on 2024 rows with a fresh run.
One system per cutoff is selected among CURRENT, (a) and (b) on the 2024 rows with a fresh run; the test
set chooses nothing. Test days without a fresh run are scored two ways: on the intersection only, and on
all 359 days with the 10 UTC features standing in for the missing fresh ones. The raw baselines take
max(high so far, HRRR remaining max) as the forecast of the final high.

Samples are read at run time through history_study.load, so the script reruns end to end from the bundled
tables. Writes results/hrrr_fresh/, about 1 minute with 2 threads.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from distribution import add_relative, band_cdf_edges, design, relative_columns, tobit_bands, tobit_fit
from experiment import HOURS, PROBS, month_prior, split_dates, stabilize
from history_study import SCOPE, load
from prepare import local_cutoff
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
L2 = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)  # the grid that selected CURRENT in distribution.run_cohort
FRESH = ["hrrr_fresh_remaining_max", "hrrr_fresh_warming"]
ADDED = FRESH + ["rel_hrrr_fresh"]
CURRENT = "CURRENT (10 UTC HRRR)"
REPLACE = "(a) fresh HRRR replaces 10 UTC"
ADD = "(b) fresh HRRR added"
CONTROL = "Control: CURRENT inputs, fresh-run rows"
SAVED = "Censored Gaussian, HRRR + LAMP/NBM"


def variants(samples, fresh):
    """Input tables for each system; a missing fresh value falls back to the 10 UTC value."""
    replaced = samples.copy()
    replaced["hrrr_remaining_max"] = fresh.hrrr_fresh_remaining_max.fillna(samples.hrrr_remaining_max)
    replaced["hrrr_warming"] = fresh.hrrr_fresh_warming.fillna(samples.hrrr_warming)
    added = samples.assign(hrrr_fresh_remaining_max=replaced.hrrr_remaining_max,
                           hrrr_fresh_warming=replaced.hrrr_warming)
    added["rel_hrrr_fresh"] = added.hrrr_fresh_remaining_max - added.LGA_max_so_far
    return {CURRENT: (samples, []), REPLACE: (add_relative(replaced), []), ADD: (added, ADDED),
            CONTROL: (samples, [])}


def fit(samples, extra, train, validation, log, hour, name):
    """Censored Gaussian of distribution.run_cohort; returns band probabilities for every row."""
    columns = relative_columns(samples, SCOPE) + extra
    x = design(samples, columns, train)
    z = np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos])
    prior = month_prior(samples.loc[train], samples)
    floor = samples.LGA_max_so_far.to_numpy()
    rise = (samples.target_max - samples.LGA_max_so_far).to_numpy()
    edges = band_cdf_edges(samples)
    target = samples.target_category.to_numpy()
    best = None
    for l2 in L2:
        theta, k = tobit_fit(x[train], z[train], rise[train], l2)
        p = stabilize(tobit_bands(theta, k, x, z, edges)[0], prior, floor)
        loss = daily_scores(p[validation], target[validation])["log_loss"].mean()
        log.append({"hour": hour, "system": name, "l2": l2, "validation_log_loss": loss,
                    "train_rows": int(train.sum()), "validation_rows": int(validation.sum())})
        if best is None or loss < best[0]:
            best = (loss, p, l2)
    return best[1], best[2]


def raw_baseline(samples, fresh, splits, has_fresh, hour):
    """MAE of max(high so far, HRRR remaining max) for the final high, raw and with a 2021-2023 seasonal bias."""
    floor = np.nan_to_num(samples.LGA_max_so_far.to_numpy(), nan=-np.inf)
    y = samples.target_max.to_numpy()
    season = (samples.index.month % 12) // 3
    rows = []
    for source, maximum, now in (("10 UTC", samples.hrrr_remaining_max, samples.hrrr_remaining_max - samples.hrrr_warming),
                                 ("fresh", fresh.hrrr_fresh_remaining_max,
                                  fresh.hrrr_fresh_remaining_max - fresh.hrrr_fresh_warming)):
        guess = np.fmax(maximum.to_numpy(), floor)
        corrected = guess.copy()
        for s in range(4):
            train = splits["train"] & has_fresh & (season == s)
            corrected[season == s] += (y - guess)[train].mean()
        for split in ("validation", "test"):
            mask = splits[split] & has_fresh
            rows.append({"hour": hour, "source": source, "split": split, "n": int(mask.sum()),
                         "raw_mae_f": float(np.abs(guess - y)[mask].mean()),
                         "bias_corrected_mae_f": float(np.abs(corrected - y)[mask].mean()),
                         "now_mae_f": float(np.nanmean(np.abs(now.to_numpy() - samples.LGA_temp.to_numpy())[mask]))})
    return rows


def paired(dates, scores, systems, base, cohort, hour, mask):
    out = []
    for name in systems:
        if name == base:
            continue
        for metric in ("log_loss", "rps", "correct"):
            diff = (scores[name][metric] - scores[base][metric]).to_numpy()[mask]
            mean, low, high = block_bootstrap(dates[mask], diff)
            out.append({"hour": hour, "cohort": cohort, "baseline": base, "candidate": name, "metric": metric,
                        "n": int(mask.sum()), "delta": mean, "ci95_low": low, "ci95_high": high,
                        "excludes_zero": low > 0 or high < 0})
    return out


def signed(value):
    return f"{round(value, 3) + 0.0:+.3f}"


def readme(coverage, scores, selection, pairs, raw, reproduction, concentration, cycles):
    """README section generated from the result tables, so a rerun after a data fix stays consistent."""
    cohort = "all 359, 10 UTC fallback"
    get = lambda h, name: scores.loc[(scores.hour == h) & (scores.system == name) & (scores.cohort == cohort)].iloc[0]
    by_hour = lambda f, spec="{:.3f}": " / ".join(spec.format(f(h)) for h in HOURS)
    val = lambda name: by_hour(lambda h: selection.loc[selection.hour == h, f"val_{name}"].item())
    test = lambda name, metric="log_loss", spec="{:.3f}": by_hour(lambda h: get(h, name)[metric], spec)
    cov = lambda column, spec="{:.1f}": by_hour(lambda h: coverage.loc[coverage.hour == h, column].item(), spec)
    rawv = lambda source, split, column: by_hour(lambda h: raw.loc[(raw.hour == h) & (raw.source == source)
                                                                   & (raw.split == split), column].item(), "{:.2f}")
    conc = lambda name, column: by_hour(lambda h: signed(concentration.loc[(concentration.hour == h) & (concentration.candidate == name),
                                                                           column].item()), "{}")

    def delta(name, which=cohort):
        rows = pairs.loc[(pairs.candidate == name) & (pairs.baseline == CURRENT) & (pairs.metric == "log_loss")
                         & (pairs.cohort == which)].set_index("hour")
        text = ", ".join(f"{signed(rows.delta[h])} [{signed(rows.ci95_low[h])}, {signed(rows.ci95_high[h])}]" for h in HOURS)
        return text, bool(rows.excludes_zero.any())

    lost = cycles.loc[cycles.status.ne("ok")]
    share = lost.date.dt.year.eq(2024).mean() if len(lost) else 0.0
    picks = [selection.loc[selection.hour == h, "selected"].item().split(" ")[0] for h in HOURS]
    chosen = ("CURRENT at all three cutoffs" if all(x == "CURRENT" for x in picks)
              else ", ".join(f"{x} at {h:02}:00" for x, h in zip(picks, HOURS)))
    full = bool((coverage.test_with_fresh == coverage.test_rows).all())
    (a_text, a_sig), (b_text, b_sig) = delta(REPLACE), delta(ADD)
    verdict = ("every interval includes zero" if not (a_sig or b_sig)
               else "intervals marked in paired_vs_current.csv exclude zero")
    lines = [
        "## Fresh HRRR: the newest run released by each cutoff",
        "",
        "`python final_project/hrrr_fresh.py fetch` and `build` replace the fixed 10 UTC HRRR run with the newest hourly run whose 2 m temperature tile (same Utah hrrrzarr archive, grid point and S3 Last-Modified rule as `hrrr.py`) was archived by the cutoff and whose leads reach local midnight; point series are cached in the gitignored `data/raw/hrrr_fresh/`, and the bundled tables are `data/hrrr_fresh_cycles.csv.gz` and `data/hrrr_fresh_features.csv.gz`.",
        f"The archive writes each run about 2 hours after initialization, so on test days the fresh run is on average {cov('mean_fresh_age_h')} h old at 09:00 / 12:00 / 15:00, against {cov('mean_10utc_age_h')} h for the 10 UTC run.",
        f"A fresh run exists on {cov('test_with_fresh', '{:.0f}')} of the 359 test days" + (", so the 10 UTC fallback is never needed on test" if full else "; on the other days the 10 UTC features stand in") + f", and on {cov('validation_with_fresh', '{:.0f}')} of the {coverage.validation_rows.iloc[0]} validation days.",
        f"Freshness helps the raw forecast: HRRR's error at the cutoff hour against the latest LGA reading falls from {rawv('10 UTC', 'validation', 'now_mae_f')} F to {rawv('fresh', 'validation', 'now_mae_f')} F on 2024 days, and the test MAE of max(high so far, HRRR remaining max) for the final high falls from {rawv('10 UTC', 'test', 'raw_mae_f')} F to {rawv('fresh', 'test', 'raw_mae_f')} F.",
        "`python final_project/hrrr_fresh_study.py` (offline, under a minute) refits CURRENT, the censored Gaussian on stations + trends + HRRR + LAMP/NBM, with (a) the fresh run replacing the 10 UTC features and (b) the fresh run added beside them, trained on 2021-2023 complete rows and tuned on 2024.",
        f"Validation log loss: CURRENT {val(CURRENT)}, (a) {val(REPLACE)}, (b) {val(ADD)}, so validation selects {chosen}.",
        f"The gap sits in a few confident misses: the mean validation difference against CURRENT is {conc(REPLACE, 'mean_delta_vs_current')} for (a) and {conc(ADD, 'mean_delta_vs_current')} for (b), but {conc(REPLACE, 'mean_delta_without_worst_3_days')} and {conc(ADD, 'mean_delta_without_worst_3_days')} without each one's three worst days (`validation_concentration.csv`); trimming flatters any system, so this describes the gap and does not override the selection.",
        f"On the same 359 test days, log loss is CURRENT {test(CURRENT)}, (a) {test(REPLACE)}, (b) {test(ADD)}; RPS {test(CURRENT, 'rps', '{:.4f}')}, {test(REPLACE, 'rps', '{:.4f}')}, {test(ADD, 'rps', '{:.4f}')}; accuracy {test(CURRENT, 'correct', '{:.1%}')}, {test(REPLACE, 'correct', '{:.1%}')}, {test(ADD, 'correct', '{:.1%}')}.",
        f"Paired log loss differences against CURRENT with 95% 7-day block intervals: (a) {a_text}; (b) {b_text}; {verdict}, so the selected system stays as validation chose.",
    ]
    if not full:
        lines.append(f"On the intersection of days with a fresh run: (a) {delta(REPLACE, 'fresh-run test days')[0]}; (b) {delta(ADD, 'fresh-run test days')[0]}.")
    if len(reproduction):
        lines.append(f"The refit CURRENT matches the saved guidance-study test probabilities to {reproduction.max_abs_probability_difference.max():.1e}.")
    lines.append(f"`test_hrrr_fresh.py` checks that no bundled run was archived after its cutoff and no newer qualifying run was skipped, the daylight-saving cutoffs and the lead window. Last-Modified is when Utah's archive wrote the tile, a conservative stand-in for public release, and dates rewritten later in the archive have no qualifying run ({len(lost)} of {len(cycles)} date-cutoffs from 2021 to August 2026, {share:.0%} of them in 2024).")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "hrrr_fresh")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    fresh_all = pd.read_csv(args.data / "hrrr_fresh_features.csv.gz", parse_dates=["date"])
    saved = ROOT / "results" / "guidance" / "test_probabilities.csv.gz"
    saved = pd.read_csv(saved, parse_dates=["date"]) if saved.exists() else None
    log, rows, pairs, coverage, selections, raw, reproduction, frames = [], [], [], [], [], [], [], []
    concentration = []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples, complete = load(args.data, hour, hrrr, guidance)
            samples = samples.loc[complete]
            fresh = fresh_all.loc[fresh_all.hour == hour].set_index("date").reindex(samples.index)
            has_fresh = fresh.hrrr_fresh_remaining_max.notna().to_numpy()
            splits = split_dates(samples.index)
            test = splits["test"]
            dates = samples.index[test]
            target = samples.target_category.to_numpy()
            fresh_val = splits["validation"] & has_fresh
            coverage.append({"hour": hour, "train_rows": int(splits["train"].sum()),
                             "train_with_fresh": int((splits["train"] & has_fresh).sum()),
                             "validation_rows": int(splits["validation"].sum()), "validation_with_fresh": int(fresh_val.sum()),
                             "test_rows": int(test.sum()), "test_with_fresh": int((test & has_fresh).sum()),
                             "mean_fresh_age_h": float(fresh.hrrr_fresh_age_h[test & has_fresh].mean()),
                             "mean_10utc_age_h": float(np.mean([(local_cutoff(d, hour).tz_convert("UTC") - d.tz_localize("UTC"))
                                                               / pd.Timedelta(hours=1) - 10 for d in dates])),
                             **{f"test_age_{int(a)}h": int(n) for a, n in
                                fresh.hrrr_fresh_age_h[test & has_fresh].value_counts().sort_index().items()}})
            systems, val_loss, chosen_l2 = {}, {}, {}
            for name, (table, extra) in variants(samples, fresh).items():
                train = splits["train"] if name == CURRENT else splits["train"] & has_fresh
                validation = splits["validation"] if name == CURRENT else fresh_val
                p, l2 = fit(table, extra, train, validation, log, hour, name)
                systems[name], chosen_l2[name] = p, l2
                # Selection compares every system on the same 2024 rows (those with a fresh run).
                val_loss[name] = daily_scores(p[fresh_val], target[fresh_val])["log_loss"].mean()
            if saved is not None:
                ref = saved.loc[saved.system.eq(SAVED) & saved.hour.eq(hour)].set_index("date").reindex(dates)
                reproduction.append({"hour": hour, "max_abs_probability_difference":
                                     float(np.abs(ref[PROBS].to_numpy() - systems[CURRENT][test]).max()),
                                     "saved_targets_match": bool((ref.target_category.to_numpy() == target[test]).all())})
            candidates = (CURRENT, REPLACE, ADD)
            selected = min(candidates, key=val_loss.get)
            every = splits["validation"]
            base_val = daily_scores(systems[CURRENT][fresh_val], target[fresh_val])["log_loss"].to_numpy()
            for name in (REPLACE, ADD):
                diff = pd.Series(daily_scores(systems[name][fresh_val], target[fresh_val])["log_loss"].to_numpy() - base_val,
                                 index=samples.index[fresh_val]).sort_values()
                concentration.append({"hour": hour, "candidate": name, "validation_days": len(diff),
                                      "mean_delta_vs_current": diff.mean(), "mean_delta_without_worst_3_days": diff.iloc[:-3].mean(),
                                      "worst_3_days": ";".join(f"{d:%Y-%m-%d}:{v:+.2f}" for d, v in diff.iloc[-3:].items())})
            selections.append({"hour": hour, "selected": selected, **{f"val_{k}": val_loss[k] for k in val_loss},
                               "val_CURRENT_all_2024_rows": daily_scores(systems[CURRENT][every], target[every])["log_loss"].mean(),
                               **{f"l2_{k}": chosen_l2[k] for k in chosen_l2}})
            systems["Validation-selected"] = systems[selected]
            scores = {name: daily_scores(p[test], target[test]) for name, p in systems.items()}
            fresh_test = has_fresh[test]
            for cohort, mask in (("all 359, 10 UTC fallback", np.ones(len(dates), bool)), ("fresh-run test days", fresh_test)):
                for name in systems:
                    rows.append({"hour": hour, "cohort": cohort, "system": name, "n": int(mask.sum()),
                                 "validation_log_loss": val_loss.get(name, val_loss[selected]),
                                 **scores[name][mask].mean().to_dict()})
                pairs += paired(dates, scores, systems, CURRENT, cohort, hour, mask)
                pairs += paired(dates, scores, {REPLACE: 0, ADD: 0}, CONTROL, cohort, hour, mask)
            raw += raw_baseline(samples, fresh, splits, has_fresh, hour)
            frames.append(pd.concat([pd.DataFrame(p[test], columns=PROBS).assign(system=name)
                                     for name, p in systems.items()])
                          .assign(hour=hour, date=np.tile(dates, len(systems)), target_category=np.tile(target[test], len(systems)),
                                  has_fresh=np.tile(fresh_test, len(systems))))
            print(f"{hour:02}:00 complete, validation selects {selected}", flush=True)
    tables = {"coverage": pd.DataFrame(coverage), "validation_grid": pd.DataFrame(log), "test_scores": pd.DataFrame(rows),
              "paired_vs_current": pd.DataFrame(pairs), "selection": pd.DataFrame(selections),
              "raw_baseline": pd.DataFrame(raw), "reproduction": pd.DataFrame(reproduction),
              "validation_concentration": pd.DataFrame(concentration)}
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(args.out / "test_probabilities.csv.gz", index=False,
                                                compression={"method": "gzip", "mtime": 0})
    (args.out / "README_SECTION.md").write_text(readme(tables["coverage"], tables["test_scores"], tables["selection"],
                                                       tables["paired_vs_current"], tables["raw_baseline"],
                                                       tables["reproduction"], tables["validation_concentration"],
                                                       pd.read_csv(args.data / "hrrr_fresh_cycles.csv.gz", parse_dates=["date"])))
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format, "display.max_rows", 200):
        for name in ("coverage", "selection", "test_scores", "raw_baseline", "reproduction", "validation_concentration"):
            print(tables[name].to_string(index=False))
        p = tables["paired_vs_current"]
        print(p.loc[p.metric.eq("log_loss")].to_string(index=False))


if __name__ == "__main__":
    main()
