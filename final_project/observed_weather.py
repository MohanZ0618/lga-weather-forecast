"""Does LaGuardia's own observed weather, or more nearby ASOS sites, improve the current best model?

CURRENT is the censored Gaussian of history_study.py: scope "Four stations + trends + HRRR + LAMP/NBM",
trained on 2021-2023 complete rows, L2 chosen from (1, 10, 30) on 2024. Candidates add METAR features
from asos.py (data/asos_features.csv.gz, 10-minute delivery allowance) joined on date and hour; missing
values get the training median (distribution.design); a column present in under half the training rows is
dropped. Candidates choose L2 on 2024 from (1, 3, 10, 30, 100, 300): CURRENT's floor of 1 plus heavier
penalties for the added columns. CURRENT refit on that grid is reported as a control, so the effect of the
wider grid alone can be told apart from the effect of the features. A development run that also allowed L2 = 0.1 and 0.3 moved
CURRENT itself at 12:00 on a 0.0006 validation margin, so the grid keeps CURRENT's floor.
  (a) CURRENT + LGA own weather (wind, pressure, dew point depression, clouds, precipitation)
  (b) (a) + nearby sites (temperature versus LGA, sky cover)
  add-one: CURRENT + each feature group; drop-one: (b) without each group
  sensitivity: (a) and (b) with a 5-minute allowance (timing of the paper's own station features);
               reported, never selected, because the allowance is a data-timing rule, not a model choice
  diagnostic: the observations-only scope "Four stations + trends" with and without (a) and (b), same rows,
              never selected; it asks whether the METAR features carry information once no guidance is given
One system per cutoff is selected on 2024 validation log loss. Every system is scored on the same 359
complete test days; paired differences use 7-day block bootstrap intervals. Writes results/observed_weather/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import shash
from asos import LGA_FEATURES, NEARBY_FEATURES, NEW_STATIONS, STATIONS
from distribution import band_cdf_edges, design, relative_columns, to_bands
from experiment import HOURS, PROBS, month_prior, split_dates, stabilize
from history_study import GROUPS, SCOPE, load
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
TRAIN_START = "2021-01-01"
MIN_PRESENT = 0.5  # a column enters only if at least half the training rows have it (else it is dropped)
CURRENT_L2 = (1.0, 10.0, 30.0)  # history_study.L2
WIDE_L2 = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
FEATURE_GROUPS = {
    "LGA wind": ["LGA_u", "LGA_v", "LGA_gust"],
    "LGA pressure": ["LGA_alti", "LGA_alti_change_3h"],
    "LGA dew point depression": ["LGA_dewpoint_depression"],
    "LGA clouds": ["LGA_sky", "LGA_ceiling_kft"],
    "LGA precipitation": ["LGA_precip_3h"],
    "Nearby temperature vs LGA": [f"{s}_temp_vs_lga" for s in NEW_STATIONS],
    "Nearby sky cover": [f"{s}_sky" for s in STATIONS[1:]],
}
OWN = [g for g in FEATURE_GROUPS if g.startswith("LGA")]
NEARBY = [g for g in FEATURE_GROUPS if g not in OWN]
CURRENT = "CURRENT"
OBS_SCOPE = "Four stations + trends"
OBS = "Diagnostic: observations only"
SAVED = "Saved CURRENT (results/guidance)"
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(LGA_FEATURES + NEARBY_FEATURES)


def systems():
    """name -> (scope, feature groups added to the scope's columns, L2 grid, allowance minutes, selectable)."""
    out = {CURRENT: (SCOPE, [], CURRENT_L2, 10, True), "CURRENT, candidate L2 grid": (SCOPE, [], WIDE_L2, 10, True),
           "(a) + LGA own weather": (SCOPE, OWN, WIDE_L2, 10, True),
           "(b) + LGA own weather + nearby": (SCOPE, OWN + NEARBY, WIDE_L2, 10, True)}
    for group in FEATURE_GROUPS:
        out[f"CURRENT + {group}"] = (SCOPE, [group], WIDE_L2, 10, True)
    for group in FEATURE_GROUPS:
        out[f"(b) without {group}"] = (SCOPE, [g for g in OWN + NEARBY if g != group], WIDE_L2, 10, True)
    out["(a), 5-min allowance (sensitivity)"] = (SCOPE, OWN, WIDE_L2, 5, False)
    out["(b), 5-min allowance (sensitivity)"] = (SCOPE, OWN + NEARBY, WIDE_L2, 5, False)
    out[OBS] = (OBS_SCOPE, [], WIDE_L2, 10, False)
    out[f"{OBS} + (a)"] = (OBS_SCOPE, OWN, WIDE_L2, 10, False)
    out[f"{OBS} + (b)"] = (OBS_SCOPE, OWN + NEARBY, WIDE_L2, 10, False)
    return out


def with_asos(samples, asos, hour, allowance):
    rows = asos.loc[asos.hour.eq(hour) & asos.allowance_min.eq(allowance)].set_index("date")
    return samples.join(rows[LGA_FEATURES + NEARBY_FEATURES], how="left", validate="one_to_one")


def fit_gaussian(samples, columns, train, validation, grid, log, tag):
    """Censored Gaussian as in history_study.fit_window; returns probabilities at the validation-best L2."""
    usable = [c for c in columns if samples.loc[train, c].notna().mean() >= MIN_PRESENT]
    x = design(samples, usable, train)
    z = np.column_stack([np.ones(len(samples)), samples.season_sin, samples.season_cos])
    prior = month_prior(samples.loc[train], samples)
    floor = samples.LGA_max_so_far.to_numpy()
    rise = (samples.target_max - samples.LGA_max_so_far).to_numpy()
    edges = band_cdf_edges(samples)
    target = samples.target_category.to_numpy()
    best = None
    for l2 in grid:
        theta = shash.fit(x[train], z[train], rise[train], l2, shape=False)
        p = stabilize(to_bands(shash.cdf(theta, x, z, edges)), prior, floor)
        assert np.isfinite(p).all()
        loss = daily_scores(p[validation], target[validation])["log_loss"].mean()
        log.append({**tag, "l2": l2, "validation_log_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, p, l2, len(usable))
    return best


def paired_rows(scores, dates, baseline, candidate, split, hour):
    out = []
    for metric in ("log_loss", "rps", "correct"):
        diff = (scores[candidate][metric] - scores[baseline][metric]).to_numpy()
        mean, low, high = block_bootstrap(dates, diff)
        out.append({"hour": hour, "split": split, "baseline": baseline, "candidate": candidate, "metric": metric,
                    "delta": mean, "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "observed_weather")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    asos = pd.read_csv(args.data / "asos_features.csv.gz", parse_dates=["date"])
    saved = pd.read_csv(ROOT / "results" / "guidance" / "test_probabilities.csv.gz", parse_dates=["date"])
    saved = saved.loc[saved.system.eq("Censored Gaussian, HRRR + LAMP/NBM")]
    log, rows, paired, selections, coverage, frames = [], [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            base, complete = load(args.data, hour, hrrr, guidance)
            splits = split_dates(base.index)
            train = splits["train"] & (base.index >= TRAIN_START) & complete
            evaluate = {"validation": splits["validation"] & complete, "test": splits["test"] & complete}
            target = base.target_category.to_numpy()
            scope_columns = {SCOPE: relative_columns(base, SCOPE) + list(GROUPS), OBS_SCOPE: relative_columns(base, OBS_SCOPE)}
            joined = {a: with_asos(base, asos, hour, a) for a in (10, 5)}
            for column in LGA_FEATURES + NEARBY_FEATURES:
                present = joined[10][column].notna().to_numpy()
                coverage.append({"hour": hour, "feature": column,
                                 **{f"{s}_present": float(present[m].mean()) for s, m in
                                    (("train", train), ("validation", evaluate["validation"]),
                                     ("test", evaluate["test"]))}})
            probs, meta = {}, {}
            for name, (scope, groups, grid, allowance, selectable) in systems().items():
                columns = scope_columns[scope] + sum((FEATURE_GROUPS[g] for g in groups), [])
                loss, p, l2, used = fit_gaussian(joined[allowance], columns, train, evaluate["validation"], grid, log,
                                           {"hour": hour, "system": name})
                probs[name] = p
                meta[name] = {"scope": scope, "features": used, "added": used - len(scope_columns[scope]),
                              "allowance_min": allowance, "l2": l2, "validation_log_loss": loss,
                              "selectable": selectable}
            chosen = min((n for n in meta if meta[n]["selectable"]), key=lambda n: meta[n]["validation_log_loss"])
            selections.append({"hour": hour, "selected": chosen, "validation_log_loss": meta[chosen]["validation_log_loss"],
                               "current_validation_log_loss": meta[CURRENT]["validation_log_loss"]})
            for split, mask in evaluate.items():
                dates = base.index[mask]
                scores = {n: daily_scores(p[mask], target[mask]) for n, p in probs.items()}
                if split == "test":
                    reference = saved.loc[saved.hour.eq(hour)].set_index("date").reindex(dates)
                    assert reference[PROBS].notna().all().all(), "saved CURRENT does not cover the test days"
                    scores[SAVED] = daily_scores(reference[PROBS].to_numpy(), target[mask])
                    for n, s in scores.items():
                        rows.append({"hour": hour, "system": n, "selected": n == chosen, "n": int(mask.sum()),
                                     **meta.get(n, {}), **s.mean().to_dict()})
                    for n in (CURRENT, "(a) + LGA own weather", "(b) + LGA own weather + nearby", chosen):
                        frames.append(pd.DataFrame(probs[n][mask], columns=PROBS).assign(
                            system=n, hour=hour, date=dates, target_category=target[mask]))
                for n in scores:
                    if n not in (CURRENT, SAVED) and not n.startswith(OBS):
                        paired.extend(paired_rows(scores, dates, CURRENT, n, split, hour))
                for n in (f"{OBS} + (a)", f"{OBS} + (b)"):
                    paired.extend(paired_rows(scores, dates, OBS, n, split, hour))
                if split == "test":
                    paired.extend(paired_rows(scores, dates, SAVED, CURRENT, split, hour))
                    if chosen != CURRENT:
                        paired.extend(paired_rows(scores, dates, SAVED, chosen, split, hour))
            print(f"{hour:02}:00 done, validation selects {chosen} ({time.time() - started:.0f} s)", flush=True)
    tables = {"validation_grid": log, "test_scores": rows, "paired_vs_current": paired, "selection": selections,
              "coverage": coverage}
    for name, table in tables.items():
        pd.DataFrame(table).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(frames, ignore_index=True).drop_duplicates(["system", "hour", "date"]).to_csv(
        args.out / "test_probabilities.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    with pd.option_context("display.width", 230, "display.float_format", "{:.4f}".format, "display.max_rows", 300,
                           "display.max_colwidth", 48):
        scores = pd.DataFrame(rows)
        print(scores[["hour", "system", "l2", "validation_log_loss", "log_loss", "rps", "correct"]].to_string(index=False))
        frame = pd.DataFrame(paired)
        print(frame.loc[frame.metric.eq("log_loss")].to_string(index=False))
        print(pd.DataFrame(selections).to_string(index=False))
    print(f"total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
