"""Does station guidance (LAMP and NBM) improve the paper's best system? Run after guidance.py build.

Cohort: the paper's matched-HRRR dates that also have a usable LAMP and NBM run (same splits: train
2021-2023, select on 2024, test September 2025 to August 2026). Two input sets are fit with the paper's
classifier families and with the distributional models in distribution.py:
  - stations + trends + LAMP/NBM (station guidance in place of HRRR)
  - stations + trends + HRRR + LAMP/NBM
Each is compared on identical test dates with the paper's selected HRRR system, pairing by date and
resampling 7-day blocks. Writes results/guidance/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from distribution import run_cohort
from experiment import FORECAST_FEATURES, HOURS, PROBS, season_features, split_dates
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
PAPER = "Four stations + trends + HRRR"
REQUIRED = ["lamp_remaining_max", "lamp_now", "lamp_cloud", "lamp_u", "lamp_v", "nbm_max", "nbm_max_sd"]
NAMES = {"Classifiers + relative features": "Classifiers", "Censored Gaussian": "Censored Gaussian",
         "Quantile boosting": "Quantile boosting"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "guidance")
    parser.add_argument("--guidance", type=Path, default=ROOT / "data" / "guidance_features.csv.gz")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paper = pd.read_csv(ROOT / "results" / "daily_predictions.csv", parse_dates=["date"])
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.guidance, parse_dates=["date"])
    log, rows, paired, coverage, selections, frames = [], [], [], [], [], []
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples = season_features(pd.read_csv(args.data / f"samples_{hour:02}.csv.gz",
                                                  parse_dates=["date"], index_col="date").sort_index())
            matched = samples.join(hrrr.loc[hrrr.hour == hour].set_index("date")[FORECAST_FEATURES],
                                   how="inner", validate="one_to_one")
            joined = matched.join(guidance.loc[guidance.hour == hour].set_index("date")[REQUIRED],
                                  how="inner", validate="one_to_one").dropna(subset=REQUIRED)
            test = split_dates(joined.index)["test"]
            dates = joined.index[test]
            coverage.append({"hour": hour, "matched_hrrr_test_dates": int(split_dates(matched.index)["test"].sum()),
                             "with_guidance": int(test.sum()), "train_dates": int(split_dates(joined.index)["train"].sum())})
            target = joined.target_category.to_numpy()[test]
            reference = paper.loc[paper.cohort.eq("Matched HRRR") & paper.hour.eq(hour) & paper.scope.eq(PAPER)]
            systems = {"Paper: selected HRRR system": reference.set_index("date").loc[dates, PROBS].to_numpy()}
            validation = split_dates(joined.index)["validation"]
            val_target = joined.target_category.to_numpy()[validation]
            val_loss = {}
            for scope, label in (("Four stations + trends + LAMP/NBM", "LAMP/NBM"),
                                 ("Four stations + trends + HRRR + LAMP/NBM", "HRRR + LAMP/NBM")):
                models, _, _ = run_cohort(hour, joined, "Matched HRRR", scope, log)
                for name, p in models.items():
                    systems[f"{NAMES[name]}, {label}"] = p[test]
                    val_loss[f"{NAMES[name]}, {label}"] = daily_scores(p[validation], val_target)["log_loss"].mean()
            # Choose one system per hour on 2024 validation log loss only; the test set picks nothing.
            chosen = min(val_loss, key=val_loss.get)
            selections.append({"hour": hour, "selected": chosen, "validation_log_loss": val_loss[chosen],
                               **{f"val_{k}": v for k, v in val_loss.items()}})
            frames.append(pd.concat([pd.DataFrame(p, columns=PROBS).assign(system=name) for name, p in systems.items()])
                          .assign(hour=hour, date=np.tile(dates, len(systems)), target_category=np.tile(target, len(systems))))
            scores = {name: daily_scores(p, target) for name, p in systems.items()}
            base = scores["Paper: selected HRRR system"]
            for name, p in systems.items():
                rows.append({"hour": hour, "system": name, "n": len(target), **scores[name].mean().to_dict()})
                if name.startswith("Paper"):
                    continue
                for metric in ("log_loss", "rps", "correct"):
                    mean, low, high = block_bootstrap(dates, (scores[name][metric] - base[metric]).to_numpy())
                    paired.append({"hour": hour, "system": name, "metric": metric, "delta_vs_paper": mean,
                                   "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0})
            print(f"{hour:02}:00 complete", flush=True)
    pd.DataFrame(coverage).to_csv(args.out / "coverage.csv", index=False)
    pd.DataFrame(log).to_csv(args.out / "validation_grid.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out / "test_scores.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out / "paired_vs_paper.csv", index=False)
    pd.DataFrame(selections).to_csv(args.out / "selection.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(args.out / "test_probabilities.csv.gz", index=False,
                                                compression={"method": "gzip", "mtime": 0})
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(coverage).to_string(index=False))
        print(pd.DataFrame(selections)[["hour", "selected", "validation_log_loss"]].to_string(index=False))
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
