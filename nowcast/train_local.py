"""Train an offline candidate using the full year before the autumn evaluation."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models import LEVELS, fit_level, score
from runtime import ARTIFACT, select_tier
from sources import HERE, save
from study import compare_routes, eligible

START, END = "2025-09-01", "2026-09-01"


def validation_scores(frame, artifact):
    routes = frame.apply(lambda row: select_tier(row, artifact), axis=1)
    probabilities = np.empty((len(frame), 5))
    for level in routes.unique():
        mask = routes.eq(level)
        probabilities[mask] = artifact["models"][level].predict(frame.loc[mask])
    return score(frame, probabilities)


def train(folder):
    status = {"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}

    def update(state, **details):
        status.update(state=state, updated=datetime.now(timezone.utc).isoformat(), **details)
        save(folder / "status.json", status)
        print(status["updated"], state, details.get("level", ""), flush=True)

    update("loading")
    try:
        data = pd.read_parquet(HERE / "results/sampled.parquet",
                               filters=[("delay", "==", 0), ("day", "<", END)])
        valid = data.loc[data.day >= START]
        frozen = joblib.load(ARTIFACT)
        candidate = {key: frozen[key] for key in ("schema", "timezone", "reference")}
        models, choices, grid = {}, {}, []
        for level in LEVELS:
            update("training", level=level)
            models[level], choices[level], scores = fit_level(eligible(data, level), level, START, END)
            grid.extend({"level": level, "candidate": name, **metrics} for name, metrics in scores.items())
            pd.DataFrame(grid).to_csv(folder / "validation.csv", index=False)

        nwp = compare_routes(eligible(valid, "latest"), models, ["station", "latest"])
        sat = compare_routes(eligible(valid, "local_cloud"), models, [nwp, "local_cloud", "upwind_cloud"])
        candidate.update(version=f"local-{folder.name}", training_end="2025-08-31",
                         selection_end="2026-08-31", models=models, selected=choices,
                         routing={"nwp": nwp, "satellite": sat})
        comparison = {"start": START, "end_exclusive": END, "frozen_version": frozen["version"],
                      "frozen": validation_scores(valid, frozen), "candidate": validation_scores(valid, candidate),
                      "routing": candidate["routing"], "selected": choices}
        joblib.dump(candidate, folder / "model.tmp", compress=3)
        (folder / "model.tmp").replace(folder / "model.joblib")
        save(folder / "comparison.json", comparison)
        update("complete")
    except Exception as exc:
        update("failed", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "started").open("x") as marker:
        marker.write(str(os.getpid()))
    with threadpool_limits(limits=2):
        train(args.run_dir)
