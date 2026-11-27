"""Course models on the guidance forecast: bagging, a small neural network and stacking (Lectures 11 and 13).

Same data, splits, scope and scoring as guidance_study.py and history_study.py: inputs "Four stations +
trends + HRRR + LAMP/NBM", complete rows only, train 2021-2023, choose on 2024 validation, test
September 2025 to August 2026 (359 dates per cutoff). The target is the remaining warming
R = final max - high so far >= 0, with a point mass at zero (censored Gaussian).

1. Bagging. B = 50 bootstrap resamples of training weeks (7-day blocks, because nearby days share weather),
   one censored Gaussian per resample at the validation-chosen L2, band probabilities averaged.
2. Neural network (numpy only). One tanh hidden layer maps the standardized design() inputs to mu and
   log sigma of the censored Gaussian for R. Minibatch Adam on the censored negative log-likelihood plus
   L2 weight decay, backpropagation written out by hand (checked against finite differences in
   test_class_models.py), early stopping on 2024 validation NLL, band probabilities averaged over five
   fixed seeds. Width, decay and architecture are chosen on 2024 band log loss: a plain network, or one
   with an extra linear path from inputs to outputs, which contains the linear censored model as a
   special case (hidden-to-output weights at zero).
3. Stacking. Linear pools of CURRENT (the censored Gaussian refit here) with the paper's classifier
   families on the same inputs (experiment.fit_scope, as in distribution.run_cohort) and with the
   network: equal weights, and weights chosen on 2024 validation. A fitted weight makes its own 2024
   score optimistic, so for selection the weighted pools are scored by 2-fold cross-fitting inside 2024
   (weights chosen on alternate weeks, scored on the others).

Every candidate is scored on validation and test and paired with CURRENT's saved test probabilities
(results/guidance/test_probabilities.csv.gz) by date, resampling 7-day blocks. Writes
results/class_models/. Offline, two threads, about one minute.
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

import experiment
import shash
from distribution import band_cdf_edges, design, original_columns, relative_columns, to_bands
from experiment import HOURS, PROBS, fit_scope, month_prior, split_dates, stabilize
from history_study import SCOPE, load
from supplement import block_bootstrap, daily_scores

ROOT = Path(__file__).resolve().parent
CURRENT_FILE = ROOT / "results" / "guidance" / "test_probabilities.csv.gz"
CURRENT_SAVED = "Censored Gaussian, HRRR + LAMP/NBM"
PAPER = "Paper: selected HRRR system"
CLASSIFIERS_SAVED = "Classifiers, HRRR + LAMP/NBM"
CURRENT = "Censored Gaussian (CURRENT, refit)"
L2_GRID = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)  # the grid guidance_study.py used for CURRENT
BAGS = 50
BLOCK_DAYS = 7
WIDTHS = (4, 8, 16, 32)
DECAYS = (1e-3, 1e-2, 3e-2)
ARCHITECTURES = ("plain", "linear path")
SEEDS = (0, 1, 2, 3, 4)
LEARNING_RATE, BATCH, MAX_EPOCHS, PATIENCE = 0.01, 64, 400, 40
WEIGHT_STEP = 0.05
HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)


# Censored Gaussian likelihood ----------------------------------------------------------------

def censored_terms(mu, log_sigma, rise):
    """Per-row negative log-likelihood of R = max(0, R*), R* ~ Normal(mu, sigma), and its derivatives.

    R > 0: -log density = log sigma + t^2 / 2 + log sqrt(2 pi), t = (R - mu) / sigma.
    R = 0: -log P(R* <= 0) = -log Phi(a), a = -mu / sigma.
    Returns (nll, d nll / d mu, d nll / d log sigma).
    """
    sigma = np.exp(log_sigma)
    zero = rise <= 1e-9
    t = (rise - mu) / sigma
    a = -mu / sigma
    log_cdf = norm.logcdf(a)
    ratio = np.exp(norm.logpdf(a) - log_cdf)  # inverse Mills ratio phi(a) / Phi(a)
    nll = np.where(zero, -log_cdf, log_sigma + 0.5 * t * t + HALF_LOG_2PI)
    d_mu = np.where(zero, ratio / sigma, -t / sigma)
    d_log_sigma = np.where(zero, a * ratio, 1.0 - t * t)
    return nll, d_mu, d_log_sigma


def gaussian_bands(mu, sigma, edges):
    """Band probabilities from P(R < e) = Phi((e - mu) / sigma) for e > 0 (R is never negative)."""
    return to_bands(np.where(edges > 0, norm.cdf((edges - mu[:, None]) / sigma[:, None]), 0.0))


# Neural network ------------------------------------------------------------------------------

def mlp_init(k, width, rng, mean_rise, linear_path=False):
    """Glorot-scale input weights; small output weights so training starts near a constant Gaussian.
    The optional linear path A (inputs straight to mu and log sigma) starts at zero."""
    params = {"W1": rng.normal(0.0, 1.0 / np.sqrt(k), (k, width)), "b1": np.zeros(width),
              "W2": rng.normal(0.0, 0.1 / np.sqrt(width), (width, 2)), "b2": np.array([mean_rise, 0.0])}
    if linear_path:
        params["A"] = np.zeros((k, 2))
    return params


def mlp_forward(params, x):
    hidden = np.tanh(x @ params["W1"] + params["b1"])
    out = hidden @ params["W2"] + params["b2"]
    if "A" in params:
        out = out + x @ params["A"]
    return hidden, out[:, 0], out[:, 1]


def mlp_loss_and_grads(params, x, rise, decay):
    """Mean censored NLL + decay * (sum of squared weights), and its gradient by backpropagation."""
    hidden, mu, log_sigma = mlp_forward(params, x)
    nll, d_mu, d_log_sigma = censored_terms(mu, log_sigma, rise)
    weights = [key for key in ("W1", "W2", "A") if key in params]  # biases are not decayed
    loss = nll.mean() + decay * sum((params[key] ** 2).sum() for key in weights)
    d_out = np.column_stack([d_mu, d_log_sigma]) / len(rise)   # d loss / d (mu, log sigma)
    d_hidden = d_out @ params["W2"].T
    d_pre = d_hidden * (1.0 - hidden * hidden)                 # tanh'(a) = 1 - tanh(a)^2
    grads = {"W2": hidden.T @ d_out, "b2": d_out.sum(axis=0), "W1": x.T @ d_pre, "b1": d_pre.sum(axis=0)}
    if "A" in params:
        grads["A"] = x.T @ d_out
    for key in weights:
        grads[key] = grads[key] + 2 * decay * params[key]
    return loss, grads


def mlp_train(x, rise, x_val, rise_val, width, decay, seed, linear_path=False, lr=LEARNING_RATE, batch=BATCH,
              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Minibatch Adam with early stopping on validation NLL. Rise is divided by its training spread
    (a pure rescaling keeps the censoring point at zero). Returns (predict function, best epoch, curve)."""
    scale = rise.std()
    r, r_val = rise / scale, rise_val / scale
    rng = np.random.default_rng(seed)
    params = mlp_init(x.shape[1], width, rng, r.mean(), linear_path)
    moments = {key: (np.zeros_like(value), np.zeros_like(value)) for key, value in params.items()}
    beta1, beta2, step = 0.9, 0.999, 0
    best, best_epoch, curve = np.inf, 0, []
    best_params = {key: value.copy() for key, value in params.items()}
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(r))
        for start in range(0, len(r), batch):
            rows = order[start:start + batch]
            _, grads = mlp_loss_and_grads(params, x[rows], r[rows], decay)
            step += 1
            for key, g in grads.items():
                m, v = moments[key]
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * g * g
                params[key] -= lr * (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + 1e-8)
        _, mu, log_sigma = mlp_forward(params, x_val)
        val = censored_terms(mu, log_sigma, r_val)[0].mean()
        curve.append(val)
        if val < best - 1e-6:
            best, best_epoch = val, epoch
            best_params = {key: value.copy() for key, value in params.items()}
        elif epoch - best_epoch >= patience:
            break

    def predict(x_new):
        _, mu, log_sigma = mlp_forward(best_params, x_new)
        return scale * mu, scale * np.exp(np.clip(log_sigma, -10, 10))

    return predict, best_epoch, curve


# Bagging -------------------------------------------------------------------------------------

def week_blocks(dates, block_days=BLOCK_DAYS):
    """Row indices of each 7-calendar-day block of the given dates."""
    blocks = np.asarray((pd.DatetimeIndex(dates) - pd.Timestamp(dates.min())).days // block_days)
    return [np.flatnonzero(blocks == b) for b in np.unique(blocks)]


def block_resample(blocks, rng):
    """One bootstrap sample: as many weeks as there are, drawn with replacement, all days of each."""
    picks = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in picks])


# Stacking ------------------------------------------------------------------------------------

def simplex(members, step=WEIGHT_STEP):
    """Every weight vector on a grid of the given step that sums to one."""
    n = round(1 / step)
    return [np.array(c) / n for c in itertools.product(range(n + 1), repeat=members) if sum(c) == n]


def pool(members, weights):
    return sum(w * p for w, p in zip(weights, members))


def mean_log_loss(p, target):
    return float(daily_scores(p, target)["log_loss"].mean())


def best_weights(members, target, mask, grid):
    losses = [mean_log_loss(pool(members, w)[mask], target[mask]) for w in grid]
    return grid[int(np.argmin(losses))], float(min(losses))


def cross_fitted_loss(members, target, validation, folds, grid):
    """Choose weights on one fold of validation, score on the other, average over all validation days."""
    total = 0.0
    for fold in np.unique(folds[validation]):
        fit = validation & (folds != fold)
        score = validation & (folds == fold)
        weights, _ = best_weights(members, target, fit, grid)
        total += daily_scores(pool(members, weights)[score], target[score])["log_loss"].sum()
    return total / validation.sum()


# Study ---------------------------------------------------------------------------------------

def censored_gaussian(x, z, rise, train, val_loss, bands):
    """CURRENT: censored Gaussian (shash.fit with the shape fixed), L2 chosen on 2024 validation."""
    log, best = [], None
    for l2 in L2_GRID:
        theta = shash.fit(x[train], z[train], rise[train], l2, shape=False)
        p = bands(theta)
        loss = val_loss(p)
        log.append({"model": CURRENT, "setting": f"l2={l2}", "validation_log_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, l2, theta, p)
    return best[1], best[2], best[3], log


def bagging(x, z, rise, train, dates, l2, theta, bands, val_loss, test, seed=4260):
    blocks = week_blocks(dates[train])
    rows_train = np.flatnonzero(train)
    rng = np.random.default_rng(seed)
    total, member_val, member_mu, member_sigma, curve, unique = 0.0, [], [], [], [], []
    for b in range(1, BAGS + 1):
        rows = rows_train[block_resample(blocks, rng)]
        unique.append(len(np.unique(rows)) / len(rows_train))
        theta_b = shash.fit(x[rows], z[rows], rise[rows], l2, shape=False, start=theta)
        p = bands(theta_b)
        total = total + p
        member_val.append(val_loss(p))
        member_mu.append(theta_b[0] + x[test] @ theta_b[1:x.shape[1] + 1])
        member_sigma.append(np.exp(z[test] @ theta_b[x.shape[1] + 1:x.shape[1] + 1 + z.shape[1]]))
        if b in (1, 5, 10, 25, 50):
            curve.append({"members": b, "validation_log_loss": val_loss(total / b)})
    bagged = total / BAGS
    spread = float(np.std(member_mu, axis=0).mean())
    summary = {"l2": l2, "training_days": int(train.sum()), "weeks": len(blocks),
               "unique_day_fraction_per_resample": float(np.mean(unique)),
               "single_fit_validation": val_loss(bands(theta)),
               "member_mean_validation": float(np.mean(member_val)),
               "member_best_validation": float(np.min(member_val)),
               "member_worst_validation": float(np.max(member_val)),
               "bagged_validation": val_loss(bagged), "test_mu_sd_across_members_f": spread,
               "test_mean_sigma_f": float(np.mean(member_sigma))}
    return bagged, summary, curve


def network(x, rise, train, validation, finish, val_loss):
    """Grid over architecture, width and decay; each setting is the 5-seed average of band probabilities,
    scored on 2024 band log loss."""
    log, best = [], None
    for architecture, width, decay in itertools.product(ARCHITECTURES, WIDTHS, DECAYS):
        probabilities, epochs, single = 0.0, [], []
        for seed in SEEDS:
            predict, epoch, _ = mlp_train(x[train], rise[train], x[validation], rise[validation], width, decay, seed,
                                          linear_path=architecture == "linear path")
            member = finish(*predict(x))
            probabilities = probabilities + member
            epochs.append(epoch)
            single.append(val_loss(member))
        probabilities = probabilities / len(SEEDS)
        loss = val_loss(probabilities)
        setting = f"{architecture}, width={width}, decay={decay}"
        log.append({"model": "MLP", "setting": setting, "validation_log_loss": loss,
                    "single_seed_validation_mean": float(np.mean(single)), "best_epochs": " ".join(map(str, epochs))})
        if best is None or loss < best[0]:
            best = (loss, setting, probabilities)
    return best[1], best[2], log


def classifiers(frame, splits, prior):
    """The paper's classifier families on the guidance inputs, as in distribution.run_cohort."""
    experiment.feature_columns = relative_columns
    try:
        tuning = []
        predictions, selected = fit_scope(frame, splits, SCOPE, prior, tuning)
    finally:
        experiment.feature_columns = original_columns
    return predictions, selected, tuning


def run_hour(hour, frame, saved):
    splits = split_dates(frame.index)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    dates = frame.index
    target = frame.target_category.to_numpy()
    floor = frame.LGA_max_so_far.to_numpy()
    rise = (frame.target_max - frame.LGA_max_so_far).to_numpy()
    edges = band_cdf_edges(frame)
    prior = month_prior(frame.loc[train], frame)
    x = design(frame, relative_columns(frame, SCOPE), train)
    z = np.column_stack([np.ones(len(frame)), frame.season_sin, frame.season_cos])

    def val_loss(p):
        return mean_log_loss(p[validation], target[validation])

    def finish(mu, sigma):
        return stabilize(gaussian_bands(mu, sigma, edges), prior, floor)

    def bands(theta):
        return stabilize(to_bands(shash.cdf(theta, x, z, edges)), prior, floor)

    timing, log = {}, []
    clock = time.perf_counter()
    l2, theta, current, grid = censored_gaussian(x, z, rise, train, val_loss, bands)
    log += grid
    timing["censored_gaussian_s"] = time.perf_counter() - clock

    clock = time.perf_counter()
    bagged, bag_summary, bag_curve = bagging(x, z, rise, train, dates, l2, theta, bands, val_loss, test)
    timing["bagging_s"] = time.perf_counter() - clock

    clock = time.perf_counter()
    mlp_setting, mlp, grid = network(x, rise, train, validation, finish, val_loss)
    log += grid
    timing["mlp_s"] = time.perf_counter() - clock

    clock = time.perf_counter()
    families, selected, tuning = classifiers(frame, splits, prior)
    log += [{"model": f"Classifier: {row['model']}", "setting": row["config"],
             "validation_log_loss": row["validation_log_loss"]} for row in tuning]
    timing["classifiers_s"] = time.perf_counter() - clock
    classifier = families[selected]
    if selected == "Two-model probability average":
        selected = "average of " + next(row["config"] for row in tuning if row["model"] == selected)

    systems = {CURRENT: current, f"Bagged censored Gaussian (B={BAGS}, weekly blocks)": bagged,
               "MLP (one tanh layer, 5 seeds)": mlp, f"Paper classifiers, guidance inputs ({selected})": classifier,
               "Random forest, guidance inputs": families["Random forest"],
               "XGBoost, guidance inputs": families["XGBoost"]}
    selection_loss = {name: val_loss(p) for name, p in systems.items()}
    weeks = np.asarray((dates - pd.Timestamp("2024-01-01")).days // BLOCK_DAYS)
    folds = np.where(validation, weeks % 2, -1)
    weights_used, pool_weights = [], {}
    for label, members in (("CURRENT + classifiers", [current, classifier]),
                           ("CURRENT + MLP", [current, mlp]),
                           ("CURRENT + classifiers + MLP", [current, classifier, mlp])):
        equal = np.full(len(members), 1 / len(members))
        systems[f"{label}, equal weights"] = pool(members, equal)
        selection_loss[f"{label}, equal weights"] = val_loss(systems[f"{label}, equal weights"])
        grid = simplex(len(members))
        weights, in_sample = best_weights(members, target, validation, grid)
        name = f"{label}, validation weights"
        systems[name] = pool(members, weights)
        selection_loss[name] = cross_fitted_loss(members, target, validation, folds, grid)
        pool_weights[name] = " / ".join(f"{w:.2f}" for w in weights)
        weights_used.append({"hour": hour, "pool": label, "weights": pool_weights[name],
                             "validation_in_sample": in_sample, "validation_cross_fitted": selection_loss[name]})
        for w in grid:
            log.append({"model": f"Pool: {label}", "setting": " / ".join(f"{v:.2f}" for v in w),
                        "validation_log_loss": mean_log_loss(pool(members, w)[validation], target[validation])})

    # Saved CURRENT and paper probabilities on the same test dates.
    test_dates = dates[test]
    reference = {}
    for name in (CURRENT_SAVED, PAPER, CLASSIFIERS_SAVED):
        rows = saved.loc[saved.system.eq(name) & saved.hour.eq(hour)].set_index("date").loc[test_dates]
        assert (rows.target_category.to_numpy() == target[test]).all()
        reference[name] = rows[PROBS].to_numpy()

    chosen = min(selection_loss, key=selection_loss.get)
    scores, paired, frames = [], [], []
    base = daily_scores(reference[CURRENT_SAVED], target[test])
    val_base = daily_scores(current[validation], target[validation])
    for name, p in systems.items():
        row = {"hour": hour, "system": name, "selection_validation_log_loss": selection_loss[name],
               "selected_by_validation": name == chosen}
        for split, mask in (("validation", validation), ("test", test)):
            s = daily_scores(p[mask], target[mask]).mean()
            row.update({f"{split}_n": int(mask.sum()), f"{split}_log_loss": s.log_loss, f"{split}_rps": s.rps,
                        f"{split}_accuracy": s.correct})
        scores.append(row)
        for split, mask, baseline, label in (("test", test, base, f"{CURRENT_SAVED} (saved)"),
                                             ("validation", validation, val_base, CURRENT)):
            if split == "validation" and name == CURRENT:
                continue
            daily = daily_scores(p[mask], target[mask])
            for metric in ("log_loss", "rps", "correct"):
                mean, low, high = block_bootstrap(dates[mask], (daily[metric] - baseline[metric]).to_numpy())
                paired.append({"hour": hour, "split": split, "candidate": name, "baseline": label, "metric": metric,
                               "delta": mean, "ci95_low": low, "ci95_high": high,
                               "excludes_zero": low > 0 or high < 0})
        frames.append(pd.DataFrame(p[test], columns=PROBS).assign(system=name, hour=hour, date=test_dates,
                                                                  target_category=target[test]))
    for name, p in reference.items():
        if name == CLASSIFIERS_SAVED:
            continue
        s = daily_scores(p, target[test]).mean()
        scores.append({"hour": hour, "system": f"{name} (saved)", "test_n": int(test.sum()),
                       "test_log_loss": s.log_loss, "test_rps": s.rps, "test_accuracy": s.correct})

    refit = {"hour": hour, "chosen_l2": l2,
             "refit_test_log_loss": mean_log_loss(current[test], target[test]),
             "saved_test_log_loss": float(base.log_loss.mean()),
             "max_abs_probability_difference": float(np.abs(current[test] - reference[CURRENT_SAVED]).max()),
             "classifier_refit_test_log_loss": mean_log_loss(classifier[test], target[test]),
             "classifier_saved_test_log_loss": mean_log_loss(reference[CLASSIFIERS_SAVED], target[test]),
             "mlp_setting": mlp_setting, "classifier_selected": selected}
    print(f"{hour:02}:00 seconds: " + ", ".join(f"{key[:-2]} {value:.1f}" for key, value in timing.items()), flush=True)
    refit["refit_minus_saved"] = refit["refit_test_log_loss"] - refit["saved_test_log_loss"]
    for entry in log:
        entry["hour"] = hour
    bag_summary["hour"] = hour
    for entry in bag_curve:
        entry["hour"] = hour
    return {"scores": scores, "paired": paired, "frames": frames, "log": log, "refit": [refit],
            "bagging": [bag_summary], "bag_curve": bag_curve, "weights": weights_used,
            "selection": [selection_row(hour, chosen, selection_loss, paired, pool_weights)]}


def selection_row(hour, chosen, selection_loss, paired, pool_weights):
    """The 2024 pick at one cutoff, its paired log-loss difference from CURRENT on both splits, and whether a
    rule that keeps the simpler CURRENT unless the 2024 interval excludes zero would still switch."""
    row = {"hour": hour, "selected": chosen, "pool_weights": pool_weights.get(chosen, ""),
           "selection_validation_log_loss": selection_loss[chosen],
           "current_validation_log_loss": selection_loss[CURRENT]}
    for entry in paired:
        if entry["candidate"] == chosen and entry["metric"] == "log_loss":
            row.update({f"{entry['split']}_delta": entry["delta"], f"{entry['split']}_ci95_low": entry["ci95_low"],
                        f"{entry['split']}_ci95_high": entry["ci95_high"]})
    row["switches_under_interval_rule"] = chosen != CURRENT and row.get("validation_ci95_high", 0.0) < 0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "class_models")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    hrrr = pd.read_csv(args.data / "hrrr_features.csv.gz", parse_dates=["date"])
    guidance = pd.read_csv(args.data / "guidance_features.csv.gz", parse_dates=["date"])
    saved = pd.read_csv(CURRENT_FILE, parse_dates=["date"])
    tables = {}
    with threadpool_limits(limits=2):
        for hour in HOURS:
            samples, complete = load(args.data, hour, hrrr, guidance)
            result = run_hour(hour, samples.loc[complete], saved)
            for key, rows in result.items():
                tables.setdefault(key, []).extend(rows)
            print(f"{hour:02}:00 complete ({time.perf_counter() - start:.0f} s)", flush=True)
    names = {"log": "validation_grid", "scores": "scores", "paired": "paired_vs_current", "refit": "reproduction",
             "bagging": "bagging", "bag_curve": "bagging_curve", "weights": "pool_weights", "selection": "selection"}
    for key, name in names.items():
        pd.DataFrame(tables[key]).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(tables["frames"], ignore_index=True).to_csv(args.out / "test_probabilities.csv.gz", index=False,
                                                          compression={"method": "gzip", "mtime": 0})
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 200,
                           "display.max_colwidth", 60):
        print(pd.DataFrame(tables["refit"]).to_string(index=False))
        print(pd.DataFrame(tables["bagging"]).to_string(index=False))
        print(pd.DataFrame(tables["weights"]).to_string(index=False))
        cols = ["hour", "system", "selection_validation_log_loss", "validation_log_loss", "test_log_loss", "test_rps",
                "test_accuracy", "selected_by_validation"]
        print(pd.DataFrame(tables["scores"])[cols].to_string(index=False))
        paired = pd.DataFrame(tables["paired"])
        print(paired.loc[paired.metric.eq("log_loss")].drop(columns=["baseline", "metric"]).to_string(index=False))
        print(pd.DataFrame(tables["selection"]).to_string(index=False))
    print(f"total {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
