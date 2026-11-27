"""Section 7 of the paper (station guidance and a censored distribution), built from the result CSVs.

make_paper.py calls extension_section(...) and extension_references(...). Every number below is read
from the outputs of guidance_study.py, class_eval.py, class_models.py, history_study.py, mos_history.py,
observed_weather.py and nonstationarity.py, so rerunning those scripts and make_paper.py keeps the text
and the tables consistent.
"""

import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HOURS = (9, 12, 15)
SELECTED = "Censored Gaussian, HRRR + LAMP/NBM"
PAPER = "Paper: selected HRRR system"


def _rows(path):
    with (RESULTS / path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _pick(rows, **match):
    hits = [row for row in rows if all(str(row[key]) == str(value) for key, value in match.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {match}, found {len(hits)}")
    return hits[0]


def _signed(value):
    """Three decimals, or four when three would round a nonzero bound to zero."""
    return f"{value:+.4f}" if 0 < abs(value) < 0.0005 else f"{value:+.3f}"


def numbers():
    scores = _rows("guidance/test_scores.csv")
    paired = _rows("guidance/paired_vs_paper.csv")
    selection = {int(row["hour"]): row["selected"] for row in _rows("guidance/selection.csv")}
    out = {"table": [], "selected": selection, "significant": []}
    for hour in HOURS:
        chosen = selection[hour]
        new, old = _pick(scores, hour=hour, system=chosen), _pick(scores, hour=hour, system=PAPER)
        change = _pick(paired, hour=hour, system=chosen, metric="log_loss")
        model = "Censored" if chosen.startswith("Censored") else "Quantile boosting"
        out["table"].append([f"{hour:02}:00", model, f"{float(old['log_loss']):.3f}", f"{float(new['log_loss']):.3f}",
                             f"{_signed(float(change['delta_vs_paper']))} [{_signed(float(change['ci95_low']))}, "
                             f"{_signed(float(change['ci95_high']))}]",
                             f"{float(old['correct']):.1%} to {float(new['correct']):.1%}"])
        if float(change["ci95_high"]) < 0:
            out["significant"].append(f"{hour:02}:00")
    out["n"] = _pick(scores, hour=9, system=SELECTED)["n"]
    import gzip
    out["zero"] = {}
    for hour in HOURS:
        with gzip.open(RESULTS.parent / "data" / f"samples_{hour:02}.csv.gz", "rt", encoding="utf-8") as file:
            days = [row for row in csv.DictReader(file) if row["target_max"] and row["LGA_max_so_far"]]
        out["zero"][hour] = sum(float(r["target_max"]) == float(r["LGA_max_so_far"]) for r in days) / len(days)
    out["all_selected"] = all(name == SELECTED for name in selection.values())

    curves = _rows("class_eval/learning_curves.csv")
    gauss = [row for row in curves if row["scope"] == "guidance" and row["model"] == "Censored Gaussian"
             and row["hour"] == "9"]
    by_n = {int(row["n_train"]): row for row in gauss}
    out["lc_small"], out["lc_full"] = float(by_n[400]["validation_log_loss"]), float(by_n[max(by_n)]["validation_log_loss"])
    out["lc_full_n"] = max(by_n)
    over = _rows("class_eval/overfitting.csv")
    paper_fit = [row for row in over if row["hour"] == "9" and "average" in row["system"].lower()]
    out["paper_train"] = float(paper_fit[0]["train_log_loss"]) if paper_fit else None
    out["paper_val"] = float(paper_fit[0]["validation_log_loss"]) if paper_fit else None

    history = _rows("history/test_scores.csv")
    base = {h: float(_pick(history, hour=h, system="Censored Gaussian, 2021-2023")["log_loss"]) for h in HOURS}
    shash = [float(_pick(history, hour=h, system="Censored SHASH, 2021-2023")["log_loss"]) - base[h] for h in HOURS]
    masked = [float(_pick(history, hour=h, system="Censored Gaussian, 2010-2023")["log_loss"]) - base[h] for h in HOURS]
    out["shash"] = (min(shash), max(shash))
    out["masked"] = (min(masked), max(masked))

    mos = _rows("mos_history/paired_vs_current.csv") if (RESULTS / "mos_history/paired_vs_current.csv").exists() else []
    out["mos_long"] = {int(row["hour"]): float(row["delta"]) for row in mos if row.get("metric") == "log_loss"
                       and row["candidate"] == "Stations + trends + GFS + NAM MOS, 2010-2023"
                       and row["baseline"] == "Stations + trends + GFS + NAM MOS, 2021-2023"}

    observed = _rows("observed_weather/paired_vs_current.csv")
    against = [row for row in observed if row["split"] == "test" and row["metric"] == "log_loss"
               and row["baseline"] == "CURRENT" and row["candidate"] != "CURRENT, candidate L2 grid"]
    out["observed_largest_gain"] = max(-float(row["delta"]) for row in against)
    out["observed_only"] = float(_pick(observed, hour=9, split="test", metric="log_loss",
                                       baseline="Diagnostic: observations only",
                                       candidate="Diagnostic: observations only + (a)")["delta"])
    out["drift"] = None
    if (RESULTS / "nonstationarity/level_shifts.csv").exists():
        shifts = _rows("nonstationarity/level_shifts.csv")
        series = {"GFS MOS": "GFS MOS daytime max", "NAM MOS": "NAM MOS daytime max", "LAMP": "LAMP remaining max",
                  "HRRR": "HRRR remaining max", "LGA relative to JFK, EWR, and TEB":
                  "LGA minus mean of JFK, EWR, TEB, cutoff temperature"}
        breaks = {name: _pick(shifts, series=label, statistic="mean", data="2010-2026 (descriptive)")
                  for name, label in series.items()}
        drift_pairs = _rows("nonstationarity/paired_vs_current.csv")
        recency = [float(_pick(drift_pairs, hour=h, metric="log_loss",
                                candidate="CURRENT inputs 2021-2023, half-life 0.5 y")["delta_vs_current"]) for h in HOURS]
        walk = _pick(drift_pairs, hour=9, metric="log_loss", candidate="Walk-forward CURRENT, no decay")
        out["drift"] = {"breaks": breaks, "recency": recency, "walk": float(walk["delta_vs_current"])}
    return out


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _month(text):
    year, month = text.split("-")
    return f"{MONTHS[int(month) - 1]} {year}"


def drift_text(drift):
    parts = [f"{name} ({float(row['shift']):+.1f}°F from {_month(row['break_month'])})"
             for name, row in drift["breaks"].items()]
    rows = [row for row in _rows("nonstationarity/paired_vs_current.csv") if row["metric"] == "log_loss"]
    def series(candidate):
        picked = [_pick(rows, hour=h, candidate=candidate) for h in HOURS]
        return interval_summary([float(r["delta_vs_current"]) for r in picked], [float(r["ci95_low"]) for r in picked],
                                [float(r["ci95_high"]) for r in picked])
    weighted, refit = series("CURRENT inputs 2021-2023, half-life 0.5 y"), series("Walk-forward CURRENT, no decay")
    outside = weighted[1] + weighted[2] + refit[1] + refit[2]
    return ("Forecast models and sensors also changed during the record. Monthly forecast errors show level "
            "shifts in " + ", ".join(parts[:-1]) + ", and " + parts[-1] + ". The NBM upgrade to version 5 in "
            "May 2026 caused no significant shift. Weighting recent training days (a six-month half-life) changed "
            f"test loss by {weighted[0]}, and refitting every month by {refit[0]}"
            + ("; none of these changes is outside its 95% interval." if not outside else
               f"; only {', '.join(outside)} falls outside its 95% interval.")
            + " One validation year cannot show which of these adjustments would hold up, so we report the "
            "fixed model.")


def learning_curve_figure(path):
    """Figure 2 in the style of Figure 1 (experiment.py): one row, three cutoffs, the same two colors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in _rows("class_eval/learning_curves.csv")
            if row["scope"] == "guidance" and int(row["n_train"]) >= 200]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), sharex=True)
    for hour, ax in zip(HOURS, axes):
        for model, color in (("Censored Gaussian", "#146e87"), ("Quantile boosting", "#7d8994")):
            points = sorted((int(r["n_train"]), float(r["train_log_loss"]), float(r["validation_log_loss"]))
                            for r in rows if int(r["hour"]) == hour and r["model"] == model)
            days = [point[0] for point in points]
            ax.plot(days, [point[2] for point in points], "-o", color=color, markersize=3.5, linewidth=1.4,
                    label=f"{model}, 2024 validation")
            ax.plot(days, [point[1] for point in points], "--", color=color, linewidth=1, label=f"{model}, training")
        ax.set(title=f"{hour:02}:00", xlabel="Most recent training days", xscale="log")
        ax.minorticks_off()
        ax.set_xticks([200, 400, 700, 1040], ["200", "400", "700", "1,040"])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Log loss")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _aligned_table(headers, rows, widths, numeric):
    """Same look as make_paper.table, with the numeric columns right-aligned and the text columns left."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from xml.sax.saxutils import escape

    cell = ParagraphStyle("ExtCell", fontName="Helvetica", fontSize=10, leading=12, textColor=colors.HexColor("#162939"))
    right = ParagraphStyle("ExtRight", parent=cell, alignment=TA_RIGHT)
    head = ParagraphStyle("ExtHead", parent=cell, fontName="Helvetica-Bold", leading=13, textColor=colors.HexColor("#126078"),
                          spaceBefore=7, spaceAfter=8)
    head_right = ParagraphStyle("ExtHeadRight", parent=head, alignment=TA_RIGHT)
    content = [[Paragraph(escape(str(h)), head_right if i in numeric else head) for i, h in enumerate(headers)]]
    content += [[Paragraph(escape(str(c)), right if i in numeric else cell) for i, c in enumerate(row)] for row in rows]
    result = Table(content, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEBEF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F6")]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#BDD1D9")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return result


def _change(value):
    return "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"


def _ci_high(path, **match):
    """Upper end of the 95% interval for one log-loss comparison, or None if that file lacks it."""
    if not (RESULTS / path).exists():
        return None
    rows = [row for row in _rows(path) if row.get("metric") == "log_loss"
            and all(str(row.get(key)) == str(value) for key, value in match.items())]
    return float(rows[0]["ci95_high"]) if len(rows) == 1 else None


def _hours(flags):
    names = [f"{h:02}:00" for h, flag in zip(HOURS, flags) if flag]
    return names


def validation_text(candidate, base):
    better = _hours([c < b - 0.001 for c, b in zip(candidate, base)])
    if not better:
        return "No better"
    return "Better at all three" if len(better) == 3 else "Better at " + ", ".join(better)


def alternatives():
    """Rows of Table 4: (idea, test change per cutoff, upper 95% bound per cutoff, 2024 validation result)."""
    def column(path, column_name, name, **extra):
        rows = _rows(path)
        return [float(_pick(rows, hour=h, system=name, **extra)[column_name]) for h in HOURS]

    def change(new, old):
        return [a - b for a, b in zip(new, old)]

    def paired(path, column_name, **match):
        rows = [row for row in _rows(path) if row.get("metric") == "log_loss"]
        return [float(_pick(rows, hour=h, **match)[column_name]) for h in HOURS]

    saved = column("guidance/test_scores.csv", "log_loss", SELECTED)
    selection = {int(r["hour"]): r for r in _rows("guidance/selection.csv")}
    guidance_val = lambda name: [float(selection[h][f"val_{name}"]) for h in HOURS]
    hist = lambda name, col="log_loss": column("history/test_scores.csv", col, name)
    cm = lambda name, col="test_log_loss": column("class_models/scores.csv", col, name)
    hist_base, cm_base = "Censored Gaussian, 2021-2023", "Censored Gaussian, HRRR + LAMP/NBM (saved)"
    cm_refit = "Censored Gaussian (CURRENT, refit)"
    fresh_rows = {"cohort": "all 359, 10 UTC fallback"}
    fresh = {"cohort": "all 359, 10 UTC fallback", "baseline": "CURRENT (10 UTC HRRR)", "candidate": "(b) fresh HRRR added"}
    pooled = {"baseline": "CURRENT (Censored Gaussian, HRRR + LAMP/NBM)", "candidate": "Pooled"}
    recency = {"candidate": "CURRENT inputs 2021-2023, half-life 0.5 y"}
    obs = lambda name, col="log_loss": column("observed_weather/test_scores.csv", col, name)
    mos = lambda name, col="log_loss": column("mos_history/test_scores.csv", col, name)
    drift = lambda name, col="log_loss": column("nonstationarity/test_scores.csv", col, name)
    pool = lambda name, col="log_loss": column("pooled_airports/test_scores.csv", col, name)
    fr = lambda name, col="log_loss": column("hrrr_fresh/test_scores.csv", col, name, **fresh_rows)
    ci = lambda path, **match: [_ci_high(path, hour=h, **match) for h in HOURS]
    quantile = "Quantile boosting, HRRR + LAMP/NBM"
    return [
        ("Sinh-arcsinh distribution [12]", change(hist("Censored SHASH, 2021-2023"), hist(hist_base)),
         ci("history/paired.csv", baseline=hist_base, candidate="Censored SHASH, 2021-2023"),
         validation_text(hist("Censored SHASH, 2021-2023", "validation_log_loss"), hist(hist_base, "validation_log_loss"))),
        ("Quantile boosting, same inputs", change(column("guidance/test_scores.csv", "log_loss", quantile), saved),
         [None] * 3, validation_text(guidance_val(quantile), guidance_val(SELECTED))),
        ("Bagging, 50 bootstrap fits", change(cm("Bagged censored Gaussian (B=50, weekly blocks)"), cm(cm_base)),
         ci("class_models/paired_vs_current.csv", split="test", baseline=cm_base,
            candidate="Bagged censored Gaussian (B=50, weekly blocks)"),
         validation_text(cm("Bagged censored Gaussian (B=50, weekly blocks)", "validation_log_loss"),
                         cm(cm_refit, "validation_log_loss"))),
        ("Neural network, one hidden layer", change(cm("MLP (one tanh layer, 5 seeds)"), cm(cm_base)),
         ci("class_models/paired_vs_current.csv", split="test", baseline=cm_base, candidate="MLP (one tanh layer, 5 seeds)"),
         validation_text(cm("MLP (one tanh layer, 5 seeds)", "validation_log_loss"), cm(cm_refit, "validation_log_loss"))),
        ("Stacking with classifiers, network", change(cm("CURRENT + classifiers + MLP, equal weights"), cm(cm_base)),
         ci("class_models/paired_vs_current.csv", split="test", baseline=cm_base,
            candidate="CURRENT + classifiers + MLP, equal weights"),
         validation_text(cm("CURRENT + classifiers + MLP, equal weights", "validation_log_loss"),
                         cm(cm_refit, "validation_log_loss"))),
        ("Observed weather, six more stations", change(obs("(b) + LGA own weather + nearby"), obs("CURRENT")),
         ci("observed_weather/paired_vs_current.csv", split="test", baseline="CURRENT",
            candidate="(b) + LGA own weather + nearby"),
         validation_text(obs("(b) + LGA own weather + nearby", "validation_log_loss"), obs("CURRENT", "validation_log_loss"))),
        ("Newest HRRR run at each cutoff", paired("hrrr_fresh/paired_vs_current.csv", "delta", **fresh),
         paired("hrrr_fresh/paired_vs_current.csv", "ci95_high", **fresh),
         validation_text(fr("(b) fresh HRRR added", "validation_log_loss"), fr("CURRENT (10 UTC HRRR)", "validation_log_loss"))),
        ("JFK, EWR, TEB days pooled (4x rows)", paired("pooled_airports/paired.csv", "delta", **pooled),
         paired("pooled_airports/paired.csv", "ci95_high", **pooled),
         validation_text(pool("Pooled", "validation_log_loss"),
                         pool("CURRENT (Censored Gaussian, HRRR + LAMP/NBM)", "validation_log_loss"))),
        ("2010–2023 training, forecasts imputed", change(hist("Censored Gaussian, 2010-2023"), hist(hist_base)),
         ci("history/paired.csv", baseline=hist_base, candidate="Censored Gaussian, 2010-2023"),
         validation_text(hist("Censored Gaussian, 2010-2023", "validation_log_loss"), hist(hist_base, "validation_log_loss"))),
        ("GFS and NAM MOS model, 2010–2023", change(mos("Stations + trends + GFS + NAM MOS, 2010-2023"), saved),
         ci("mos_history/paired_vs_current.csv", baseline="CURRENT (saved)",
            candidate="Stations + trends + GFS + NAM MOS, 2010-2023"),
         validation_text(mos("Stations + trends + GFS + NAM MOS, 2010-2023", "validation_log_loss"),
                         mos("CURRENT (saved)", "validation_log_loss"))),
        ("Recency weights, six-month half-life", paired("nonstationarity/paired_vs_current.csv", "delta_vs_current", **recency),
         paired("nonstationarity/paired_vs_current.csv", "ci95_high", **recency),
         validation_text(drift("CURRENT inputs 2021-2023, half-life 0.5 y", "validation_log_loss"),
                         drift("CURRENT (saved)", "validation_log_loss"))),
    ]


def interval_summary(deltas, lows, highs):
    """Signed changes and which cutoffs fall outside their 95% interval."""
    text = ", ".join(f"{d:+.3f}" for d in deltas[:-1]) + f", and {deltas[-1]:+.3f}"
    better = _hours([h < 0 for h in highs])
    worse = _hours([l > 0 for l in lows])
    return text, better, worse


def better_on_test(rows):
    """Ideas whose test loss is lower than the selected model's beyond the 95% interval."""
    names = {9: "09:00", 12: "12:00", 15: "15:00"}
    found = []
    for idea, _, highs, _ in rows:
        hours = [names[h] for h, high in zip(HOURS, highs) if high is not None and high < 0]
        if hours:
            found.append((idea, hours))
    return found


def extension_section(body, heading, table, image, spacer, nonstationarity_text=""):
    from reportlab.platypus import CondPageBreak, KeepTogether

    n = numbers()
    if n["drift"] and not nonstationarity_text:
        nonstationarity_text = drift_text(n["drift"])
    figure = learning_curve_figure(RESULTS / "class_eval" / "learning_curves_paper.png")
    old = [float(row[2]) for row in n["table"]]
    new = [float(row[3]) for row in n["table"]]
    cut = [(a - b) / a for a, b in zip(old, new)]
    noon_accuracy = n["table"][1][5]
    mos = ""
    if n["mos_long"]:
        rows = [row for row in _rows("mos_history/paired_vs_current.csv") if row["metric"] == "log_loss"]
        picked = [_pick(rows, hour=h, candidate="Stations + trends + GFS + NAM MOS, 2010-2023",
                        baseline="Stations + trends + GFS + NAM MOS, 2021-2023") for h in HOURS]
        text, better, worse = interval_summary([float(r["delta"]) for r in picked], [float(r["ci95_low"]) for r in picked],
                                               [float(r["ci95_high"]) for r in picked])
        verdict_years = ("all within their 95% intervals" if not better and not worse else
                         f"a clear gain at {' and '.join(better)}" if better else f"clearly worse at {' and '.join(worse)}")
        mos = (" We also tested more training years with forecasts that exist in every year, GFS and NAM station "
               "MOS archived from 2008: training on 2010–2023 instead of 2021–2023 changed that model's test loss "
               f"by {text}, {verdict_years}. It stayed behind the censored model.")
    rows = [row for row in _rows("history/paired.csv") if row["metric"] == "log_loss"]
    picked = [_pick(rows, hour=h, baseline="Censored Gaussian, 2021-2023", candidate="Censored Gaussian, 2010-2023")
              for h in HOURS]
    _, _, imputed_worse = interval_summary([float(r["delta"]) for r in picked], [float(r["ci95_low"]) for r in picked],
                                           [float(r["ci95_high"]) for r in picked])
    imputed = (" Filling in LAMP, NBM, and HRRR for the years before they existed made the model worse at "
               + " and ".join(imputed_worse) + ", because one set of weights had to cover days with and without them."
               if imputed_worse else
               " Filling in LAMP, NBM, and HRRR for the years before they existed did not help.")
    paper_gap = (f" The Table 1 classifier average has a larger gap: {n['paper_train']:.2f} training against "
                 f"{n['paper_val']:.2f} validation at 09:00."
                 if n["paper_train"] is not None else "")
    picks = {hour: ("the censored model" if name.startswith("Censored") else "quantile boosting")
             for hour, name in n["selected"].items()}
    if len(set(picks.values())) == 1:
        selection_text = f"Validation chose {picks[9]} with HRRR and the station forecasts at all three cutoffs."
    else:
        censored = [f"{h:02}:00" for h in HOURS if picks[h] == "the censored model"]
        other = [f"{h:02}:00" for h in HOURS if picks[h] != "the censored model"]
        selection_text = ("Validation chose the censored model with HRRR and the station forecasts at "
                          + " and ".join(censored) + ", and quantile boosting on the same inputs, which predicts "
                          "quantiles of the same remaining rise, at " + " and ".join(other) + ".")
    sig = n["significant"]
    significant_text = ("all three cutoffs" if len(sig) == 3 else
                        " and ".join(sig) if sig else "none of the cutoffs")
    alternative_rows = alternatives()
    rows4 = [[idea, *(_change(value) for value in values), val] for idea, values, _, val in alternative_rows]
    winners = better_on_test(alternative_rows)
    if len(winners) == 1:
        idea, hours = winners[0]
        verdict = (f"One idea, {idea[0].lower() + idea[1:] if idea[1].islower() else idea}, had lower test loss "
                   f"beyond its 95% interval at {' and '.join(hours)}, but 2024 validation did not select it.")
    elif winners:
        listed = "; ".join(f"{idea} at {' and '.join(hours)}" for idea, hours in winners)
        verdict = (f"{len(winners)} ideas had lower test loss beyond their 95% intervals ({listed}), but 2024 "
                   "validation selected none of them.")
    else:
        some_val = sum(val != "No better" for *_, val in alternative_rows)
        verdict = (f"{some_val} of the {len(alternative_rows)} beat the censored model on 2024 at one or more "
                   "cutoffs, but none was clearly better on the test year." if some_val else
                   "None was clearly better than the censored model.")
    return [
        heading("7. Station guidance and a censored forecast"),
        body("The classifiers in Sections 3 to 6 treat the five bands as unrelated labels. They cannot use the "
             "order of the bands, and they have to learn from data that the final high is never below the high "
             "already observed. Section 8 lists a calibrated continuous forecast as a next step. Here we replace "
             "the band target with a continuous one and add station forecasts as inputs."),
        body("The new target is the remaining rise, R = Y − M, where Y is the final high and M is the highest "
             "reading in the hours completed so far. R is never negative. It is exactly zero on days when the high "
             f"has already occurred: {n['zero'][9]:.0%} of days at 09:00, {n['zero'][12]:.0%} at 12:00, and "
             f"{n['zero'][15]:.0%} at 15:00. We model R as a normal variable censored at zero, a Tobit regression "
             "[8]. Its mean is a weighted sum of the standardized inputs, and its standard deviation varies with "
             "the season."),
        body("The weights maximize the likelihood of the training-year rises, and a day whose high was already "
             "reached contributes the probability of no further rise. An L2 penalty, the only hyperparameter, is "
             "tuned on 2024. The fitted model gives a distribution for the final high, M + R, and a band's "
             "probability is the chance that M + R falls inside it. Bands below the current high therefore get "
             "zero probability, and the same fit can price any set of bands."),
        body("HRRR is raw output from one weather-model run each morning. We add two National Weather Service "
             "products that adjust model output using past observations at LGA: LAMP [10] and the National Blend "
             "of Models (NBM) text bulletin [11], both from the Iowa Environmental Mesonet archive [9]. At each "
             "cutoff we use the newest forecast released by then, assuming LAMP appears 55 minutes and NBM 66 "
             "minutes after the run starts. Seven features come from these forecasts: LAMP's highest temperature "
             "for the rest of the day, the latest observation minus LAMP's forecast for the current hour, LAMP's "
             "afternoon cloud cover and its two wind components, and NBM's forecast daytime high with its standard "
             "deviation."),
        body("Training, validation, and test years are those of Section 4. We compared six candidates on 2024 "
             "validation log loss: the Table 1 classifier families, the censored model, and quantile boosting, "
             "each with and without HRRR. " + selection_text + " Table 3 compares validation's choice with the "
             "selected HRRR system of Table 1 on the same dates. Log loss falls by "
             f"{cut[0]:.0%} at 09:00, {cut[1]:.0%} at 12:00, and {cut[2]:.0%} at 15:00, and the 95% intervals "
             f"exclude zero at {significant_text}. Accuracy at noon rises from {noon_accuracy}."),
        KeepTogether([
            _aligned_table(["Time", "Model", "Table 1 loss", "New loss", "Difference [95% interval]", "Accuracy"],
                           n["table"], [40, 92, 64, 58, 150, 112], numeric={2, 3, 4, 5}),
            spacer(1, 8),
            body(f"<b>Table 3.</b> Test log loss on the {n['n']} dates of the matched HRRR rows in Table 1. "
                 "Intervals pair the two systems by date and resample seven-day blocks 5,000 times."),
        ]),
        body(f"Figure 2 plots the censored model's learning curve. Its 2024 validation loss at 09:00 is "
             f"{n['lc_small']:.3f} with the most recent 400 training days and {n['lc_full']:.3f} with all "
             f"{n['lc_full_n']:,}, and its training loss sits at about the same level. That is the pattern of a "
             f"model limited by bias rather than variance: adding days from the same years would barely change "
             f"it. Quantile boosting on the same inputs keeps a gap between training and validation loss.{paper_gap}"),
        image(str(figure), width=500, height=159.6),
        body("<b>Figure 2.</b> Training and 2024 validation log loss with the station forecasts as inputs, against "
             "the number of most recent training days. Solid lines are validation and dashed lines are training."),
        body("Table 4 lists the other ideas we tried, each tuned on 2024 validation and scored on the test dates "
             "of Table 3, and compared with the censored model. " + verdict + " A model limited by bias gains little "
             "from extra flexibility, and the observed weather and nearby stations add little that the station "
             "forecasts do not already contain." + mos + imputed),
        KeepTogether([
            _aligned_table(["Idea", "09:00", "12:00", "15:00", "On 2024 validation"], rows4, [200, 48, 48, 48, 156],
                           numeric={1, 2, 3}),
            spacer(1, 8),
            body("<b>Table 4.</b> Change in test log loss against the censored model (positive values are worse), and "
                 "the cutoffs where the idea beat the censored model on 2024 validation by at least 0.001."),
        ]),
        body(nonstationarity_text) if nonstationarity_text else spacer(1, 0),
        CondPageBreak(120),
    ]


def extension_limitations(body):
    """One paragraph after the original Section 8 text, which is kept unchanged."""
    return [body("Section 7 completes two items on this list, the continuous forecast and the cloud-cover test. "
                 "Archived guidance has no receipt times, so its release delays are assumptions. Refitting as new "
                 "labels arrive, training on the official climate-report high, and checking inputs against an "
                 "arrival-time feed remain open.")]


def extension_references(body):
    return [
        body("[8] Tobin, J. (1958). Estimation of relationships for limited dependent variables. "
             "<i>Econometrica</i>, 26(1), 24–36."),
        body("[9] Iowa Environmental Mesonet. <link href='https://mesonet.agron.iastate.edu/mos/'>MOS archive "
             "(LAMP, NBM text, GFS and NAM MOS).</link>"),
        body("[10] Ghirardelli, J. E., and Glahn, B. (2010). The Meteorological Development Laboratory's aviation "
             "weather prediction system. <i>Weather and Forecasting</i>, 25(4), 1027–1051."),
        body("[11] NOAA Meteorological Development Laboratory. "
             "<link href='https://vlab.noaa.gov/web/mdl/nbm'>National Blend of Models.</link>"),
        body("[12] Jones, M. C., and Pewsey, A. (2009). Sinh-arcsinh distributions. <i>Biometrika</i>, 96(4), "
             "761–780."),
    ]
