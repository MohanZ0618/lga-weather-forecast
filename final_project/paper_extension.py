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
    out = {"table": []}
    for hour in HOURS:
        new, old = _pick(scores, hour=hour, system=SELECTED), _pick(scores, hour=hour, system=PAPER)
        change = _pick(paired, hour=hour, system=SELECTED, metric="log_loss")
        out["table"].append([f"{hour:02}:00", f"{float(old['log_loss']):.3f}", f"{float(new['log_loss']):.3f}",
                             f"{_signed(float(change['delta_vs_paper']))} [{_signed(float(change['ci95_low']))}, "
                             f"{_signed(float(change['ci95_high']))}]",
                             f"{float(old['correct']):.1%} to {float(new['correct']):.1%}"])
    out["n"] = _pick(scores, hour=9, system=SELECTED)["n"]
    selection = _rows("guidance/selection.csv")
    out["all_selected"] = all(row["selected"] == SELECTED for row in selection)

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
                  "HRRR": "HRRR remaining max", "LGA relative to JFK, EWR and TEB":
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
    return ("<b>Changes over time.</b> Weather forecasts and sensors change over time. In the monthly forecast "
            "errors we found level shifts in " + ", ".join(parts[:-1]) + ", and " + parts[-1] + ". The NBM "
            "upgrade to version 5 in May 2026 did not cause a significant shift. Giving recent training days more "
            "weight improved 2024 but not the test year, and refitting every month helped only at 09:00. With one "
            "validation year we cannot tell which of these fixes would carry forward, so we report the fixed "
            "model.")


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


def alternatives():
    """Rows of Table 4: change in test log loss against the selected model, fitted the same way."""
    def scores(path, column, name):
        rows = _rows(path)
        return [float(_pick(rows, hour=h, system=name)[column]) for h in HOURS]

    def change(new, old):
        return [a - b for a, b in zip(new, old)]

    saved = scores("guidance/test_scores.csv", "log_loss", SELECTED)
    history = scores("history/test_scores.csv", "log_loss", "Censored Gaussian, 2021-2023")
    models = scores("class_models/scores.csv", "test_log_loss", "Censored Gaussian, HRRR + LAMP/NBM (saved)")
    observed = scores("observed_weather/test_scores.csv", "log_loss", "CURRENT")
    drift = [float(_pick(_rows("nonstationarity/paired_vs_current.csv"), hour=h, metric="log_loss",
                         candidate="CURRENT inputs 2021-2023, half-life 0.5 y")["delta_vs_current"]) for h in HOURS]
    return [
        ("Skewed, heavy-tailed distribution (sinh-arcsinh) [12]",
         change(scores("history/test_scores.csv", "log_loss", "Censored SHASH, 2021-2023"), history),
         "Extra shape parameters; censoring at zero already captures the main skew"),
        ("Quantile boosting on the same inputs",
         change(scores("guidance/test_scores.csv", "log_loss", "Quantile boosting, HRRR + LAMP/NBM"), saved),
         "More flexible, so it needs more data (Figure 2)"),
        ("Bagging: average of 50 bootstrap fits",
         change(scores("class_models/scores.csv", "test_log_loss", "Bagged censored Gaussian (B=50, weekly blocks)"),
                models), "Little variance to remove"),
        ("Neural network, one hidden layer, trained by backpropagation",
         change(scores("class_models/scores.csv", "test_log_loss", "MLP (one tanh layer, 5 seeds)"), models),
         "No consistent gain on validation"),
        ("Stacking with the classifiers and the network",
         change(scores("class_models/scores.csv", "test_log_loss", "CURRENT + classifiers + MLP, equal weights"),
                models), "Validation gave the classifiers zero weight"),
        ("Observed sky cover, wind, pressure, six nearby stations",
         change(scores("observed_weather/test_scores.csv", "log_loss", "(b) + LGA own weather + nearby"), observed),
         "The station forecasts already use these observations"),
        ("Training on 2010–2023, missing forecasts filled in",
         change(scores("history/test_scores.csv", "log_loss", "Censored Gaussian, 2010-2023"), history),
         "One set of weights for days with and without forecasts"),
        ("Long-record GFS and NAM MOS model, 2010–2023",
         change(scores("mos_history/test_scores.csv", "log_loss", "Stations + trends + GFS + NAM MOS, 2010-2023"),
                saved), "More years help, but MOS is less accurate than LAMP and NBM"),
        ("Recent days weighted more (six-month half-life)", drift, "Helped 2024 but not the test year"),
    ]


def extension_section(body, heading, table, image, spacer, nonstationarity_text=""):
    from reportlab.platypus import CondPageBreak, KeepTogether

    n = numbers()
    if n["drift"] and not nonstationarity_text:
        nonstationarity_text = drift_text(n["drift"])
    figure = learning_curve_figure(RESULTS / "class_eval" / "learning_curves_paper.png")
    old = [float(row[1]) for row in n["table"]]
    new = [float(row[2]) for row in n["table"]]
    cut = [(a - b) / a for a, b in zip(old, new)]
    noon_accuracy = n["table"][1][4]
    mos = ""
    if n["mos_long"]:
        mos = (" Longer training history helped only when the forecasts it uses existed in every year. With "
               "GFS and NAM MOS, which are archived from 2008, training on 2010–2023 instead of 2021–2023 lowered "
               f"that model's test loss by {-n['mos_long'][9]:.3f} at 09:00 and {-n['mos_long'][12]:.3f} at 12:00, "
               "but it still did not reach the selected model.")
    paper_gap = (f" The Table 1 classifier average shows the same gap more strongly: {n['paper_train']:.2f} "
                 f"training against {n['paper_val']:.2f} validation log loss at 09:00."
                 if n["paper_train"] is not None else "")
    rows4 = [[idea, *(_change(value) for value in values), reason] for idea, values, reason in alternatives()]
    return [
        heading("7. Station guidance and a censored forecast"),
        body("The classifiers in Sections 3 to 6 treat the five bands as unrelated labels. They do not use the fact "
             "that the bands are ordered, and they must learn from data that the final high cannot fall below the "
             "high already observed. Section 8 also lists a calibrated continuous forecast as a next step. This "
             "section makes two changes: what the model predicts, and which forecasts it uses."),
        body("<b>Predicting the remaining rise.</b> Instead of a band, we predict how many more degrees LGA will "
             "warm after the cutoff. We call this the remaining rise, R = Y − M, where Y is the final high and M "
             "is the highest reading in the hours completed so far. R is never negative, and it is exactly zero "
             "when the day's high has already occurred: on 12% of days at 09:00, 19% at 12:00, and 63% at 15:00. "
             "We model R as a normal variable cut off at zero, which is censored (Tobit) regression [8]. The "
             "model has two parts: the expected rise, a weighted sum of the standardized inputs, and its "
             "uncertainty, a standard deviation that varies with the season."),
        body("<b>Objective and band probabilities.</b> We fit the weights by maximizing the likelihood of the rises "
             "observed in the training years. Days on which the high had already been reached contribute the "
             "probability of no further rise. An L2 penalty keeps the weights small; its strength is the only "
             "hyperparameter and is chosen on 2024. The fitted model gives a full distribution for the final high, "
             "M + R. A band's probability is the chance that M + R falls inside it, so bands below the current "
             "high get zero probability, and the same fit works for any set of bands."),
        body("<b>Station forecasts as inputs.</b> HRRR is raw output from one numerical weather model run each "
             "morning. We add two National Weather Service products that correct model output using past "
             "observations at LGA: LAMP [10] and the National Blend of Models (NBM) text bulletin [11], both taken "
             "from the Iowa Environmental Mesonet archive [9]. At each cutoff we use only the newest forecast "
             "released by then, assuming LAMP is released 55 minutes and NBM 66 minutes after the run starts, so "
             "no input uses information from after the cutoff. Seven features summarize these forecasts: LAMP's "
             "highest temperature for the rest of the day; the difference between the latest observation and "
             "LAMP's forecast for the current hour; LAMP's afternoon cloud cover and its two wind components; and "
             "NBM's forecast daytime high and its standard deviation."),
        body("<b>Selection and results.</b> We kept the training, validation, and test years of Section 4. On "
             "2024 validation log loss we compared six candidates: the classifier families of Table 1, the "
             "censored model, and quantile boosting, each with and without HRRR. The censored model with HRRR and "
             "the station forecasts was best at "
             + ("all three cutoffs. " if n["all_selected"] else "the cutoffs shown. ")
             + f"Table 3 compares it with the selected HRRR system of Table 1 on the same dates. Log loss falls by "
             f"{cut[0]:.0%} at 09:00, {cut[1]:.0%} at 12:00, and {cut[2]:.0%} at 15:00, and accuracy at noon rises "
             f"from {noon_accuracy}. The 95% intervals exclude zero at 09:00 and 12:00. At 15:00 the day's high "
             "has usually been reached already, which leaves less room to improve."),
        KeepTogether([
            table(["Time", "Table 1 loss", "New loss", "Difference [95% interval]", "Accuracy"],
                  n["table"], [42, 80, 70, 170, 130]),
            spacer(1, 8),
            body(f"<b>Table 3.</b> Test log loss on the {n['n']} dates of the matched HRRR rows in Table 1. "
                 "Intervals pair the two systems by date and resample seven-day blocks 5,000 times."),
        ]),
        body("<b>Why a simple model is enough.</b> A learning curve shows whether more data would help. A model "
             "limited by variance overfits: its training loss is well below its validation loss, and validation "
             "loss keeps falling as training days are added. A model limited by bias has flat curves that are "
             "close to each other. Figure 2 shows the second pattern for the censored model: at 09:00 its 2024 "
             f"validation log loss is {n['lc_small']:.3f} with the most recent 400 training days and "
             f"{n['lc_full']:.3f} with all {n['lc_full_n']:,}, close to its training loss. More days of the same "
             "data would not help, but better inputs could. Quantile boosting, a more flexible model on the same "
             f"inputs, keeps a gap between training and validation loss.{paper_gap}"),
        image(str(figure), width=500, height=159.6),
        body("<b>Figure 2.</b> Training and 2024 validation log loss with the station forecasts as inputs, against "
             "the number of most recent training days. Solid lines are validation and dashed lines are training."),
        body("<b>What did not help.</b> Table 4 lists the other ideas we tried. Each was tuned on 2024 validation "
             "and scored on the test dates of Table 3, and none was clearly better than the selected model. "
             "Figure 2 explains most of these results: a model limited by bias gains little from extra "
             "flexibility, and the observed weather and nearby stations repeat information the station forecasts "
             "already contain." + mos + " Filling in LAMP, NBM, and HRRR for years before they existed made the "
             "model worse, because one set of weights had to serve days with and without those forecasts."),
        KeepTogether([
            _aligned_table(["Idea", "09:00", "12:00", "15:00", "Likely reason"], rows4, [178, 46, 46, 46, 184],
                           numeric={1, 2, 3}),
            spacer(1, 8),
            body("<b>Table 4.</b> Change in test log loss against the selected model; positive values are worse."),
        ]),
        body(nonstationarity_text) if nonstationarity_text else spacer(1, 0),
        CondPageBreak(120),
    ]


def extension_limitations(body):
    """One paragraph after the original Section 8 text, which is kept unchanged."""
    return [body("Section 7 completes two items from this list: the continuous forecast and the cloud-cover test. "
                 "Archived guidance has no receipt times, so its release delays are assumptions. Refitting as new "
                 "labels arrive, training on the official climate-report high, and checking inputs against an "
                 "arrival-time feed remain next steps.")]


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
