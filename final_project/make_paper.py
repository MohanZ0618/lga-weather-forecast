"""Build the paper PDF from the reproduced result CSVs."""

import csv
import gzip
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
DELIVERY = ROOT / "deliverables"
TEAL = colors.HexColor("#126078")
INK = colors.HexColor("#162939")
GRAY = colors.HexColor("#465766")
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=13, textColor=INK, spaceAfter=8)
HEADING = ParagraphStyle("Heading", parent=BODY, fontName="Helvetica-Bold", textColor=TEAL, spaceBefore=7)
TITLE = ParagraphStyle("Title", parent=BODY, fontName="Helvetica-Bold", fontSize=23, leading=27, spaceAfter=12)
CELL = ParagraphStyle("Cell", parent=BODY, leading=12)
RIGHT = ParagraphStyle("Number", parent=CELL, alignment=TA_RIGHT)
TEAM = ["Mohan Zhang", "Mingyeung Wu", "Maxwell Liss"]
HRRR = "Four stations + trends + HRRR"
TRENDS = "Four stations + trends"
BANDS = ["below 40", "40 to <55", "55 to <70", "70 to <85", "85 or above"]
SHORT = {"Two-model probability average": "Two-model average", "Logistic regression": "Logistic regression"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def body(text):
    return Paragraph(text, BODY)


def heading(text):
    return Paragraph(text, HEADING)


def table(headers, rows, widths):
    content = [[Paragraph(escape(str(cell)), HEADING) for cell in headers]]
    content += [[Paragraph(escape(str(cell)), CELL if i < 3 else RIGHT)
                 for i, cell in enumerate(row)] for row in rows]
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


def page_footer(canvas, document):
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(51, 33, "CS4260 · Fall 2026")
    canvas.drawRightString(561, 33, str(document.page))


def metric(row, name):
    return float(row[name])


def main():
    DELIVERY.mkdir(exist_ok=True)
    rows = read_csv(RESULTS / "main_table.csv")
    changes = read_csv(RESULTS / "paired_comparisons.csv")
    errors = read_csv(RESULTS / "errors.csv")
    boundaries = read_csv(RESULTS / "by_near_boundary.csv")
    coverage = read_csv(RESULTS / "by_month.csv")
    overlap = read_csv(DATA / "archive_overlap_2024.csv")
    stamp = read_csv(DATA / "timestamp_overlap_2024.csv")[0]
    valid = read_csv(RESULTS / "all_metrics.csv")
    blend_count = sum(row["model"] == "Two-model probability average"
                      and row["scope"] not in ("Monthly climatology", "Remaining warming") for row in rows)
    matched = [row for row in rows if row["scope"] == HRRR]
    full_n = next(row["n"] for row in rows if row["scope"] == "LGA")
    matched_n = matched[0]["n"]
    data_counts = {}
    with gzip.open(DATA / "samples_09.csv.gz", "rt", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            date = row["date"]
            part = "train" if date < "2024" else "validation" if date < "2025" else (
                "overlap" if date < "2025-09" else "test")
            data_counts[part] = data_counts.get(part, 0) + 1
    point_counts = {}
    with gzip.open(DATA / "hrrr_cycles.csv.gz", "rt", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            status = row["status"]
            point_counts[status] = point_counts.get(status, 0) + 1
    archive_agree = sum(row["target_category"] == row["legacy_target_category"] for row in overlap)
    hrrr_improves = sum(row["candidate"] == HRRR and metric(row, "delta_log_loss") < 0 for row in changes)
    stations_improve = sum(row["candidate"] == "Four stations" and metric(row, "delta_log_loss") < 0 for row in changes)
    worst = errors[0]
    late = [row for row in valid if row["scope"] == HRRR and row["hour"] == "15" and row["split"] == "test"]
    xgb_late = next(row for row in late if row["model"] == "XGBoost")
    test_days = [int(row["n"]) for row in coverage if row["scope"] == "LGA" and row["hour"] == "9"]
    same_band_errors = sum(abs(int(row["predicted"]) - int(row["target_category"])) == 1 for row in errors)
    contributions_path = ROOT / "contributions.json"
    contributions = json.loads(contributions_path.read_text(encoding="utf-8")) if contributions_path.exists() else {}

    content = [
        Paragraph("Predicting LaGuardia’s daily maximum temperature band", TITLE),
        body("Mohan Zhang, Mingyeung Wu, and Maxwell Liss<br/>CS4260 Introduction to Artificial Intelligence · Fall 2026"),
        heading("Abstract"),
        body(f"We estimate five probabilities for LaGuardia Airport’s same-day maximum temperature at "
             f"09:00, 12:00, and 15:00 New York time. We extend our Progress 1 study with a full-year "
             f"holdout, a historical remaining-warming baseline, and archived High-Resolution Rapid Refresh "
             f"(HRRR) forecasts. The observation experiment covers {full_n} held-out days and the matched "
             f"forecast experiment covers {matched_n}. Adding HRRR lowered log loss at {hrrr_improves} of "
             f"the 3 cutoffs. The validation-selected HRRR systems attained test log losses of "
             f"{', '.join(f'{metric(row, 'log_loss'):.3f}' for row in matched)} in time order."),
        heading("1. Problem and supervised formulation"),
        body("A morning temperature reading leaves several hours for the day to warm further. By afternoon, "
             "the observed maximum can already settle the relevant temperature band. We ask how much additional "
             "information comes from nearby airports and a numerical temperature forecast as the day progresses. "
             "The project fits Topic 4 because each historical day has a target label and the task is supervised "
             "multiclass classification with probabilistic outputs."),
        body("At cutoff h, the input x(d,h) contains weather observations timestamped by h, seasonal features, "
             "and optionally an archived forecast available by h. The target y(d) is one of five ordered "
             "bands of the day's highest retained hourly LGA observation. Fahrenheit thresholds are 40, 55, "
             "70, and 85. A temperature exactly on a threshold belongs to the warmer band. The classifier "
             "returns p(y | x), a vector of nonnegative values summing to one. We compare the expected quality "
             "of these distributions by held-out mean negative log probability of the observed class [4]."),
        heading("2. Data and label construction"),
        body("We use the NOAA Local Climatological Data version 2 station files [1] for LGA, JFK, EWR, and "
             "TEB from 2018 through August 2026. These records derive from GHCNh, the replacement for the "
             "ISD/Global Hourly archive used in Progress 1. We keep METAR and SPECI observations, convert "
             "Celsius to Fahrenheit, and use the last timestamped row in each UTC hour, then require "
             "20 hours with valid temperature and coverage of all four six-hour New York time blocks. "
             "The label represents the hourly observation series. Official daily and market settlement highs "
             "can use different observations and reporting rules."),
        body(f"The source timestamps are fixed Eastern standard time. We convert them before computing New "
             f"York dates and cutoffs. At {int(stamp['pairs']):,} shared valid LGA timestamps in 2024, the "
             f"mean absolute temperature difference was {float(stamp['mean_abs_difference_f']):.3f}°F. "
             f"The two sources agreed on {archive_agree} of {len(overlap)} overlapping daily bands. "
             f"This overlap check caught a five-hour error during development."),
        heading("Contribution statement"),
        *[body(f"<b>{name}.</b> {escape(contributions.get(name, '[Team member to confirm actual contributions.]'))}")
          for name in TEAM],
        PageBreak(),
        heading("3. Features, baselines, and models"),
        body("At each station we use the latest temperature, relative humidity, and wind speed within the "
             "past hour, as well as the observed high so far, the previous day's high, and the change relative "
             "to roughly three hours earlier. The high so far uses only completed hourly bins, so it is a "
             "lower bound on the daily target even when a report at the cutoff is superseded later in that "
             "same hour. Two additional trend features divide temperature change by the actual elapsed time "
             "over roughly one and three hours. We use sine and cosine of the day of year to represent seasonality."),
        body("For HRRR, we extract the 2-meter temperature at the grid point nearest LGA from one 10:00 UTC "
             "forecast cycle per date [2,3]. The features are the maximum modeled temperature from the cutoff "
             "to before local midnight, and that maximum minus the modeled temperature at the cutoff. We "
             f"require the tile's S3 Last-Modified time to fall between initialization and 09:00 New York time. "
             f"Of the attempted cycles, {point_counts['ok']:,} met the checks, {point_counts['outside_issue_window']} "
             f"fell outside the issue window, and {point_counts['missing']} tiles were absent."),
        body("<b>Baselines.</b> Monthly climatology estimates class frequencies using only training days, "
             "with one pseudocount per band. The remaining-warming baseline adds a distribution of historical "
             "remaining temperature increases to the current day's observed high and bins the resulting highs. "
             "A Gaussian kernel weights historical days by distance in calendar month (1.5 months) and "
             "observed high so far (5°F). This baseline tests whether machine learning adds skill once some "
             "of the day's temperatures are already available."),
        table(["Family", "What drives its probabilities", "Validation settings"],
              [["Logistic regression", "Linear logits over scaled features", "C = 0.1, 1, 10"],
               ["KNN", "Distance-weighted votes in scaled feature space", "k = 15, 40, 80"],
               ["Gaussian naive Bayes", "Class-conditional Gaussian densities", "Smoothing = 1e-9, 0.01, 0.1"],
               ["Random forest", "Average of 200 bootstrap trees", "Depth 5, 10, unlimited"],
               ["XGBoost", "200 sequential trees with soft probabilities", "Depth 2, 3, 5"]],
              [110, 224, 174]),
        Spacer(1, 10),
        body("We fit preprocessing only on training rows. The scikit-learn models [5] use median imputation. "
             "Logistic regression, KNN, and naive Bayes use standardization. XGBoost [6] handles missing "
             "predictors internally and uses learning rate 0.04 with row and column subsampling of 0.9. "
             "The random forest uses a minimum leaf size of four. Tree methods use seed 4260."),
        body("For every model and the remaining-warming baseline, we mix 1% monthly climatology into the "
             "distribution before removing bands already surpassed by the observed high and renormalizing. "
             "This stabilizes probability estimates from models such as KNN that can assign exact zeros. "
             "The equal-weight blend averages the distributions of the two best distinct model families on "
             "validation. Forecast postprocessing and distribution mixtures have precedent in weather "
             "forecasting [7]. Here, both blend weights are fixed at one half."),
        heading("4. Experimental protocol"),
        body(f"Ground observations train on 2018–2023 ({data_counts['train']:,} dates). The matched HRRR "
             f"experiment uses available 2021–2023 dates for both sides. We tune and select on 2024 "
             f"({data_counts['validation']} observation dates). January–August 2025 is an overlap period "
             f"reported separately because Progress 1 examined most of it. The final test is September "
             f"2025–August 2026. Inside each cohort, both sides of an ablation use the same date set. "
             f"Each model is fitted separately for each clock time."),
        PageBreak(),
        heading("5. Results on the final holdout"),
        body("Table 1 reports final-period log loss, Brier score, accuracy, and macro F1. Log loss "
             "penalizes assigning little probability to the true band. The multiclass Brier score is "
             "the mean sum of squared error against the one-hot target. Accuracy and macro F1 use the "
             "most probable band."),
    ]
    labels = {"Monthly climatology": "Monthly climatology", "Remaining warming": "Remaining warming",
              "LGA": "LGA only", "Four stations": "Four stations", TRENDS: "With trends", HRRR: "With trends + HRRR"}
    results_rows = []
    for row in rows:
        if row["cohort"] == "Observations" or row["scope"] in (TRENDS, HRRR):
            scope = labels[row["scope"]]
            if row["cohort"] == "Matched HRRR" and row["scope"] == TRENDS:
                scope = "With trends*"
            model = SHORT.get(row["model"], row["model"])
            if row["scope"] in ("Monthly climatology", "Remaining warming"):
                model = "Baseline"
            results_rows.append([f"{int(row['hour']):02}:00", scope, model, row["n"],
                                 f"{metric(row, 'log_loss'):.3f}", f"{metric(row, 'brier'):.3f}",
                                 f"{metric(row, 'accuracy'):.1%}", f"{metric(row, 'macro_f1'):.3f}"])
    content += [
        table(["Time", "Features / baseline", "Validation selection", "Days", "Loss", "Brier", "Acc.", "F1"],
              results_rows, [42, 110, 133, 39, 44, 44, 51, 45]),
        Spacer(1, 8),
        body("<b>Table 1.</b> Each time block contains the observation cohort and the pair trained on "
             "the matched HRRR cohort. *The starred observation model uses the same dates as the HRRR row."),
    ]
    paired = []
    for cutoff in (9, 12, 15):
        values = [metric(change, "delta_log_loss") for change in changes if int(change["hour"]) == cutoff]
        paired.append([f"{cutoff:02}:00", *[f"{value:+.4f}" for value in values]])
    content += [
        table(["Time", "Add nearby stations", "Add trends", "Trends vs. warming", "Add HRRR"],
              paired, [60, 113, 102, 126, 107]),
        Spacer(1, 8),
        body(f"<b>Table 2.</b> Differences in mean test log loss on paired dates. Negative values favor the "
             f"added inputs or the trends model in the third comparison. The nearby stations improved "
             f"{stations_improve} of the 3 cutoffs. The blend was selected on validation in {blend_count} "
             f"of 15 model comparisons across input sets and cohorts."),
        PageBreak(),
        heading("6. Where the forecasts fail"),
        body(f"HRRR's largest gain came at 09:00, when more of the daytime heating was still ahead. "
             f"At 15:00 the reduction was only 0.0002. The validation-selected 15:00 HRRR blend "
             f"scored {metric(matched[-1], 'log_loss'):.3f}, while the corresponding XGBoost candidate "
             f"scored {metric(xgb_late, 'log_loss'):.3f} on the same test dates. The holdout also showed "
             f"that the added temperature trends raised loss in the morning and at noon. At 15:00 the "
             f"remaining-warming baseline reached 0.093 log loss on the matched dates."),
    ]
    for hour in (9, 12, 15):
        close, far = sorted((row for row in boundaries if row["scope"] == HRRR and int(row["hour"]) == hour),
                            key=lambda row: row["near_boundary"], reverse=True)
        content.append(body(
            f"<b>{hour:02}:00.</b> Within 2°F of a band boundary, the HRRR model classified "
            f"{metric(close, 'accuracy'):.1%} correctly on {close['n']} days. Farther away, accuracy "
            f"was {metric(far, 'accuracy'):.1%} on {far['n']} days."))
    content += [
        body(f"Across the three times, {same_band_errors} of {len(errors)} mistakes were one band away. "
             f"The most confident miss occurred on {worst['date'][:10]} at {int(worst['hour']):02}:00. "
             f"The model gave {metric(worst, 'confidence'):.1%} to the {escape(BANDS[int(worst['predicted'])])}°F "
             f"band. The observed high was {metric(worst, 'target_max'):.2f}°F. Confidence can stay high "
             "when a small error in the temperature trajectory changes the label."),
        Image(str(RESULTS / "reliability.png"), width=500, height=159.6),
        body("<b>Figure 1.</b> Top-class reliability for the selected HRRR systems on the final test. "
             "Each point groups predicted confidence in a tenth-wide interval. Larger points represent "
             "more dates. The diagonal marks agreement between average confidence and observed accuracy."),
        heading("7. Limitations and next steps"),
        body(f"The final test contains {full_n} observation dates across 12 months, with "
             f"{min(test_days)}–{max(test_days)} eligible days per month. The matched tests "
             "describe dates with verifiable forecast tiles. Archive availability can differ from operational "
             "delivery. Station timestamps give measurement time, and the archive provides limited information "
             "about transmission delays. Adjacent days and multiple cutoffs on one date are dependent. Results "
             "describe this airport and test year. With another month we would run rolling seasonal tests and "
             "compare a calibrated continuous forecast with the five-class models. We would also test hourly "
             "cloud cover and verify station inputs against an arrival-time feed."),
        heading("AI-use acknowledgment"),
        body("OpenAI Codex assisted with searching for data and published methods, implementing and debugging "
             "the pipeline, running checks, and drafting the paper and presentation. Evaluation results "
             "come from the executable experiment and are included with the submission."),
        PageBreak(),
        heading("References"),
        body("[1] NOAA National Centers for Environmental Information. Local Climatological Data Version 2. "
             "<link href='https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data'>Dataset and technical documentation.</link>"),
        body("[2] NOAA Global Systems Laboratory. "
             "<link href='https://rapidrefresh.noaa.gov/hrrr/'>High-Resolution Rapid Refresh.</link>"),
        body("[3] University of Utah MesoWest. "
             "<link href='https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/ex_python_plot_zarr.html'>HRRR Zarr point extraction example and archive.</link>"),
        body("[4] Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, "
             "and estimation. <i>JASA</i>, 102(477), 359–378. "
             "<link href='https://doi.org/10.1198/016214506000001437'>doi:10.1198/016214506000001437.</link>"),
        body("[5] Pedregosa, F., et al. (2011). Scikit-learn machine learning in Python. "
             "<i>JMLR</i>, 12, 2825–2830. <link href='https://jmlr.org/papers/v12/pedregosa11a.html'>Article.</link>"),
        body("[6] Chen, T., and Guestrin, C. (2016). XGBoost. A scalable tree boosting system. "
             "<i>KDD</i>, 785–794. <link href='https://doi.org/10.1145/2939672.2939785'>doi:10.1145/2939672.2939785.</link>"),
        body("[7] Raftery, A. E., Gneiting, T., and Berrocal, V. (2006). Combining spatial statistical "
             "and ensemble information in probabilistic weather forecasts. "
             "<link href='https://stat.uw.edu/research/tech-reports/combining-spatial-statistical-and-ensemble-information-probabilistic-weather-forecasts'>University of Washington technical report.</link>"),
    ]
    document = SimpleDocTemplate(str(DELIVERY / "CS4260_Final_Paper.pdf"), pagesize=letter,
                                 title="Predicting LaGuardia’s daily maximum temperature band",
                                 author=", ".join(TEAM), leftMargin=46, rightMargin=46,
                                 topMargin=40, bottomMargin=54)
    document.build(content, onFirstPage=page_footer, onLaterPages=page_footer)


if __name__ == "__main__":
    main()
