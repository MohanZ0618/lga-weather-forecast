"""Build the figures and Progress 1 report from the saved experiment results."""

import argparse
import calendar
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
TITLE = "Probabilistic Weather Forecasting Using Supervised Machine Learning"
MEMBERS = "Mohan Zhang, Mingyeung Wu, Maxwell Liss"
BANDS = ["<40°F", "40–<55°F", "55–<70°F", "70–<85°F", "≥85°F"]
SPLITS = {2023: "Training", 2024: "Validation", 2025: "Test"}
COUNTS_FIGURE = "temperature_category_counts.png"
LOSS_FIGURE = "test_log_loss_by_cutoff.png"
METRIC_HEADERS = ["Accuracy", "Macro F1", "Log loss", "Brier"]
COLORS = ["#687583", "#007B78", "#2C5D9D", "#AB5728"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def hour_label(hour):
    return f"{int(hour):02d}:00"


def metric_cells(row):
    return [f"{float(row['accuracy']):.1%}", f"{float(row['macro_f1']):.3f}",
            f"{float(row['log_loss']):.4f}", f"{float(row['brier']):.3f}"]


def figures(labels, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = Counter((int(row["date"][:4]), int(row["target_category"])) for row in labels)
    figure, axis = plt.subplots(figsize=(8.2, 2.75), layout="constrained")
    for offset, (year, split) in enumerate(SPLITS.items()):
        positions = [band + (offset - 1) * 0.24 for band in range(len(BANDS))]
        axis.bar(positions, [counts[year, band] for band in range(len(BANDS))], 0.22,
                 label=f"{year} {split.lower()}", color=COLORS[offset + 1])
    axis.set(xticks=range(len(BANDS)), xticklabels=BANDS, ylabel="Eligible dates", xlabel="LGA maximum category")
    axis.legend(frameon=False, ncols=3, loc="upper center")
    axis.set_ylim(0, max(counts.values()) * 1.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(OUTPUTS / COUNTS_FIGURE, dpi=180)
    plt.close(figure)

    test = [row for row in metrics if row["split"] == "test"]
    upper_limit = max(float(row["log_loss"]) for row in test) * 1.11
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True, layout="constrained")
    for axis, scope in zip(axes, ("LGA only", "Four stations")):
        for color, model in zip(COLORS, ("Climatology", "Logistic Regression", "Random Forest", "XGBoost")):
            rows = [row for row in test if row["scope"] == scope and row["model"] == model]
            rows.sort(key=lambda row: int(row["hour"]))
            axis.plot([hour_label(row["hour"]) for row in rows], [float(row["log_loss"]) for row in rows],
                      marker="o", markersize=4, color=color, label=model)
        axis.set(title=scope, xlabel="New York cutoff", ylim=(0, upper_limit))
        axis.grid(axis="y", color="#DFE3E6")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("2025 test log loss")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False)
    figure.savefig(OUTPUTS / LOSS_FIGURE, dpi=180)
    plt.close(figure)


def report_pages(labels, metrics, selected, quality, errors):
    days = {year: [row["date"] for row in labels if row["date"].startswith(str(year))] for year in SPLITS}
    coverage = [
        [str(year), split, f"{min(days[year])} to {max(days[year])}", str(len(days[year])),
         str(365 + calendar.isleap(year) - len(days[year]))]
        for year, split in SPLITS.items()
    ]
    lookup = {(int(r["hour"]), r["scope"], r["model"], r["split"]): r for r in metrics}
    test_rows = [lookup[row["hour"], row["scope"], row["model"], "test"] for row in selected]
    base = lookup[9, "LGA only", "Climatology", "test"]
    best_test = min(test_rows, key=lambda row: float(row["log_loss"]))
    error_rows = [
        [row["date"], hour_label(row["hour"]), f"{float(row['target_max']):.2f}°F",
         BANDS[int(row["target_category"])], BANDS[int(row["predicted_category"])],
         f"{float(row['confidence']):.1%}"]
        for row in errors[:5]
    ]
    adjacent = sum(abs(int(row["predicted_category"]) - int(row["target_category"])) == 1 for row in errors)
    noon = [row for row in metrics if row["scope"] == "LGA only" and row["hour"] == "12"]
    pages = [
        [
            ("title", TITLE),
            ("subtitle", "CS4260 Progress 1 Report"),
            ("text", MEMBERS),
            ("heading", "Same Day Prediction Task"),
            ("text", "We revised the proposal from forecasting tomorrow’s high to estimating today’s final maximum "
             "at LaGuardia Airport (LGA). At 09:00, 12:00, and 15:00 New York time, the model returns five "
             "probabilities that sum to 1. The categories are below 40°F, 40°F to below 55°F, 55°F to below "
             "70°F, 70°F to below 85°F, and 85°F or higher."),
            ("text", f"On the {len(days[2025])} eligible 2025 test dates, the validation-selected "
             f"{hour_label(best_test['hour'])} {best_test['model']} recorded "
             f"{float(best_test['log_loss']):.3f} log loss and {float(best_test['accuracy']):.1%} accuracy. "
             f"The climatology baseline recorded {float(base['log_loss']):.3f} log loss and "
             f"{float(base['accuracy']):.1%} accuracy."),
            ("heading", "Data and Eligible Dates"),
            ("text", "We downloaded 12 NOAA Global Hourly CSVs from the public S3 archive. They cover LGA "
             "(72503014732), JFK (74486094789), EWR (72502014734), and TEB (72502594741) for UTC source years "
             "2023 through 2025. The raw manifest records each source URL, byte count, and SHA-256 hash."),
            ("text", "LGA defines the target for every model. An eligible New York calendar day has at least "
             "20 valid hourly temperatures and at least one in each six-hour period starting at 00:00, "
             "06:00, 12:00, and 18:00. The target is the largest retained temperature on that day."),
            ("table", (["Local year", "Split", "First and last eligible dates", "Eligible", "Excluded"], coverage,
                       [0.85, 1.0, 3.33, 0.82, 0.9])),
            ("caption", "Excluded dates are measured against the full local calendar year and include dates outside the archive coverage."),
            ("heading", "Features and Chronological Split"),
            ("text", "Each station contributes its latest temperature, humidity, and wind from an observation "
             "at most one hour old, its maximum since midnight, a roughly three-hour temperature change, "
             "and its previous calendar day’s observed maximum. The model also uses month. The older reading "
             "for the temperature change comes from the hour ending three hours before cutoff."),
            ("text", "Each cutoff uses a localized wall-clock time, including on daylight saving dates. "
             "We trim readings to the cutoff before keeping the last one per UTC hour. Previous-day maxima "
             "use the prior local date and can use partial coverage when present. Missing inputs remain missing "
             "until model preprocessing. The 2023 training set alone fits imputers, scaling, and models. "
             "We use 2024 to select models and reserve 2025 for testing."),
        ],
        [
            ("heading", "Observation Cleaning and Category Counts"),
            ("text", "We kept FM-15 and FM-16 reports. In TMP and DEW, the first field gives tenths of a degree "
             "Celsius and the second gives quality. In WND, fields four and five give tenths of a meter per "
             "second and quality. We accepted quality codes 0, 1, 4, and 5 and treated sentinel ±9999 and "
             "invalid values as missing. Temperature is converted to Fahrenheit and relative humidity is "
             "derived from temperature and dew point, clipped to 0–100%."),
            ("text", "We parsed observation times as UTC, removed duplicate timestamps, and retained the "
             "last report per UTC hour for the exported hourly tables. The original timestamps are preserved "
             "and displayed in New York time. Source years in the table follow the UTC file names."),
            ("table", (["Station", "Source year", "Routine reports", "Hourly rows", "Temp missing", "RH missing", "Wind missing"],
                       [[row[key] for key in ("station", "source_year", "routine_reports", "hourly_rows",
                                             "missing_temp", "missing_rh", "missing_wind")] for row in quality],
                       [0.68, 0.83, 1.18, 1.05, 1.06, 1.01, 1.09])),
            ("figure", (COUNTS_FIGURE, "Figure 1. Eligible LGA dates by maximum temperature category and year.")),
        ],
        [
            ("heading", "Preliminary Model Results"),
            ("text", "We compared a training class-prior baseline with Logistic Regression, Random Forest, "
             "and XGBoost at each cutoff using LGA alone and all four stations. Hyperparameters were fixed. "
             "For each cutoff, we selected the non-baseline model and station set with the lowest 2024 "
             "validation log loss and kept its 2023-trained model for the demo."),
            ("table", (["Cutoff", "Selected model", "Stations", "Validation loss", "Test loss", "Test accuracy"],
                       [[hour_label(row["hour"]), row["model"], row["scope"], f"{row['validation_log_loss']:.4f}",
                         f"{float(test['log_loss']):.4f}", f"{float(test['accuracy']):.1%}"]
                        for row, test in zip(selected, test_rows)],
                       [0.65, 1.63, 1.23, 1.23, 0.92, 1.24])),
            ("text", "Accuracy scores the highest-probability category. Macro F1 weights the five categories "
             "equally. Log loss penalizes low probability assigned to the observed category. The multiclass "
             "Brier score averages the sum of squared errors over all five probabilities. Lower values "
             "are better for both probability scores."),
            ("subheading", "Noon Results Using LGA Only"),
            ("table", (["Model", "Split", *METRIC_HEADERS], [
                [row["model"], row["split"].title(), *metric_cells(row)] for row in noon
            ], [1.69, 1.03, 1.02, 1.04, 1.08, 1.04])),
            ("figure", (LOSS_FIGURE, "Figure 2. Test log loss by prediction cutoff, model, and station configuration.")),
        ],
        [
            ("heading", "High Confidence Errors and Next Steps"),
            ("text", f"Across all selected cutoffs, {adjacent} of {len(errors)} incorrect predictions missed "
             "by one category. The table ranks the five most confident mistakes by the probability assigned "
             "to the predicted category. The same date can contribute predictions at multiple cutoffs."),
            ("table", (["Date", "Cutoff", "Actual high", "Actual band", "Predicted band", "Confidence"],
                       error_rows, [1.12, 0.71, 1.02, 1.32, 1.45, 1.28])),
            ("subheading", "Limits of the Initial Experiment"),
            ("text", "The experiment has one year of training. The 2025 test data cover only the first part "
             "of the year, so the results reflect that seasonal mix. Broad categories can also conceal "
             "temperature errors within a band. The hourly-observation maximum may differ from the official "
             "daily maximum or a prediction-market settlement value."),
            ("text", "The retrospective analysis uses observation timestamps as the availability cutoff. "
             "It does not model historical publication delays. The live demo fetches Aviation Weather Center "
             "METAR on demand. Its provider and format differ from the training archive, so input coverage "
             "and measured values can differ."),
            ("text", "Accuracy alone does not establish probability calibration. We plan to expand the training "
             "period, check performance across seasons and category boundaries, and use reliability diagrams "
             "and another held-out period to evaluate calibration. A live replay that includes publication "
             "delays would help quantify the operational difference."),
            ("text", "For the next milestone, we suggest dividing the work among three tasks. One teammate can "
             "extend the data and coverage audit, one can evaluate calibration and model errors, and one can "
             "maintain the demo and update the report."),
            ("subheading", "Data References"),
            ("text", "NOAA NCEI Global Hourly Integrated Surface Database and its format specification. "
             "The exact S3 object URLs are in data/raw/manifest.json."),
            ("text", "National Weather Service Aviation Weather Center Data API. "
             "The demo uses the /api/data/metar endpoint."),
        ],
    ]
    for split, year in (("validation", 2024), ("test", 2025)):
        rows = [row for row in metrics if row["split"] == split]
        pages.append([
            ("heading", f"Full {split.title()} Results"),
            ("text", f"All model, station, and cutoff combinations use the same {len(days[year])} eligible {year} dates."),
            ("table", (["Cutoff", "Stations", "Model", "N", *METRIC_HEADERS], [
                [hour_label(row["hour"]), row["scope"], row["model"], row["n"], *metric_cells(row)] for row in rows
            ], [0.62, 1.03, 1.64, 0.41, 0.82, 0.78, 0.86, 0.74])),
        ])
    return pages, noon


def write_markdown(pages):
    lines = []
    prefixes = {"title": "# ", "subtitle": "## ", "heading": "## ", "subheading": "### "}
    for page in pages:
        for kind, content in page:
            if kind == "table":
                headers, rows, _ = content
                lines.extend(["| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]])
            elif kind == "figure":
                filename, caption = content
                lines.extend([f"![{caption}]({filename})", "", f"*{caption}*"])
            elif kind == "caption":
                lines.append(f"*{content}*")
            else:
                lines.append(prefixes.get(kind, "") + content)
            lines.append("")
    (OUTPUTS / "Progress1_Report.md").write_text("\n".join(lines), encoding="utf-8")


def write_docx(pages):
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    document.core_properties.title = TITLE
    document.core_properties.author = MEMBERS
    section = document.sections[0]
    section.left_margin = section.right_margin = Inches(0.8)
    section.top_margin = section.bottom_margin = Inches(0.72)
    section.footer_distance = Inches(0.36)
    styles = document.styles
    for name, size in (("Normal", 11), ("Title", 27), ("Subtitle", 13), ("Heading 1", 18),
                       ("Heading 2", 13), ("Caption", 9)):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string("000000")
    styles["Title"].font.bold = True
    styles["Subtitle"].font.italic = False
    title_border = styles["Title"].element.find(".//" + qn("w:pBdr"))
    if title_border is not None:
        title_border.getparent().remove(title_border)
    styles["Normal"].paragraph_format.space_after = Pt(7)
    styles["Normal"].paragraph_format.line_spacing = 1.12
    styles["Caption"].paragraph_format.space_before = Pt(3)
    styles["Caption"].paragraph_format.space_after = Pt(8)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("CS4260 Progress 1  ·  ").font.size = Pt(9)
    number = OxmlElement("w:fldSimple")
    number.set(qn("w:instr"), "PAGE")
    footer._p.append(number)

    def element(parent, name, **attributes):
        child = OxmlElement(f"w:{name}")
        for key, value in attributes.items():
            child.set(qn(f"w:{key}"), str(value))
        parent.append(child)
        return child

    def add_table(headers, rows, widths):
        table = document.add_table(rows=0, cols=len(headers))
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        borders = element(table._tbl.tblPr, "tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            element(borders, edge, val="single", sz=4, color="D9D9D9")
        margins = element(table._tbl.tblPr, "tblCellMar")
        for edge, width in (("top", 38), ("bottom", 38), ("left", 75), ("right", 75)):
            element(margins, edge, w=width, type="dxa")
        for index, values in enumerate([headers, *rows]):
            row = table.add_row()
            properties = row._tr.get_or_add_trPr()
            element(properties, "cantSplit")
            if index == 0:
                element(properties, "tblHeader")
            for cell, value, width in zip(row.cells, values, widths):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if index == 0:
                    element(cell._tc.get_or_add_tcPr(), "shd", fill="DCE7EF")
                elif index % 2 == 0:
                    element(cell._tc.get_or_add_tcPr(), "shd", fill="F6F8FA")
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.05
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = paragraph.add_run(value)
                run.font.size = Pt(9.5)
                run.bold = index == 0

    paragraph_styles = {
        "title": "Title", "subtitle": "Subtitle", "heading": "Heading 1",
        "subheading": "Heading 2", "caption": "Caption", "text": "Normal",
    }
    for page_index, page in enumerate(pages):
        if page_index:
            document.add_page_break()
        for item_index, (kind, content) in enumerate(page):
            if kind == "table":
                add_table(*content)
            elif kind == "figure":
                filename, caption = content
                document.add_picture(str(OUTPUTS / filename), width=Inches(6.65))
                figure_paragraph = document.paragraphs[-1]
                figure_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                figure_paragraph.paragraph_format.space_before = Pt(9)
                figure_paragraph.paragraph_format.space_after = Pt(0)
                figure_paragraph.paragraph_format.keep_with_next = True
                document.add_paragraph(caption, "Caption")
            else:
                paragraph = document.add_paragraph(content, paragraph_styles[kind])
                if kind == "text" and item_index > 0 and page[item_index - 1][0] == "table":
                    paragraph.paragraph_format.space_before = Pt(7)
    document.save(OUTPUTS / "Progress1_Report.docx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--docx-only", action="store_true", help="Use the existing figures to build Word output")
    mode.add_argument("--figures-and-markdown", action="store_true", help="Skip Word output")
    args = parser.parse_args()
    labels = read_csv(PROCESSED / "lga_daily_labels.csv")
    metrics = read_csv(OUTPUTS / "metrics.csv")
    selected = json.loads((OUTPUTS / "selected_models.json").read_text(encoding="utf-8"))
    quality = read_csv(PROCESSED / "data_quality.csv")
    errors = read_csv(OUTPUTS / "prediction_errors.csv")
    pages, noon = report_pages(labels, metrics, selected, quality, errors)
    if not args.docx_only:
        figures(labels, metrics)
        write_markdown(pages)
        with (OUTPUTS / "noon_lga_results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(noon)
    if not args.figures_and_markdown:
        write_docx(pages)


if __name__ == "__main__":
    main()
