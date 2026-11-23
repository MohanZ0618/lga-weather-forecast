import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const build = path.join(root, "build");
const delivery = path.join(root, "deliverables");
const { Presentation, PresentationFile } = await import(pathToFileURL(process.env.CS4260_ARTIFACT_TOOL).href);
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const colors = { ink: "#153245", muted: "#425866", teal: "#147287", light: "#D8EBEF",
  white: "#FFFFFF", bg: "#F4F8F9", dark: "#113044", orange: "#A34B27" };
const hrrr = "Four stations + trends + HRRR";
const trends = "Four stations + trends";
const cutoffs = ["9", "12", "15"];
const band = ["under 40°F", "40 to <55°F", "55 to <70°F", "70 to <85°F", "85°F and up"];

async function records(file) {
  const [header, ...lines] = (await fs.readFile(path.join(root, "results", file), "utf8")).trim().split(/\r?\n/);
  const keys = header.split(",");
  return lines.map(line => Object.fromEntries(line.split(",").map((value, i) => [keys[i], value])));
}

const main = await records("main_table.csv");
const paired = await records("paired_comparisons.csv");
const edge = await records("by_near_boundary.csv");
const [miss] = await records("errors.csv");
const all = await records("all_metrics.csv");
const fmt = value => Number(value).toFixed(3);
const pct = value => (100 * Number(value)).toFixed(1) + "%";
const change = value => (Number(value) >= 0 ? "+" : "−") + Math.abs(value).toFixed(4);
const row = (hour, scope, cohort = "Observations") => main.find(
  item => item.hour === hour && item.scope === scope && item.cohort === cohort);

function text(slide, value, x, y, width, height, size, options = {}) {
  const shape = slide.shapes.add({
    name: value.slice(0, 50), geometry: "textbox",
    position: { left: x, top: y, width, height },
    fill: "none", line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { typeface: "Aptos", fontSize: size + "pt", color: options.color ?? colors.ink,
    bold: options.bold ?? false, verticalAlignment: "middle",
    insets: { left: 0, right: 0, top: 0, bottom: 0 } };
}

function rect(slide, x, y, width, height, fill) {
  slide.shapes.add({ geometry: "rect", position: { left: x, top: y, width, height },
    fill, line: { fill: "none", width: 0 } });
}

function slideFor(title, note) {
  const slide = deck.slides.add();
  slide.background.fill = colors.bg;
  rect(slide, 64, 50, 64, 5, colors.teal);
  text(slide, title, 64, 73, 1152, 75, 36, { bold: true });
  text(slide, String(deck.slides.items.length).padStart(2, "0"), 1167, 658, 49, 23, 12, { color: colors.muted });
  slide.speakerNotes.text = note;
  return slide;
}

function lines(slide, headings, details, start = 190, step = 130) {
  headings.forEach((heading, i) => {
    const y = start + i * step;
    text(slide, heading, 64, y, 1140, 39, 23, { bold: true, color: colors.teal });
    text(slide, details[i], 64, y + 43, 1110, step - 45, 19);
  });
}

const cover = deck.slides.add();
cover.background.fill = colors.dark;
rect(cover, 70, 117, 110, 7, "#7DC8D4");
text(cover, "Predicting LaGuardia’s daily\nmaximum temperature band", 70, 190, 1130, 188, 50,
  { color: colors.white, bold: true });
text(cover, "Same-day probability forecasts at 09:00, 12:00, and 15:00", 74, 400, 1110, 45, 24,
  { color: "#C9E7EC" });
text(cover, "Mohan Zhang, Mingyeung Wu, and Maxwell Liss", 74, 567, 1110, 40, 19, { color: colors.white });
text(cover, "CS4260 · Fall 2026", 74, 613, 1110, 30, 15, { color: "#C9E7EC" });
cover.speakerNotes.text = "Presenter 1, 0:00–0:40. Open with the three daily forecast times and the decision to predict a probability distribution.";

const problem = slideFor("One date, five possible temperature bands",
  "Presenter 1, 0:40–2:00. Define x(d,h), y(d) and p(y|x). Explain the target using the last report in each UTC hour and the New York calendar day. Walk through 69.98°F versus 70.00°F.");
band.forEach((label, i) => {
  const x = 64 + i * 232;
  rect(problem, x, 250, 225, 84, ["#DCECF3", "#C1E0EB", "#A2D0DD", "#F0DACA", "#EDC6AF"][i]);
  text(problem, label, x + 12, 269, 204, 43, 20, { bold: true });
});
text(problem, "Target", 64, 395, 1140, 34, 20, { bold: true, color: colors.teal });
text(problem, "The day’s highest retained hourly temperature at LGA.", 64, 438, 1140, 50, 24);
text(problem, "Output", 64, 509, 1140, 34, 20, { bold: true, color: colors.teal });
text(problem, "A probability for each band, using only information available at the cutoff.", 64, 549, 1150, 57, 21);

const data = slideFor("A full-year test follows model selection",
  "Presenter 1, 2:00–4:00. Explain why no shuffled split. Show four station names and mention that LCD v2 uses fixed EST. The mean absolute temperature difference at 8,757 overlapping timestamps was 0.001°F after alignment. Source: https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data");
const timeline = [["2018–2023", "Train"], ["2024", "Select"], ["Jan–Aug 2025", "Progress overlap"], ["Sep 2025–Aug 2026", "Final holdout"]];
timeline.forEach(([dates, role], i) => {
  const x = 65 + i * 290;
  rect(data, x, 234, 270, 6, i === 3 ? colors.orange : colors.teal);
  text(data, dates, x, 263, 270, 45, 19, { bold: true });
  text(data, role, x, 321, 270, 44, 20, { color: colors.muted });
});
text(data, "LGA, JFK, Newark, and Teterboro", 64, 441, 1135, 50, 25, { bold: true });
text(data, "NOAA station observations. Time stamps were cross-checked against the 2024 archive.", 64, 501, 1124, 68, 20);

lines(slideFor("How much warming is still possible?",
  "Presenter 2, 4:00–5:20. Explain the Gaussian weighting of earlier days by month and observed high so far. Walk through physical support: an already-observed high of 72°F rules out colder bands. HRRR is a physics model. Source: https://rapidrefresh.noaa.gov/hrrr/"),
  ["Monthly climatology", "Remaining-warming baseline", "HRRR forecast features"],
  ["Estimate the five bands from historical frequencies for the same month.",
    "Add historical temperature increases to today’s high so far. Weight similar days more heavily.",
    "Use the modeled remaining-day maximum and modeled warming from a verified 10:00 UTC cycle."]);

lines(slideFor("Five model families and an equal-weight blend",
  "Presenter 2, 5:20–6:45. Explain logistic logits, neighbor votes, Gaussian likelihoods, bootstrapped trees and boosted trees. Show that 2024 picks hyperparameters within each model, the two families for the blend, and whether the blend is selected. The final test was scored after the settings were fixed."),
  ["Linear, neighbor, and generative models", "Bagged and boosted trees", "Probability averaging"],
  ["Logistic regression, distance-weighted KNN, and Gaussian naive Bayes.",
    "Random forest and XGBoost. Evaluate three settings per model family.",
    "Average the two best families on 2024 validation. Log loss selects the final configuration."]);

const scores = slideFor("Forecast uncertainty falls through the day",
  "Presenter 2, 6:45–8:00. Explain why log loss looks at the full probabilities. Both series here use the same 359 test dates. Point to the difference between early and late predictions.");
text(scores, "Test log loss on 359 matched dates · lower is better", 64, 144, 1130, 35, 15, { color: colors.muted });
const values = cutoffs.flatMap(hour => [
  Number(row(hour, "Remaining warming", "Matched HRRR").log_loss), Number(row(hour, hrrr, "Matched HRRR").log_loss)]);
const scale = 700 / Math.max(...values);
cutoffs.forEach((hour, i) => {
  const y = 188 + 133 * i;
  text(scores, hour.padStart(2, "0") + ":00", 64, y + 15, 126, 55, 25, { bold: true });
  for (let j = 0; j < 2; j++) {
    const value = values[i * 2 + j];
    const width = value * scale;
    rect(scores, 209, y + j * 39, width, 28, j === 0 ? "#648396" : colors.teal);
    text(scores, fmt(value), 222 + width, y + j * 39 - 2, 152, 34, 18, { bold: true });
  }
});
rect(scores, 209, 609, 21, 21, "#648396");
text(scores, "Remaining-warming baseline", 242, 595, 380, 48, 18);
rect(scores, 693, 609, 21, 21, colors.teal);
text(scores, "Four stations + trends + HRRR", 725, 595, 460, 48, 18);

const comparison = slideFor("Which inputs improve the forecasts?",
  "Presenter 3, 8:00–9:35. Walk across the paired changes. Negative is better. Translate a morning change into practical terms. The HRRR comparison trains both models on matched dates. Explain why a ranking by final test score cannot select a production model retrospectively.");
const cols = ["Add nearby stations", "Add trends", "Add HRRR"];
cols.forEach((label, i) => text(comparison, label, 265 + i * 310, 222, 296, 53, 20, { bold: true }));
cutoffs.forEach((hour, i) => {
  const y = 304 + 79 * i;
  const options = paired.filter(item => item.hour === hour);
  rect(comparison, 64, y, 1151, 63, i % 2 === 0 ? colors.light : "#EAF3F5");
  text(comparison, hour.padStart(2, "0") + ":00", 85, y + 12, 156, 41, 22, { bold: true });
  [options[0], options[1], options[3]].forEach((item, col) =>
    text(comparison, change(item.delta_log_loss), 265 + col * 310, y + 12, 240, 41, 22,
      { bold: true, color: Number(item.delta_log_loss) < 0 ? colors.teal : colors.orange }));
});
text(comparison, "Change in mean test log loss on paired dates. Negative values favor the added inputs.",
  64, 578, 1145, 52, 17, { color: colors.muted });

const errors = slideFor("Forecasts are harder near a band boundary",
  "Presenter 3, 9:35–11:00. Compare the two accuracy groups. Explain that these are descriptive slices, not separate training tasks. Rehearse the actual highest-confidence error and identify the crossed threshold.");
const near = edge.find(item => item.scope === hrrr && item.hour === "9" && item.near_boundary === "True");
const far = edge.find(item => item.scope === hrrr && item.hour === "9" && item.near_boundary === "False");
text(errors, pct(near.accuracy), 64, 204, 530, 93, 51, { bold: true, color: colors.orange });
text(errors, pct(far.accuracy), 653, 204, 530, 93, 51, { bold: true, color: colors.teal });
text(errors, "Within 2°F of a cutoff", 64, 307, 560, 50, 23);
text(errors, "Farther from a cutoff", 653, 307, 530, 50, 23);
text(errors, "09:00 HRRR model · " + near.n + " near-boundary days and " + far.n + " farther-away days",
  64, 390, 1150, 44, 18, { color: colors.muted });
const missTime = miss.hour.padStart(2, "0") + ":00";
text(errors, miss.date.slice(0, 10) + " at " + missTime, 64, 487, 1130, 39, 21, { bold: true });
text(errors, pct(miss.confidence) + " on " + band[Number(miss.predicted)] +
  ". The observed high was " + Number(miss.target_max).toFixed(2) + "°F.", 64, 542, 1148, 53, 20);

const end = slideFor("What we would test next",
  "Presenter 3, 11:00–12:00. Summarize the information value by cutoff, explain the archive timing limitation, then invite questions. Each member should answer at least one question.");
const hrrrWins = paired.filter(item => item.candidate === hrrr && Number(item.delta_log_loss) < 0).length;
const betterNear = paired.filter(item => item.candidate === "Four stations" && Number(item.delta_log_loss) < 0).length;
const blendComparison = all.filter(item => item.split === "test" && item.selected === "True" &&
  item.model === "Two-model probability average").length;
lines(end, ["What the holdout showed", "What the archive can establish", "Next experiment"],
  ["HRRR lowered loss at " + hrrrWins + " cutoffs, with its largest gain at 09:00. Nearby stations helped at " +
    betterNear + " cutoffs. Validation selected the blend " + blendComparison + " times.",
   "One airport, one full-year test, and an hourly-observation target. Archived station timestamps represent measurement time.",
   "Test calibration across additional years and verify arrival times with a real-time station feed."]);

await fs.mkdir(build, { recursive: true });
await fs.mkdir(delivery, { recursive: true });
const file = await PresentationFile.exportPptx(deck);
await file.save(path.join(build, "CS4260_Presentation_Draft.pptx"));
for (const [index, slide] of deck.slides.items.entries()) {
  const png = await slide.export({ format: "png", scale: 0.7 });
  await fs.writeFile(path.join(build, "slide-" + (index + 1) + ".png"), new Uint8Array(await png.arrayBuffer()));
}
