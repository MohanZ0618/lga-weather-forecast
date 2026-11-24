# Forecasting the brackets listed on Polymarket

[English](README.md) · [简体中文](README.zh-CN.md)

This study estimates the probability of every listed bracket for the daily New York City high-temperature contract. Each forecast uses LaGuardia weather available by 09:00, 12:00 or 15:00 New York time. A correct top pick means the highest-probability bracket matched the official winner. On 413 dates with all three forecasts, the selected models were correct 746 times out of 1,239 (60.2%). At 09:00 alone they were correct on 202 days (48.9%). [Model comparison and hyperparameters](../results/bracket_study/old09/REPORT.md)

## Replay the selected forecasts

Use Python 3.13 and start in the repository root. The commands below use a Windows virtual environment. On macOS or Linux use `.venv/bin/python`. The included snapshots support offline training and evaluation after dependencies are installed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r final_project/bracket_study/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For a single archived morning forecast, run `.\.venv\Scripts\python.exe final_project/bracket_study/old09.py forecast --date 2026-06-01`.

## Select a family, then tune and freeze

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s final_project -p 'test*.py' -v
.\.venv\Scripts\python.exe final_project/bracket_study/train.py screen
.\.venv\Scripts\python.exe final_project/bracket_study/train.py tune
.\.venv\Scripts\python.exe final_project/bracket_study/train.py freeze
.\.venv\Scripts\python.exe final_project/bracket_study/replay.py
.\.venv\Scripts\python.exe final_project/bracket_study/report.py
```

The screen uses January–April 2025 validation to select a model family and weather sources for each hour. Tuning uses May–August validation for the selected family. Each fold fits on earlier weather and uses market outcomes settled by the end of the fold. The final fit uses weather labels through August 31, 2025. Run `train.py all` for all three training stages or pass `--hour 9`, `--hour 12` or `--hour 15` for one time. Results under `final_project/results/bracket_study` include the selected families, parameters, frozen models, and per-bracket probabilities. [Forecast export](../results/bracket_study/bracket_probabilities.csv.gz) · [Paired scores](../results/bracket_study/paired.csv)

## Compare morning models and forecast times

```powershell
.\.venv\Scripts\python.exe final_project/bracket_study/morning.py select
.\.venv\Scripts\python.exe final_project/bracket_study/morning.py predict
.\.venv\Scripts\python.exe final_project/bracket_study/morning.py score
.\.venv\Scripts\python.exe final_project/bracket_study/old09.py select
.\.venv\Scripts\python.exe final_project/bracket_study/old09.py evaluate
.\.venv\Scripts\python.exe final_project/bracket_study/time_study.py
```

The morning challenger uses actual top-pick accuracy to select its family in January–April 2025 and its probability settings in May–August. It compares a fixed fit with monthly updates that exclude the latest seven days. The scoring step joins each archived probability to its settled market ID. The original model's 15 later morning forecasts are bundled in `morning/reference_tail.csv.gz`. Rebuilding the morning features with `morning_data.py` requires local NOAA, IEM and HRRR raw caches. [Morning scores](../results/bracket_study/morning/metrics.csv)

The follow-up keeps the original morning model fixed. It selects probability adjustments on outcomes available before March 1, 2026, then checks them on subsequent dates. The forecast-time experiment fits the same morning architecture separately by hour and jointly across hours. All fits in this comparison use weather labels from before September 1, 2025. [Adjustment results](../results/bracket_study/old09/metrics.csv) · [Shared and separate fits](../results/bracket_study/old09/time_metrics.csv)

## Inputs and evaluation

- The station panels use NOAA Local Climatological Data for four airports. Weather Underground and NOAA contract rules use an all-report temperature proxy for training. Routine METAR contracts use their own routine-report proxy. Scoring uses the actual settled market winner and the applicable listed brackets. Contract archives lack a full history of rule revisions. [Weather and settlement archives](../rolling/README.md)
- The National Blend of Models (NBM) input uses hourly NBH guidance for KLGA. The archive object's publication timestamp must precede or equal issue time, and the forecast must cover the last whole hour of the local day. The archive search starts one hour before issue and tries two and three hours earlier as needed. To refresh these sources, run `nbm_archive.py` with network access. [NOAA NBM archive guide](https://vlab.noaa.gov/web/mdl/nbm-download)
- `bracket_probabilities.csv.gz` contains the date, issue time, assumed input delay, contract ID, bracket, probability and outcome. Probabilities sum to one per forecast. `daily.csv.gz` reports top-pick accuracy, log loss and Brier score. `course.csv` covers the original five bands. `inputs.json` records input hashes.
- The main three-time study ends November 6, 2026. Fifteen subsequent mornings use supplemental IEM observations. Confidence intervals pair forecasts by market date and resample seven-calendar-day blocks 5,000 times. All scores describe historical replay.
