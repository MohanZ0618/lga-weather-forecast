# Real-time LGA temperature probabilities

Run the following from the weather-project directory. The local model file is included.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Day-ahead and same-day comparison

The app opens on **Forecast comparison**. Select Today, Tomorrow, Yesterday, or Historical comparison. The orange dashed line is the daily high forecast using data available by **noon on the preceding New York calendar day**. The blue line updates the same target at each issued quarter-hour while the app is open. Green markers are hourly LGA observations. After a completed day has sufficient observations, the chart shows its measured high and the signed error of each forecast. During the day it shows the gap to measured temperature. The app records each previous-noon forecast when first requested and reuses it on later visits. A first visit after the cutoff reconstructs it from receipt-filtered reports and an already-published forecast cycle.

Both horizons offer 16 ranges by default. Interior ranges are 5°F wide with open tails below 25°F and at or above 95°F. The original five ranges are also available. Fine probabilities allocate each broad band's probability according to the smoothed distribution of historical temperature errors from the calibration period. Completed-hour observations set the intraday lower bound. The probability trend lets you select individual ranges.

The day-ahead estimator uses four airports at the previous noon and, when available, all hourly 2 m temperatures for the entire target day from the prior day's 12 UTC HRRR forecast. The numerical cycle must have been published by noon. A station-only estimator covers missing cycles. The intraday estimator corrects the expected maximum using aircraft-weather station reports, HRRR, GOES cloud features and historical remaining warming. The exact-temperature estimators compared a baseline, ridge regression, and gradient boosting by validation MAE. The original frozen classifier continues to provide same-day broad-band probabilities. Previous-noon broad probabilities come from the day-ahead estimator's calibration errors. [HRRR archive](https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_variables.html) · [METAR API](https://aviationweather.gov/data/api/)

Run `python nowcast/train_temperature.py` to reproduce the temperature models with the local station archive, cached previous-noon cycles and the original fusion study outputs. Add `--fetch-weather` to fill missing sampled HRRR cycles. Fitting uses target dates through December 2024. January–August 2025 selects the estimator. September 2025–August 2026 supplies calibration errors. Autumn 2026 is a retrospective evaluation on matching dates for both horizons. The historical station inputs use observation time as a proxy for receipt time. The live path checks receipt time. Autumn results apply to the sampled LGA days and times. See `results/temperature_metrics.csv`, `results/temperature_validation.csv`, and `results/temperature_replay.parquet`.

The **Latest observations** tab refreshes every minute while the page is open and issues one forecast per completed 15-minute interval from 06:00 through 20:00 America/New_York. Outside that window it shows the most recent scheduled forecast with its date. The chart fills with successfully issued forecasts. **Fusion backtest** displays sampled out-of-sample forecasts. The five Fahrenheit bands use cut points of 40, 55, 70 and 85. The target is the maximum of the retained hourly station observations for the local calendar day.

## Data and availability

- Four airports provide temperature, dew point, wind and cloud reports. The live reader checks METAR receipt time and observation time. Historical LCD v2 timestamps are fixed Eastern standard time. Replay also evaluates 15- and 30-minute station-report delays. The high-so-far feature uses the last report in each completed hour, allowing a 15-minute settling margin. A missing fresh LGA temperature or missing historical high triggers the previous successful forecast, visibly dated. [Aviation Weather Center](https://aviationweather.gov/data/api/) · [NOAA LCD v2](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data)
- The numerical forecast reader checks cycles from newest to oldest, up to 12 hours back. The LGA grid point is row 698, column 1556. A cycle must already be published and cover the last full hour of the local day. The features include remaining hourly maximum, peak timing, contemporaneous forecast bias and total cloud cover. A missing cloud field can be imputed. Archived temperature-cycle availability determines the HRRR fallback. [University of Utah point extraction](https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/ex_python_plot_zarr.html)
- NOAA GOES-East CONUS cloud masks are projected onto the LGA region using the navigation in each NetCDF file. Only valid binary decisions with good DQF enter each regional average. At least half of a region's pixels must be usable. The feature table includes 10 km and 40 km cloud fraction, an upwind sector, 30/60-minute changes, solar elevation interactions and the difference from HRRR cloud cover. The upwind sector follows the observed wind direction and spans ±45° from 10 to 40 km. Scan end and archive availability must precede the corresponding reference time. Current cloud information expires after 30 minutes. GOES-19 replaced GOES-16 at 15:10 UTC on April 7, 2025. [NOAA product documentation](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC01503) · [GOES-East handover](https://www.ospo.noaa.gov/data/messages/2025/04/MSG_20250407_1510.html)

## Training and evaluation

The four-season pilot uses seven dates in each season. The wider backfill rotates quarter-hour cutoffs across morning, midday and afternoon/evening. Outside the pilot it samples one cutoff every third day before September 2026 and three cutoffs on each subsequent test day. This bounds archive transfer. The live scheduler supports all 57 slots. Older archive objects with replacement timestamps after the cutoff are excluded.

Historical LGA days from 2018–2020 supply the distribution of remaining warming at each quarter-hour, weighted by season and current observed high. Candidates include the analogue probabilities, logistic regression, random forest, XGBoost, quantile boosting on remaining warming, and a shared logistic model for crossing each temperature boundary. The threshold model uses the distance to the boundary and the available weather inputs. Ordered exceedance probabilities are converted to five band probabilities; bands ruled out by completed-hour observations receive zero probability.

Two expanding chronological folds with validation years 2023 and 2024 produce out-of-fold probabilities for fitting temperatures and identifying blend candidates. Models fit through December 2024. January–August 2025 selects candidates by date-weighted accuracy with a 1% relative log-loss tolerance versus logistic regression. Routing also compares each richer tier against the validated fallback. The frozen selection is recorded in `results/selection.json`. September 2025–August 2026 is the previously studied comparison period. September–November 2026 is a separate autumn test. Each model comparison documents its coverage. A gain at one clock time or on one airport's autumn data can differ in other seasons.

From the repository directory:

```powershell
.\.venv\Scripts\python.exe nowcast/runtime.py
.\.venv\Scripts\python.exe nowcast/runtime.py --replay nowcast/examples/20261120-2000.inputs.json
.\.venv\Scripts\python.exe -m unittest discover -s nowcast -p test_nowcast.py -v
```

To retrain offline from the included modeling table and analogue reference, run `python nowcast/study.py --reuse-samples`. To rebuild the modeling table, start with the four-station raw files prepared by `final_project/prepare.py`, run `python nowcast/backfill.py --pilot`, then `python nowcast/backfill.py`, then `python nowcast/study.py`. Rebuilding from NOAA transfers several gigabytes. The downloader caches compact statistics, source URLs and forecast timestamps. To regenerate paired comparisons and coverage from the snapshot cache, run `python nowcast/evaluate.py`.

## Independent local training (Windows)

From the project directory, run `.\nowcast\start_training.ps1`. Each invocation starts one detached Python process using the project environment and cached data. You can close the terminal and chat while the computer stays awake. The launcher prints the run directory under `nowcast/training_runs`. The worker updates `status.json` and `training.log` as each model group runs. Errors are recorded in `errors.log`.

This candidate uses observations through August 2025 for model fitting and September 2025 through August 2026 for model and input selection. The worker evaluates the frozen and candidate systems on the same validation dates and saves `comparison.json`, `validation.csv`, and a candidate `model.joblib` in the run directory. The app loads its deployed model from `nowcast/artifacts`. The candidate's validation scores are selection results. A prospective accuracy estimate requires later observations.

## Frozen evaluation

The autumn evaluation covered 67 days from September 1 through November 6, 2026, with three sampled forecasts per day. Accuracy weights each date equally. Across the 201 forecasts the deployed system reached 82.09% band accuracy and 0.3990 log loss. The airport-only model reached 85.57% and 0.3608. In the preceding September 2025–August 2026 evaluation, the corresponding accuracies were 89.92% and 85.71% across 119 sampled dates.

On the 199 autumn forecasts with both HRRR and satellite data, holding the logistic estimator and training cases fixed gave identical accuracy after adding local satellite cloud features. Log loss improved from 0.3597 to 0.3498. Adding upwind and cloud-change features kept accuracy level and gave log loss of 0.3577. The deployed system also changes estimators between tiers. See the control rows in `results/ablation.csv` for the feature-only comparison.

## Result files

- `metrics.csv` gives date-weighted accuracy, log loss and multiclass Brier score, with counts. It includes separate delay assumptions, time windows, individual sampled quarter-hours and boundary errors for the deployed system.
- `ablation.csv` compares each additional data source. The control rows use logistic regression and identical training and test dates for the pair. The fixed-versus-latest control excludes the HRRR cloud variable from both sides to isolate cycle timing.
- `paired_systems.csv` compares model-selected tiers and the deployed route on matched cases. Intervals resample whole dates with replacement; they describe sampling variability within the test period.
- `coverage.csv`, `validation.csv` and `predictions.parquet` give source coverage, candidate scores and the underlying five probabilities.

