# LGA same-day temperature bands

CS4260 Final Project · Mohan Zhang, Mingyeung Wu, Maxwell Liss

At 09:00, 12:00 and 15:00 New York time the model estimates the probability that LaGuardia's same-day hourly-observation maximum is below 40°F, 40 to below 55°F, 55 to below 70°F, 70 to below 85°F, or 85°F and above.

The [market-bracket experiment](bracket_study/README.md) evaluates the actual daily NYC temperature contracts at the same three issue times. The Streamlit replay opens on the selected models. They picked the winning bracket in 746 of 1,239 forecasts (60.2%) across 413 days with all three forecasts. At 09:00 alone they were correct on 202 of those days (48.9%). The [issue-time and morning-model comparison](results/bracket_study/old09/REPORT.md) reports shared versus separate training and early-morning calibration. The [original results report (Chinese)](results/bracket_study/REPORT.zh-CN.md) includes seasonal and listing-only baselines and the course five-band comparison.

## Reproduce the paper

Run the following from the extracted archive root using Python 3.13.

```text
python -m pip install -r final_project/requirements.txt
python final_project/experiment.py
python -m unittest discover -s final_project -p test_final.py -v
```

The second command retrains every candidate from the included sample files and writes `final_project/results/main_table.csv`, `paired_comparisons.csv`, and `reliability.png`. It also writes the validation grid, all model scores, individual predictions, and error summaries. Python package installation requires an internet connection. Training from the included data runs offline. The bundled tables have one row per eligible date at each cutoff.

## Extension (paper Section 7)

Run after `experiment.py`, from the extracted archive root. Each script runs offline from the bundled tables in under two minutes.

```text
python final_project/supplement.py
python final_project/distribution.py
python final_project/guidance_study.py
python final_project/history_study.py
python final_project/class_eval.py
python final_project/class_models.py
python final_project/mos_history.py
python final_project/observed_weather.py
python final_project/nonstationarity.py
cd final_project && python -m unittest test_final test_guidance test_history test_class_eval test_class_models test_mos_history test_observed_weather test_nonstationarity
python make_paper.py
```

The extension adds station guidance from the Iowa Environmental Mesonet archive: LAMP and the National Blend of Models text bulletin, using the newest run released by each cutoff (`guidance.py`). A censored Gaussian model of the warming still to come (`distribution.py`, `shash.py`) lowers test log loss from 0.258, 0.234 and 0.094 to 0.200, 0.159 and 0.072 on the same 359 days. The other scripts test longer history, course methods (learning curves, cross-validation, bagging, a neural network), observed weather and nearby stations, and model and sensor drift. None of them beats the selected model. Each script writes `results/<name>/`, and `results/<name>/README_SECTION.md` explains its result. `make_paper.py` builds Section 7 from those files through `paper_extension.py`. To download the public source files again, run `guidance.py fetch`, `mos.py fetch`, `asos.py fetch` and `history.py`; the matching `build` steps rebuild the bundled feature tables.

## Data provenance and rebuilding

- `prepare.py` downloads 36 [NOAA LCD v2 files](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data) derived from GHCNh. `data/station_sources.csv` contains station IDs in the URLs, source coverage and SHA-256 hashes. It converts the fixed Eastern standard time in the CSV to actual New York time, retains FM-15 and FM-16 observations, and keeps the last report in each completed UTC hour for daily labels. `timestamp_overlap_2024.csv` and `archive_overlap_2024.csv` audit the switch from the Progress 1 Global Hourly source.
- `hrrr.py` reads one tile per date from the [University of Utah HRRR Zarr forecast archive](https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/ex_python_plot_zarr.html). The closest cell is grid row 698, column 1556 (tile 0.4.10). The script uses forecast leads 1–18 from 10:00 UTC and requires the S3 Last-Modified timestamp to precede or equal the earliest cutoff. The archive timestamp is evidence of object availability. It may exclude reuploaded files. The compact `data/hrrr_cycles.csv.gz` cache contains point forecasts, source URLs, timestamps, hashes, and exclusion statuses.
- To rebuild the observation tables, run `python final_project/prepare.py`. To rebuild forecast features from the point-forecast cache, run `python final_project/hrrr.py`. To fetch the original HRRR tiles again, run in a fresh copy without the cycle cache. This transfers several hundred MB. The remote archive can revise station records or replace objects, so refreshed results can differ from the supplied snapshot.

## Experimental protocol

The observation cohort trains on 2018–2023. The matched HRRR cohort trains on available data in 2021–2023. All models select on 2024 and reserve September 2025 through August 2026 for final testing. January through August 2025 is reported separately for context. Within a cohort, all methods use identical evaluation dates. The four feature configurations allow direct station, trend, and NWP comparisons. Validation log loss selects a setting per family and the two models for equal-weight blending. Random forest and XGBoost use a fixed seed. The Brier score is the sum of squared probability errors over the five classes averaged over dates.

The code uses the team's Progress 1 definitions from `weather.py` and standard models from scikit-learn and XGBoost. The report cites primary sources and documents AI assistance.

## Update the contribution statement

Save `final_project/contributions.json` as a JSON object mapping each of the three authors' names as written above to a one-sentence description of their actual contribution. After running the experiment, install ReportLab and run `python final_project/make_paper.py` to rebuild the PDF with those statements.
