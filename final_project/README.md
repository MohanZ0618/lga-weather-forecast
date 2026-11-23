# LGA same-day temperature bands

CS4260 Final Project · Mohan Zhang, Mingyeung Wu, Maxwell Liss

At 09:00, 12:00 and 15:00 New York time the model estimates the probability that LaGuardia's same-day hourly-observation maximum is below 40°F, 40 to below 55°F, 55 to below 70°F, 70 to below 85°F, or 85°F and above.

## Reproduce the paper

Run the following from the extracted archive root using Python 3.13.

```text
python -m pip install -r final_project/requirements.txt
python final_project/experiment.py
python -m unittest discover -s final_project -p test_final.py -v
```

The second command retrains every candidate from the included sample files and writes `final_project/results/main_table.csv`, `paired_comparisons.csv`, and `reliability.png`. It also writes the validation grid, all model scores, individual predictions, and error summaries. Python package installation requires an internet connection. Training from the included data runs offline. The bundled tables have one row per eligible date at each cutoff.

## Data provenance and rebuilding

- `prepare.py` downloads 36 [NOAA LCD v2 files](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data) derived from GHCNh. `data/station_sources.csv` contains station IDs in the URLs, source coverage and SHA-256 hashes. It converts the fixed Eastern standard time in the CSV to actual New York time, retains FM-15 and FM-16 observations, and keeps the last report in each completed UTC hour for daily labels. `timestamp_overlap_2024.csv` and `archive_overlap_2024.csv` audit the switch from the Progress 1 Global Hourly source.
- `hrrr.py` reads one tile per date from the [University of Utah HRRR Zarr forecast archive](https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/ex_python_plot_zarr.html). The closest cell is grid row 698, column 1556 (tile 0.4.10). The script uses forecast leads 1–18 from 10:00 UTC and requires the S3 Last-Modified timestamp to precede or equal the earliest cutoff. The archive timestamp is evidence of object availability. It may exclude reuploaded files. The compact `data/hrrr_cycles.csv.gz` cache contains point forecasts, source URLs, timestamps, hashes, and exclusion statuses.
- To rebuild the observation tables, run `python final_project/prepare.py`. To rebuild forecast features from the point-forecast cache, run `python final_project/hrrr.py`. To fetch the original HRRR tiles again, run in a fresh copy without the cycle cache. This transfers several hundred MB. The remote archive can revise station records or replace objects, so refreshed results can differ from the supplied snapshot.

## Experimental protocol

The observation cohort trains on 2018–2023. The matched HRRR cohort trains on available data in 2021–2023. All models select on 2024 and reserve September 2025 through August 2026 for final testing. January through August 2025 is reported separately for context. Within a cohort, all methods use identical evaluation dates. The four feature configurations allow direct station, trend, and NWP comparisons. Validation log loss selects a setting per family and the two models for equal-weight blending. Random forest and XGBoost use a fixed seed. The Brier score is the sum of squared probability errors over the five classes averaged over dates.

The code uses the team's Progress 1 definitions from `weather.py` and standard models from scikit-learn and XGBoost. The report cites primary sources and documents AI assistance.

## Update the contribution statement

Save `final_project/contributions.json` as a JSON object mapping each of the three authors' names as written above to a one-sentence description of their actual contribution. After running the experiment, install ReportLab and run `python final_project/make_paper.py` to rebuild the PDF with those statements.
