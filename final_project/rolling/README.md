# Fixed-cutoff rolling validation

Run the commands below from the `weather-project` directory with the project environment.
The source stage uses the four station archives in `final_project/data/raw`, the original
fixed-cycle cache, and public HRRR and GOES objects. The compact features and publication
records are stored in `final_project/data/rolling`. Historical station timestamps proxy
for receipt time. The delayed replays shift all same-day station inputs by 15 or 30
minutes and use only fully closed receipt-adjusted hours for the observed lower bound.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s final_project -p 'test*.py' -v
.\.venv\Scripts\python.exe final_project/experiment.py --out final_project/results/rolling/reproduction
.\.venv\Scripts\python.exe final_project/rolling/data.py --refresh-tail
.\.venv\Scripts\python.exe final_project/rolling/data.py
.\.venv\Scripts\python.exe final_project/rolling/data.py --autumn-fixed
.\.venv\Scripts\python.exe final_project/rolling/data.py --external
.\.venv\Scripts\python.exe final_project/rolling/provenance.py
.\.venv\Scripts\python.exe final_project/rolling/study.py stations
.\.venv\Scripts\python.exe final_project/rolling/study.py fixed
.\.venv\Scripts\python.exe final_project/rolling/study.py search
.\.venv\Scripts\python.exe final_project/rolling/audit.py
.\.venv\Scripts\python.exe final_project/rolling/late_replay.py
```

Use `--hour 9`, `--hour 12` or `--hour 15` with a training stage to run one hour.
Completed archive days are resumable. Re-run the external stage after transient
source errors. The original fixed 10 UTC cycle uses the project's 09:00 availability
rule for every cutoff. New cycles use their actual archival availability and cover
the last full hour of the local day. GOES requires a scan ending within 30 minutes,
with both scan completion and availability at or before the issue time.
The cloud forecast feature is mean cloud cover over remaining hourly leads.

The station stage compares 2018 and 2021 training starts and four feature sets with
the original hour-specific ensemble fixed. The fixed and search stages carry forward
the selected station feature set. The advanced tier with the lowest validation loss
under the fixed estimators enters the full hyperparameter search alongside the station
and fixed-cycle tiers. Each finalist tier gets the same 30 configurations and an equal
blend of the best two different families. Unavailable rich inputs route to the
validation-selected station model. Individual fields missing within a station can be
imputed from training data. Fields wholly missing during fitting are excluded.

Selection minimizes the equal-weight mean of the 2023, 2024 and January–August 2025
fold losses. Within 0.001, selection prefers fewer data sources, then a single model,
then fewer input fields and lower loss. The selected models and the comparison models
are refitted through August 2025. The historical and autumn replay scores are separate.
The reference in the new replay uses the original ensemble configuration refitted
through the same training endpoint, including a station fallback. The original
published fits are reproduced separately. Weekly intervals use 5,000 paired draws
of seven-calendar-day blocks with a fixed seed.

Under `final_project/results/rolling`, each hour has its selected bundle, ranked search
settings, fold scores, per-date probabilities, reliability bins, and matched and routed
input comparisons. The root contains the combined scores, chosen and alternative
configurations, upstream hashes, data quality and verification results.

The downloadable NOAA LCD snapshot ends on the morning of November 7, 2026.
The separate Iowa Environmental Mesonet replay compares overlapping hourly-maximum
labels, then evaluates later eligible METAR dates through November 21. This supplement
uses the already frozen models and is reported separately because the archive and
measurement precision differ.

## Historical Polymarket brackets and one-degree probabilities

Run these after the weather inputs above are available. Market archives use the
official Gamma NYC Daily Weather series (10005). Each day supplies its own listed
ranges, opening timestamps, written resolution rules and uniquely resolved Yes market.

```powershell
.\.venv\Scripts\python.exe final_project/rolling/market_data.py
.\.venv\Scripts\python.exe final_project/rolling/market_study.py
.\.venv\Scripts\python.exe final_project/rolling/market_audit.py
```

Historical rules vary between Weather Underground, NOAA station time series and
routine AWC METAR reports. The recent METAR rules prefer the signed tenths-Celsius
temperature in the remark, fall back to the body temperature, exclude SPECI and use
local observation dates. Conversion to whole Fahrenheit uses nearest integer with
exact halves rounded toward positive infinity. AWC archive responses can reflect
later corrections. Archived station observations supply exact-degree proxies for days
without the designated source's exact recorded value. A winning two-degree bracket
does not identify the winning individual degree. The market metrics always use the
actual resolved Yes bracket.

For each issue time a regression predicts a high relative to a station or numerical
weather anchor. Its bias and residual scale are estimated on the preceding 120 days
with an earlier training-only fit. The regressor is then refitted through the fold
cutoff. Integrating the residual distribution between half degrees gives probabilities
for individual integer temperatures. The models compare ridge regularization, tree
leaf counts and uncertainty scale multipliers. Historical official bracket log loss
before September 2025 selects the combination. Earlier folds provide a secondary
exact-degree proxy score. The final interval and all market weather predictions use
the August 2025 training endpoint. Prices are only read as final binary payouts for
settlement. The archived wording and listing times describe the event as reconstructed
today; an intraday revision history is unavailable.

In `results/rolling/polymarket`, `historical_contract_bins.csv.gz` holds each actual
listed bracket. `integer_labels.csv` holds whole-degree observations and proxies.
`one_degree_*.parquet` contains the complete degree distributions on replay dates.
`contract_probabilities.csv.gz` gives the sum for every eligible listed bracket.
`market_metrics.csv`, `paired_results.csv` and `degree_metrics.csv` distinguish
market settlement accuracy from agreement with the single-degree archive target.

