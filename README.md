# Same Day Weather Forecasting at LaGuardia

CS4260 project by Mohan Zhang, Mingyeung Wu, and Maxwell Liss.

The default **Polymarket 历史回放** page replays the archived same-day probabilities at 09:00, 12:00 and 15:00 New York time. Choose a date and cutoff, inspect the probability of each listed bracket, and reveal the official settlement. The **Forecast comparison** page in the sidebar plots the previous-noon forecast, same-day updates, measured temperatures, and forecast errors for a common target date. It offers both 5°F ranges and the original five categories. The [forecast guide](nowcast/README.md) explains the two horizons and their held-out scores.

The **Latest observations** page uses a quarter-hour model from 06:00 through
20:00 New York time with airport METAR, HRRR cycles, and GOES-East cloud masks.
**Fusion backtest** replays its evaluated forecasts. The original three-cutoff
study is available under **Historical replay**. See the [nowcast guide](nowcast/README.md)
for launch, data coverage, and reproduction commands.

The repository includes processed observations, fitted models, experiment results, and reports. After installing the dependencies, run `streamlit run app.py` to open the demo. The raw NOAA CSVs are downloaded and cached by `python weather.py`. Their source URLs and SHA-256 hashes are recorded in `data/raw/manifest.json`.

The original proposal forecast the next day's high. This implementation predicts **today's final hourly-observation maximum at LGA** at 09:00, 12:00, and 15:00 New York time. It returns probabilities for five Fahrenheit bands (below 40, 40 to below 55, 55 to below 70, 70 to below 85, and 85 or above). The target is an hourly-observation maximum and can differ from the official daily maximum or a settlement value.

## Run locally

Use Python 3.13. Start in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python weather.py
python reporting.py
python -m unittest discover -s tests -v
streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` and launch the app with `.\.venv\Scripts\python.exe -m streamlit run app.py`.

The first training run downloads 12 official NOAA files. Later runs reuse the original CSVs. Run `python weather.py --refresh` to replace them with the current S3 copies. Run `python reporting.py` again after retraining. `data/raw/manifest.json` identifies the exact inputs used for each run.

## Data and experiment

- The archive contains NOAA Global Hourly CSVs for LGA, JFK, EWR, and TEB. The pipeline checks required columns, keeps FM-15 and FM-16, applies quality codes, and retains the last report per UTC hour for the exported observations and daily labels. [NOAA Global Hourly](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) · [ISD format specification](https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf)
- LGA target dates need at least 20 valid hourly temperatures and coverage in all four local six-hour blocks. Every experiment uses the same eligible dates. The split is 2023 for training, 2024 for validation, and available eligible 2025 dates for testing.
- A station contributes latest temperature, relative humidity, and wind, maximum observed so far, approximately three-hour temperature change, and previous-day observed maximum. The model also gets the month. The current day is truncated at the local cutoff before hourly selection, and the previous-day feature uses the prior calendar date.
- Climatology, Logistic Regression, Random Forest, and XGBoost are evaluated for LGA alone and all four stations at each cutoff. Models and preprocessing fit on 2023. Per-cutoff selection uses 2024 log loss among the non-baseline candidates. The saved models retain their training-only fit.

## Files

| Location | Contents |
| --- | --- |
| `weather.py` | Download, cleaning, labels, features, training, and evaluation |
| `reporting.py` | Figures, Markdown report, Word report, and the noon results table |
| `app.py` | Historical replay and manual live prediction |
| `tests/test_weather.py` | Cutoff leakage, daylight saving, label coverage, staleness, probabilities, and NOAA parsing |
| `data/raw/` | Original station-year CSVs and the SHA-256 manifest |
| `data/processed/` | Hourly tables, data quality statistics, eligible labels, and a sample table for each cutoff |
| `outputs/metrics.csv` | 48 validation and test metric rows |
| `outputs/selected_models.json` and `outputs/model_*.joblib` | Validation selections and fitted artifacts |
| `outputs/selected_test_predictions.csv` and `outputs/prediction_errors.csv` | Probabilities and misclassifications from the selected models |
| `outputs/noon_lga_results.csv` | Validation and test metrics for LGA alone at noon |
| `outputs/*.png` | Category counts and test log-loss figures |
| `outputs/Progress1_Report.md` and `outputs/Progress1_Report.docx` | Progress 1 reports generated from the actual run |

## Demo

Historical replay shows the observed high alongside the original three-cutoff model's distribution. Latest observations refreshes automatically while open, using the latest scheduled New York cutoff, four airports' METAR reports, HRRR and GOES-East. The page shows observation and publication times and the input set used. Fusion backtest displays the frozen system's saved out-of-sample probabilities. The [nowcast guide](nowcast/README.md) includes the independent local training command and measured results. [Aviation Weather Center Data API](https://aviationweather.gov/data/api/)
