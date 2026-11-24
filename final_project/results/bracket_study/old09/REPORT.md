# How much does the forecast improve later in the day?

[English](REPORT.md) · [简体中文](REPORT.zh-CN.md)

At 09:00, 12:00 and 15:00 New York time, the models forecast which listed Polymarket bracket will contain LaGuardia's final high temperature. The boundaries can change each day. A forecast scores one hit if its highest-probability bracket matches the official settlement. The 413 dates with all three forecasts run from September 3, 2025 through November 6, 2026.

| When the forecast is used | Correct / forecasts | Top-pick accuracy |
| --- | ---: | ---: |
| Once at 09:00 | 202 / 413 | 48.9% |
| Once at 12:00 | 241 / 413 | 58.4% |
| Once at 15:00 | 303 / 413 | 73.4% |
| All three issues, equally weighted | 746 / 1,239 | 60.2% |

The afternoon forecast corrected 138 morning misses and lost 37 morning hits, for a net gain of 101 days. As the day progresses, the forecast can use a higher observed temperature floor, recent warming trends and newer weather guidance. On these same dates, a baseline based on historical temperatures for the time of year picked the winner on 26 days (6.3%). Using the day's listed intervals and the historical positions of winning brackets produced 73 hits (17.7%).

## Why use separate models?

Model family and weather sources were selected using January–April 2025 validation log loss on actual market brackets. Hyperparameters were then tuned on May–August validation. The morning blend estimates the chance of crossing each temperature threshold in two ways. Logistic regression estimates that chance directly. The second model first estimates whether the temperature will rise above its observed high, then uses CatBoost to estimate the remaining rise. At noon and 15:00 separately trained XGBoost models constrain the probability of exceeding a threshold to decrease or stay constant as the threshold rises. Each system uses four airports, HRRR and National Blend of Models guidance when available, with logistic regression at lower input tiers.

| Forecast | Selected hyperparameters |
| --- | --- |
| 09:00 | Equal blend, logistic C = 4.0713, CatBoost depth 6, 333 trees, learning rate 0.09070, L2 = 1.3988 |
| 12:00 | Monotonic XGBoost, depth 6, 426 trees, learning rate 0.01731, L2 = 3.5926 |
| 15:00 | Monotonic XGBoost, depth 6, 939 trees, learning rate 0.01501, L2 = 4.9948 |

To isolate the effect of sharing training data, we also fit two versions of the morning architecture with its original hyperparameters. Both used the same weather features, forecast-hour indicators, settlement definitions and training cutoff. One fit each hour separately. The other pooled observations across hours, sharing the fitted weights within each settlement definition and weather-input tier.

| System | 09:00 | 12:00 | 15:00 | Overall hits and accuracy | Mean log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected models by forecast time | 48.9% | 58.4% | 73.4% | 746/1,239 (60.2%) | 0.9584 |
| Morning architecture fitted separately | 48.9% | 56.2% | 73.8% | 739/1,239 (59.6%) | 0.9535 |
| Morning architecture fitted jointly | 49.6% | 54.0% | 72.6% | 728/1,239 (58.8%) | 0.9571 |

With the architecture held fixed, pooling changed accuracy by −0.89 percentage points (95% paired interval −2.68 to +0.81). We averaged the three issues per date, then resampled seven-calendar-day blocks 5,000 times. The pooled model itself rose from 49.6% at 09:00 to 72.6% at 15:00. Later updates helped even with shared weights. The observed top-pick scores favor keeping the selected system, although the size of the benefit from separate fitting is uncertain. [Forecasts](time_probabilities.csv.gz) · [Daily scores](time_daily.csv.gz) · [Summary](time_metrics.csv) · [Paired intervals](time_comparison.csv)

## Why retain the original morning blend?

The morning comparison includes 430 dates with forecasts from both models, 17 more than the three-time comparison. The challenger averaged ridge regression (α = 40) and shallow gradient boosting (7 leaves, 140 iterations, learning rate 0.045, L2 = 5) to correct errors in the weather forecast. It converted that estimate to bracket probabilities using historical forecast errors, included more detailed morning observations and updated monthly with a seven-day gap. The original blend won **216/430 (50.2%)**, the challenger **212/430 (49.3%)**. The 95% paired interval for the challenger's change in accuracy ran from −5.9 to +4.1 percentage points. We kept the original 09:00 model. [Morning comparison by period](../morning/metrics.csv)

Holding the fitted morning model fixed, we tested 36 adjustments: three weights on the threshold model (25%, 50%, 75%), three temperature shifts (−0.5, 0, +0.5°F), an optional whole-degree frequency adjustment, and an optional floor using observations through 08:55. Routine METAR contracts always used the routine-report floor. We treated a result as available 48 hours after the last bracket closed and used only outcomes available before March 1, 2026 to select settings. Eligible settings had to preserve the original hit count in each of two calibration periods and keep the increase in mean log loss within 0.02. We then ranked by total hits, lower log loss and fewer changes.

On 169 calibration days, a 75% threshold-model weight and the newer morning floor improved hits from 88 to 93. Temperature shift and whole-degree adjustment were both zero. We applied those fixed settings to 258 later days.

| Subsequent period in 2026 | Original hits | Adjusted hits | Original / adjusted log loss |
| --- | ---: | ---: | ---: |
| March 1 – August 31 | 78/176 | 83/176 | 1.348 / 1.329 |
| September 1 – November 6 | 37/67 | 31/67 | 1.134 / 1.132 |
| November 7–21 | 12/15 | 12/15 | 1.010 / 1.010 |
| Overall | **127/258 (49.2%)** | **126/258 (48.8%)** | **1.273 / 1.259** |

The adjustment corrected six picks and lost seven, changing accuracy by −0.39 percentage points (95% paired interval −3.46 to +2.73). It modestly improved the probability assigned to the winner on average. For the goal of choosing one bracket each morning, the replay retains the original 50:50 blend. [All 36 settings](validation.csv) · [Locked candidate](candidate.json) · [Scores by period](metrics.csv) · [Active setting](active.json)

## Timing and scope

- The selected models and the pooled comparison trained on weather labels before September 1, 2025. The forecasts use the weather available at each issue time and the market's listed intervals. Afternoon observations cannot update an earlier pick.
- Across the 258 follow-up days, the recomputed original probabilities matched the archived values within 0.00000001 per bracket. The adjustment and pooling experiments exported their forecasts before joining official winners for scoring.
- These are historical reconstructions. Station archives lack individual receipt timestamps and contracts lack a complete historical record of rule edits. The last 15 mornings use supplemental IEM observations and are excluded from the three-time comparison. Previous studies had already examined these dates. Establishing prospective accuracy requires new forecasts recorded before settlement.
