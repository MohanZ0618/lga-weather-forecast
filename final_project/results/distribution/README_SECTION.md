## Supplementary checks

`python final_project/supplement.py` runs after `experiment.py` (a few seconds) and writes three tables to `final_project/results/`:

- `supplement_scores.csv` adds the ranked probability score (RPS) for every selected system. The five bands are ordered, so RPS charges more for probability placed several bands from the outcome; log loss and accuracy do not.
- `supplement_paired.csv` gives 95% intervals for the Table 2 differences (log loss, RPS, accuracy), pairing by date and resampling 7-day blocks 5,000 times, because nearby days share weather.
- `supplement_hrrr.csv` scores a raw-HRRR baseline: max(observed high so far, HRRR remaining max + bias) with a Gaussian error, bias and spread fit per season on 2021–2023 only. It asks whether the learned model adds anything over the forecast it consumes.

- `python final_project/distribution.py` (about 40 seconds) tests the continuous forecast named under next steps. It predicts the remaining warming (final high minus the closed-hour high so far, zero on 12% of days at 09:00 and 63% at 15:00) and integrates it onto the five bands. A censored Gaussian regression with a linear mean and a seasonal spread ties the selected classifiers at every cell (log loss within the 95% block-bootstrap interval) and is better on RPS at 12:00 with HRRR. Quantile boosting gives no gain and its 80% intervals cover 74 to 86% of outcomes. The classifiers do not improve when given the same floor-relative features. Outputs are in `final_project/results/distribution/`.

On the committed results, every nearby-station interval includes zero, and trends never help (at 09:00 they slightly worsen RPS). HRRR helps at 09:00 and 12:00 but not at 15:00. The learned HRRR blend beats the raw HRRR baseline at every cutoff.
