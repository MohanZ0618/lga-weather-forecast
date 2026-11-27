## Longer history and a skewed distribution

`history.py` downloads LCD v2 for 2010 to 2018 (gitignored) and builds 2010 to 2017 samples with `prepare.py`'s own functions into `data/history/`. The rebuilt 2018 rows equal the bundled samples in every cell (`data/history/overlap_check_2018.csv`, 100% of cells at all three cutoffs). `history_study.py` then tests two ideas on the same 359 test days, choosing on 2024 validation.

Longer training windows with LAMP, NBM and HRRR imputed before they existed change the censored Gaussian's test log loss from 0.182, 0.136 and 0.070 (2021 to 2023, complete rows) to 0.231, 0.151 and 0.073 (2018 to 2023) and 0.253, 0.161 and 0.074 (2010 to 2023). Where inputs exist in every year (observations only), 2010 to 2023 instead of 2018 to 2023 moves the censored Gaussian from 0.342, 0.190 and 0.073 to 0.342, 0.193 and 0.072 and quantile boosting from 0.327, 0.193 and 0.072 to 0.312, 0.198 and 0.075.

A sinh-arcsinh shape (`shash.py`, Jones and Pewsey 2009) adds skew and tail weight to the censored Gaussian. Its test log loss is 0.183, 0.139 and 0.067 against 0.182, 0.136 and 0.070. `test_history.py` checks the SHASH gradient against finite differences, parameter recovery on simulated data and the 2018 overlap.
