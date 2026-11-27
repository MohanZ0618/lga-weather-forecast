## Station guidance: LAMP and NBM

`guidance_study.py` adds station guidance from the Iowa Environmental Mesonet MOS archive: LAMP (four runs a day from July 2020) and the NBM text bulletin (four runs a day from November 2018). `guidance.py fetch` downloads the yearly KLGA files (URLs and SHA-256 in `data/guidance_sources.csv`) and `guidance.py build` writes `data/guidance_features.csv.gz`. Each cutoff uses the newest run released by then, assuming release 55 minutes (LAMP) and 66 minutes (NBM) after initialization. The seven features are LAMP's highest temperature for the rest of the day, its error at the current hour, its afternoon cloud cover and two wind components, and NBM's daytime maximum and standard deviation.

Six candidates were compared on 2024 validation log loss (the paper's classifier families, the censored Gaussian and quantile boosting, each with and without HRRR). On the paper's 359 matched test days:

| Cutoff | Chosen on 2024 | Paper log loss | New log loss | Difference [95% interval] | Accuracy |
| --- | --- | ---: | ---: | --- | --- |
| 09:00 | Censored Gaussian | 0.265 | 0.182 | -0.083 [-0.117, -0.050] | 89.7% to 92.5% |
| 12:00 | Censored Gaussian | 0.205 | 0.137 | -0.068 [-0.118, -0.027] | 93.3% to 94.7% |
| 15:00 | Quantile boosting | 0.084 | 0.061 | -0.023 [-0.046, -0.001] | 96.1% to 97.2% |

No feature uses a run before its assumed release (`test_guidance.py`), and the full-year downloads match an independent earlier download byte for byte. The archive has no receipt times, so the release delays are assumptions.
