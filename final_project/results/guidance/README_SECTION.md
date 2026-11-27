## Station guidance: LAMP and NBM

`python final_project/guidance_study.py` (about 40 seconds, offline) adds station guidance from the Iowa Environmental Mesonet MOS archive: LAMP (four runs a day, from July 2020) and the NBM text bulletin (four runs a day, from November 2018). `guidance.py fetch` downloads the 16 yearly KLGA files (URLs and SHA-256 in `data/guidance_sources.csv`); `guidance.py build` writes the bundled `data/guidance_features.csv.gz`. At each cutoff it uses the newest run issued by then, assuming release 55 min (LAMP) and 66 min (NBM) after initialization. Seven features: LAMP's highest temperature for the rest of the day, LAMP's error at the current hour, LAMP's afternoon cloud cover and wind (east and north components), and NBM's daytime maximum and its standard deviation.

The system is chosen per cutoff on 2024 validation log loss among six candidates (paper classifier families, censored Gaussian and quantile boosting, each with and without HRRR). The censored Gaussian with HRRR plus LAMP/NBM wins at all three cutoffs. On the same 359 test days as the paper's selected HRRR system:

| Cutoff | Paper log loss | With LAMP/NBM | Difference [95% block interval] | Accuracy |
| --- | ---: | ---: | --- | --- |
| 09:00 | 0.258 | 0.200 | −0.058 [−0.095, −0.023] | 88.9% → 90.8% |
| 12:00 | 0.234 | 0.159 | −0.075 [−0.112, −0.040] | 89.4% → 93.6% |
| 15:00 | 0.094 | 0.072 | −0.022 [−0.045, +0.0003] | 97.2% → 97.8% |

Checks: no feature uses a run before its assumed release (`test_guidance.py`); the gain survives adding one hour to every release delay (−0.056, −0.075, −0.024, all intervals below zero); the downloaded full-year files match an independent earlier download byte for byte. The archive has no receipt times, so release delays are assumptions, not measurements. LAMP alone, without HRRR, does about as well (`results/guidance/test_scores.csv`).
