"""Checks for class_eval.py: time-ordered folds, the learning-curve subsets and the CURRENT reproduction."""

import unittest

import numpy as np
import pandas as pd

from class_eval import CV_YEARS, FIRST, REFERENCE, TOLERANCE, fit_current, most_recent, rolling_folds
from experiment import HOURS, split_dates
from history_study import load
from prepare import DATA


class RollingFolds(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2010-01-01", "2026-08-31", freq="D")
        rng = np.random.default_rng(0)
        self.eligible = rng.random(len(self.index)) > 0.2  # some dates missing, as with incomplete sources

    def test_no_fold_trains_on_its_own_or_a_later_year(self):
        for fold in rolling_folds(self.index, self.eligible):
            train, validation = self.index[fold["train"]], self.index[fold["validation"]]
            self.assertTrue(len(train) and len(validation))
            self.assertTrue((train.year < fold["year"]).all())
            self.assertTrue((validation.year == fold["year"]).all())
            self.assertLess(train.max(), validation.min())
            self.assertFalse((fold["train"] & fold["validation"]).any())
            self.assertGreaterEqual(train.min(), pd.Timestamp(FIRST))

    def test_folds_never_touch_the_test_period_or_the_2025_gap(self):
        for fold in rolling_folds(self.index, self.eligible):
            used = self.index[fold["train"] | fold["validation"]]
            self.assertLess(used.max(), pd.Timestamp("2025-01-01"))
        with self.assertRaises(ValueError):
            rolling_folds(self.index, self.eligible, years=(2025,))

    def test_folds_grow_forward_and_the_last_is_the_single_split(self):
        folds = rolling_folds(self.index, self.eligible)
        self.assertEqual([f["year"] for f in folds], list(CV_YEARS))
        for earlier, later in zip(folds, folds[1:]):
            self.assertTrue((earlier["train"] <= later["train"]).all())       # training set only grows
            self.assertTrue((earlier["validation"] <= later["train"]).all())  # last year's validation joins training
        splits = split_dates(self.index)
        start = self.index >= FIRST
        np.testing.assert_array_equal(folds[-1]["train"], splits["train"] & self.eligible & start)
        np.testing.assert_array_equal(folds[-1]["validation"], splits["validation"] & self.eligible)

    def test_only_eligible_rows_are_used(self):
        for fold in rolling_folds(self.index, self.eligible):
            self.assertFalse((fold["train"] & ~self.eligible).any())
            self.assertFalse((fold["validation"] & ~self.eligible).any())


class MostRecent(unittest.TestCase):
    def test_takes_the_latest_n_eligible_days(self):
        index = pd.date_range("2021-01-01", "2023-12-31", freq="D")
        mask = np.asarray(index.day % 3 != 0)
        for n in (1, 100, 400):
            subset = most_recent(mask, n)
            self.assertEqual(subset.sum(), n)
            self.assertFalse((subset & ~mask).any())
            self.assertEqual(index[subset].max(), index[mask].max())
            dropped = index[mask & ~subset]
            self.assertTrue(len(dropped) == 0 or dropped.max() < index[subset].min())
        np.testing.assert_array_equal(most_recent(mask, None), mask)
        np.testing.assert_array_equal(most_recent(mask, int(mask.sum()) + 50), mask)


class CurrentReproduction(unittest.TestCase):
    def test_current_matches_the_quoted_test_log_loss(self):
        hrrr = pd.read_csv(DATA / "hrrr_features.csv.gz", parse_dates=["date"])
        guidance = pd.read_csv(DATA / "guidance_features.csv.gz", parse_dates=["date"])
        for hour in HOURS:
            samples, complete = load(DATA, hour, hrrr, guidance)
            current = fit_current(samples, complete)
            self.assertEqual(int((split_dates(samples.index)["test"] & complete).sum()), 359)
            self.assertLessEqual(abs(current["test_log_loss"] - REFERENCE[hour]), TOLERANCE, hour)


if __name__ == "__main__":
    unittest.main()
