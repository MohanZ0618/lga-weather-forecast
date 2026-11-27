"""Checks for pooled_airports.py: rebuilt LGA rows, guidance timing at every airport, re-centering, the
weighted censored Gaussian and the evaluation protocol. Runs offline on data/pooled/ and the results."""

import unittest

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

import pooled_airports as pa
import shash
from distribution import add_relative, relative_columns
from experiment import season_features
from prepare import DATA, local_cutoff

RESULTS = pa.OUT


def simulated(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    day = rng.uniform(0, 2 * np.pi, n)
    z = np.column_stack([np.ones(n), np.sin(day), np.cos(day)])
    rise = np.maximum(1.0 + x @ np.array([1.5, -0.5, 0.3]) + np.exp(0.5 + 0.2 * z[:, 1]) * rng.normal(size=n), 0.0)
    return x, z, rise


class RebuiltInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stations, cls.labels, cls.guidance = pa.load_pooled()

    def test_rebuilt_lga_rows_equal_the_bundled_samples(self):
        check = pd.read_csv(pa.POOLED / "overlap_check_lga.csv")
        self.assertEqual(sorted(check.hour), list(pa.HOURS))
        self.assertTrue(check.same_dates.all() and check.same_columns.all())
        self.assertTrue((check.cells_equal == 1.0).all())

    def test_rebuilt_klga_guidance_equals_the_bundled_table(self):
        check = pd.read_csv(pa.POOLED / "overlap_check_lga.csv")
        self.assertTrue(check.klga_text_identical.all())
        self.assertTrue((check.klga_cells_equal == 1.0).all())

    def test_mos_files_hold_the_requested_station(self):
        sources = pd.read_csv(pa.POOLED / "mos_sources.csv")
        self.assertEqual(set(sources.station), {pa.ICAO[a] for a in pa.AIRPORTS})
        self.assertTrue((sources.stations_in_file == sources.station).all())

    def test_no_guidance_run_is_used_before_its_assumed_release_at_any_airport(self):
        table = pd.read_csv(pa.POOLED / "guidance.csv.gz", parse_dates=["date", "lamp_run", "nbm_run"])
        cutoff = pd.Series([local_cutoff(d, h).tz_convert("UTC") for d, h in zip(table.date, table.hour)])
        for airport in pa.AIRPORTS:
            mine = table.airport.eq(airport)
            self.assertGreater(table.loc[mine, "lamp_remaining_max"].notna().mean(), 0.9, airport)
            self.assertGreater(table.loc[mine, "nbm_max"].notna().mean(), 0.9, airport)
            self.assertFalse((table.lamp_run[mine] + pd.Timedelta(minutes=55) > cutoff[mine]).any(), airport)
            self.assertFalse((table.nbm_run[mine] + pd.Timedelta(minutes=66) > cutoff[mine]).any(), airport)

    def test_neighbours_are_ordered_by_distance(self):
        self.assertEqual(pa.neighbours("LGA"), ["JFK", "TEB", "EWR"])
        self.assertEqual(pa.neighbours("EWR"), ["TEB", "LGA", "JFK"])

    def test_lga_frame_is_the_current_without_hrrr_input_set(self):
        bundled_guidance = pd.read_csv(DATA / "guidance_features.csv.gz", parse_dates=["date"])
        for hour in pa.HOURS:
            frame = pa.target_frame("LGA", hour, self.stations, self.labels, self.guidance)
            samples = pd.read_csv(DATA / f"samples_{hour:02}.csv.gz", parse_dates=["date"], index_col="date")
            samples = samples.loc[frame.index]
            mos = bundled_guidance.loc[bundled_guidance.hour.eq(hour)].set_index("date")[pa.REQUIRED]
            samples = add_relative(season_features(samples.join(mos, how="left")))
            rename = {"own": "LGA", "n1": "JFK", "n2": "TEB", "n3": "EWR"}
            mapped = [rename[c.split("_")[0]] + c[c.index("_"):] if c.split("_")[0] in rename else c for c in pa.COLUMNS]
            current = relative_columns(samples, "Four stations + trends + LAMP/NBM")
            self.assertEqual(sorted(mapped), sorted(current))
            np.testing.assert_allclose(frame[pa.COLUMNS].to_numpy(float), samples[mapped].to_numpy(float),
                                       rtol=0, atol=1e-9, equal_nan=True)
            np.testing.assert_array_equal(frame.target_max, samples.target_max)

    def test_other_targets_use_their_own_labels_observations_and_guidance(self):
        for hour in pa.HOURS:
            for airport in pa.AIRPORTS[1:]:
                frame = pa.target_frame(airport, hour, self.stations, self.labels, self.guidance)
                label = self.labels.loc[self.labels.airport.eq(airport)].set_index("date")
                np.testing.assert_array_equal(frame.target_max, label.loc[frame.index, "target_max"])
                table = self.stations[hour].loc[frame.index]
                np.testing.assert_array_equal(frame.own_temp, table[f"{airport}_temp"])
                np.testing.assert_array_equal(frame.n1_max_so_far, table[f"{pa.neighbours(airport)[0]}_max_so_far"])
                mos = self.guidance.loc[self.guidance.airport.eq(airport) & self.guidance.hour.eq(hour)].set_index("date")
                np.testing.assert_array_equal(frame.nbm_max, mos.nbm_max.reindex(frame.index))
                rise = (frame.target_max - frame.own_max_so_far).dropna()
                self.assertTrue((rise >= 0).all())  # the closed-hour high never exceeds the day's high
                self.assertEqual(frame[f"is_{airport}"].unique().tolist(), [1.0])

    def test_training_validation_and_test_rows_follow_the_protocol(self):
        hrrr = pd.read_csv(DATA / "hrrr_features.csv.gz", parse_dates=["date"])
        rows, _ = pa.assemble(9, self.stations, self.labels, self.guidance, pd.DatetimeIndex(hrrr.loc[hrrr.hour.eq(9), "date"]))
        train = rows.role.eq("train")
        self.assertLess(rows.index[train].max(), pd.Timestamp("2024-01-01"))
        self.assertTrue(rows.loc[~train, "airport"].eq("LGA").all())  # only LGA is ever scored
        lga_train = set(rows.index[train & rows.airport.eq("LGA")])
        self.assertTrue(set(rows.index[train & ~rows.airport.eq("LGA")]) <= lga_train)
        test = rows.index[rows.role.eq("test")]
        self.assertTrue(((test >= "2025-09-01") & (test < "2026-09-01")).all())


class WeightedCensoredGaussian(unittest.TestCase):
    def test_unit_weights_give_shash_gaussian_loss_and_gradient(self):
        x, z, rise = simulated()
        theta = np.r_[0.5, 0.2, -0.1, 0.3, 0.4, 0.1, -0.2]
        loss, grad = pa.weighted_loss_and_grad(theta, x, z, rise, 3.0, np.ones(len(rise)))
        full = np.r_[theta, np.zeros(4)]
        ref_loss, ref_grad = shash.loss_and_grad(full, x, z, rise, 3.0)
        self.assertAlmostEqual(loss, ref_loss, places=10)
        np.testing.assert_allclose(grad, ref_grad[:len(theta)], rtol=1e-8, atol=1e-10)

    def test_gradient_matches_finite_differences_with_weights(self):
        x, z, rise = simulated(seed=1)
        weight = np.random.default_rng(2).uniform(0.5, 3, len(rise))
        theta = np.r_[0.8, 0.5, -0.2, 0.1, 0.3, 0.05, 0.1]
        analytic = pa.weighted_loss_and_grad(theta, x, z, rise, 1.0, weight)[1]
        numeric = approx_fprime(theta, lambda t: pa.weighted_loss_and_grad(t, x, z, rise, 1.0, weight)[0], 1e-7)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)

    def test_integer_weight_equals_duplicating_rows(self):
        x, z, rise = simulated(seed=3)
        weight = np.where(np.arange(len(rise)) < 100, 2.0, 1.0)
        doubled = np.r_[np.arange(len(rise)), np.arange(100)]
        weighted = pa.weighted_fit(x, z, rise, 0.0, weight)
        duplicated = shash.fit(x[doubled], z[doubled], rise[doubled], 0.0, shape=False)
        points = np.linspace(0.5, 8, 4)[None, :].repeat(len(rise), axis=0)
        np.testing.assert_allclose(shash.cdf(weighted, x, z, points), shash.cdf(duplicated, x, z, points), atol=1e-4)


class Results(unittest.TestCase):
    def test_scored_on_currents_359_test_days_and_training_dates(self):
        checks = pd.read_csv(RESULTS / "checks.csv")
        self.assertTrue((checks.test_dates == 359).all())
        self.assertTrue(checks.same_test_dates_as_current.all() and checks.current_targets_match.all())
        self.assertTrue((checks.lga_train_dates == checks.lga_train_dates_guidance_study).all())

    def test_selection_is_the_validation_minimum(self):
        selection = pd.read_csv(RESULTS / "selection.csv")
        arms = [f"val_{name}" for name in pa.ARMS.values()]
        for _, row in selection.iterrows():
            self.assertEqual(row.selected_without_hrrr, row[arms].astype(float).idxmin()[4:])

    def test_weight_one_reproduces_the_shifted_pool(self):
        checks = pd.read_csv(RESULTS / "checks.csv")
        self.assertTrue((checks.weighted_fit_at_weight_1_vs_c_val_gap.abs() < 1e-4).all())


if __name__ == "__main__":
    unittest.main()
