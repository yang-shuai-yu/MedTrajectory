from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.calibration_auc import (  # noqa: E402
    DAYS_PER_YEAR,
    age_stratified_delong_rows,
    binary_auc,
    build_horizon_case_control,
    compute_midrank,
    extract_official_lm_records,
    get_auc_delong_var,
    precompute_official_prediction_indices,
    select_age_landmarks,
    select_shared_age_landmarks,
)
from semantic_delphi_ukb.evaluate_calibration_auc import load_followup_end_ages  # noqa: E402


class CalibrationAucTest(unittest.TestCase):
    def test_midrank_and_auc_handle_ties(self):
        np.testing.assert_allclose(compute_midrank(np.asarray([1.0, 2.0, 2.0, 4.0])), [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(binary_auc(np.asarray([0.0, 0.5]), np.asarray([0.5, 1.0])), 0.875)

    def test_delong_reports_perfect_separation(self):
        auc, variance = get_auc_delong_var(np.asarray([0.1, 0.2, 0.3]), np.asarray([0.7, 0.8, 0.9]))
        self.assertEqual(auc, 1.0)
        self.assertEqual(variance, 0.0)

    def test_official_records_filter_missing_prediction_position(self):
        targets = np.asarray([[7, 2, 3], [2, 3, 4], [7, 3, 4]])
        input_ages = np.asarray([[50.0, 60.0, 70.0], [40.0, 50.0, 60.0], [55.0, 65.0, 75.0]])
        target_ages = np.asarray([[50.0, 70.0, 80.0], [50.0, 60.0, 70.0], [65.0, 75.0, 85.0]])
        scores = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
        prediction_indices = precompute_official_prediction_indices(input_ages, target_ages, 0.1)
        records = extract_official_lm_records(
            targets,
            input_ages,
            target_ages,
            scores,
            token_id=7,
            patient_ids=np.asarray([10, 11, 12]),
            prediction_indices=prediction_indices,
        )
        self.assertEqual(int(records["labels"].sum()), 1)
        self.assertNotIn(-10000.0, records["prediction_age_days"].tolist())
        self.assertTrue(set(records["patient_ids"].tolist()).issubset({10, 11, 12}))

    def test_age_stratification_keeps_one_record_per_patient(self):
        rows = age_stratified_delong_rows(
            scores=np.asarray([0.8, 0.9, 0.1, 0.2]),
            labels=np.asarray([1, 1, 0, 0]),
            prediction_age_days=np.asarray([51, 52, 51, 52]) * DAYS_PER_YEAR,
            patient_ids=np.asarray([1, 1, 2, 3]),
            age_groups=[50, 55],
            rng=np.random.default_rng(4),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_cases"], 1)
        self.assertEqual(rows[0]["n_controls"], 2)
        self.assertEqual(rows[0]["auc"], 1.0)

    def test_horizon_controls_require_complete_followup(self):
        patient_disease_ages = [
            [np.asarray([58 * DAYS_PER_YEAR])],
            [np.asarray([], dtype=np.float64)],
            [np.asarray([], dtype=np.float64)],
        ]
        labels, eligible = build_horizon_case_control(
            patient_ids=np.asarray([0, 1, 2]),
            prediction_age_days=np.asarray([55, 55, 55]) * DAYS_PER_YEAR,
            patient_disease_ages=patient_disease_ages,
            patient_last_ages=np.asarray([60, 57, 65]) * DAYS_PER_YEAR,
            horizon_years=5,
            disease_idx=0,
        )
        np.testing.assert_array_equal(labels, [1, 0, 0])
        np.testing.assert_array_equal(eligible, [True, False, True])

    def test_evaluator_prefers_prepared_followup_end_ages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            np.save(data_dir / "val_followup_end_age_days.npy", np.asarray([60.0, 70.0], dtype=np.float32))
            observed_last_ages = np.asarray([50.0, 55.0], dtype=np.float32)
            loaded = load_followup_end_ages(data_dir, "val", observed_last_ages, max_patients=0)
            np.testing.assert_array_equal(loaded, [60.0, 70.0])

    def test_landmarks_are_limited_to_one_per_patient_age_bin(self):
        tokens = np.asarray([[2, 3, 4, 0], [2, 3, 0, 0]])
        ages = np.asarray([[41, 42, 47, -10000], [44, 46, -10000, -10000]], dtype=np.float64) * DAYS_PER_YEAR
        landmarks = select_age_landmarks(tokens, ages, np.asarray([0, 1]), [40, 45, 50], np.random.default_rng(3))
        observed = list(zip(landmarks["patient_ids"].tolist(), landmarks["age_start_years"].tolist()))
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), {(0, 40.0), (0, 45.0), (1, 40.0), (1, 45.0)})

    def test_shared_landmarks_use_frozen_target_age_and_causal_position(self):
        tokens = np.asarray([[2, 3, 4, 0], [2, 3, 0, 0]])
        ages = np.asarray([[41, 42, 47, -10000], [44, 46, -10000, -10000]], dtype=np.float64) * DAYS_PER_YEAR
        selected = select_shared_age_landmarks(
            tokens,
            ages,
            np.asarray([0, 1]),
            {0: [{"age_start_years": 40.0, "target_age_days": 46 * DAYS_PER_YEAR}], 1: [{"age_start_years": 40.0, "target_age_days": 45 * DAYS_PER_YEAR}]},
        )
        np.testing.assert_array_equal(selected["patient_ids"], [0, 1])
        np.testing.assert_allclose(selected["prediction_age_days"], np.asarray([46, 45]) * DAYS_PER_YEAR, atol=1e-6)
        np.testing.assert_allclose(selected["position_age_days"], np.asarray([42, 44]) * DAYS_PER_YEAR, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
