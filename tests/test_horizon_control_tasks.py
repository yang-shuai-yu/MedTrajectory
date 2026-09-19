from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.horizon_control_metrics import (  # noqa: E402
    binary_auc,
    calibration_metrics,
    paired_patient_bootstrap,
    summarize_scores,
)
from semantic_delphi_ukb.compare_horizon_control_tasks import (  # noqa: E402
    _compact_bootstrap_macro_values,
    _compact_metrics,
    _load_compact_inputs,
    calibration_table,
    factorial_auc_interaction,
    holm_adjust,
    main as compare_main,
    macro_metrics,
    paired_macro_bootstrap,
)
from semantic_delphi_ukb.evaluate_horizon_control_tasks import load_model  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import load_model as load_calibration_model  # noqa: E402
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_model import SurvivalHorizonConfig, SurvivalHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.evaluate_horizon_control_tasks import disease_scores_from_lm  # noqa: E402
from semantic_delphi_ukb.calibration_auc import select_shared_age_landmarks  # noqa: E402


class HorizonControlTasksTest(unittest.TestCase):
    def test_metrics_and_calibration_are_finite(self):
        values = summarize_scores(np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0, 0, 1, 1]))
        self.assertEqual(values["auc"], 1.0)
        self.assertEqual(values["auprc"], 1.0)
        self.assertTrue(np.isfinite(values["brier"]))
        self.assertTrue(np.isfinite(values["calibration_slope"]))

    def test_patient_bootstrap_uses_paired_rows(self):
        rows = []
        for patient, label in enumerate([0, 0, 1, 1]):
            rows.append({"patient_index": patient, "label": label, "a": 0.2 + 0.2 * label, "b": 0.1 + 0.1 * label})
        result = paired_patient_bootstrap(rows, {"a": "a", "b": "b"}, "a", "b", bootstrap=100, seed=3)
        self.assertEqual(result["patient_count"], 4)
        self.assertGreater(result["delta_auc"], -1e-9)

    def test_lm_group_scores_use_only_requested_tokens(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        scores = disease_scores_from_lm(logits, [[1, 3], [0]])
        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertAlmostEqual(float(scores[0, 0]), float(torch.logsumexp(torch.tensor([2.0, 4.0]), 0)))

    def test_auc_boundaries_and_ties(self):
        self.assertTrue(np.isnan(binary_auc(np.asarray([0.1, 0.2]), np.asarray([1, 1]))))
        self.assertTrue(np.isnan(binary_auc(np.asarray([0.1, 0.2]), np.asarray([0, 0]))))
        self.assertAlmostEqual(binary_auc(np.asarray([0.1, 0.2, 0.2, 0.9]), np.asarray([0, 1, 0, 1])), 0.875)

    def test_lm_probability_metrics_are_explicitly_disabled(self):
        values = calibration_metrics(np.asarray([-3.0, 3.0]), np.asarray([0, 1]), probability_scores=False)
        self.assertTrue(np.isnan(values["brier"]))
        self.assertTrue(np.isnan(values["calibration_slope"]))

    def test_max_aggregation_and_out_of_range_tokens(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        scores = disease_scores_from_lm(logits, [[1, 8], [0]], aggregation="max")
        self.assertEqual(float(scores[0, 0]), 2.0)
        self.assertEqual(float(scores[0, 1]), 1.0)

    def test_paired_comparison_rejects_label_mismatch(self):
        left = [{"patient_index": 1, "disease_id": "D", "horizon_years": 1.0, "label": 1, "score": 0.8}]
        right = [{"patient_index": 1, "disease_id": "D", "horizon_years": 1.0, "label": 0, "score": 0.2}]
        with self.assertRaises(ValueError):
            paired_macro_bootstrap(left, right, bootstrap=10, seed=1)

    def test_paired_comparison_rejects_duplicate_landmark_keys(self):
        rows = [
            {"patient_index": 1, "disease_id": "D", "horizon_years": 1.0, "sex": "female", "age_start_years": 50.0, "label": 1, "score": 0.8},
            {"patient_index": 1, "disease_id": "D", "horizon_years": 1.0, "sex": "female", "age_start_years": 50.0, "label": 1, "score": 0.7},
        ]
        with self.assertRaises(ValueError):
            paired_macro_bootstrap(rows, rows, bootstrap=10, seed=1)

    def test_calibration_table_excludes_lm_scores(self):
        rows = [
            {"patient_index": 1, "disease_id": "D", "horizon_years": 1.0, "sex": "female", "age_start_years": 50.0, "label": 0, "score": -1.0, "score_is_probability": False},
            {"patient_index": 2, "disease_id": "D", "horizon_years": 1.0, "sex": "female", "age_start_years": 50.0, "label": 1, "score": 0.8, "score_is_probability": True},
        ]
        table = calibration_table(rows, bins=10)
        self.assertEqual(sum(row["count"] for row in table), 1)

    def test_macro_metrics_preserves_sex_and_age_cells(self):
        rows = []
        patient = 0
        for sex in ("female", "male"):
            for age in (50.0, 55.0):
                for label, score in ((0, 0.1), (1, 0.9)):
                    rows.append({
                        "patient_index": patient,
                        "disease_id": "D",
                        "horizon_years": 5.0,
                        "sex": sex,
                        "age_start_years": age,
                        "label": label,
                        "score": score,
                        "score_is_probability": True,
                    })
                    patient += 1
        result = macro_metrics(rows)
        self.assertEqual(result["strata"], 4)
        self.assertEqual(result["auc"], 1.0)

    def test_paired_comparison_requires_identical_landmark_rows(self):
        base = {"disease_id": "D", "horizon_years": 1.0, "sex": "female", "age_start_years": 50.0, "label": 0, "score": 0.2}
        left = [{"patient_index": 1, **base}]
        right = [{"patient_index": 2, **base}]
        with self.assertRaises(ValueError):
            paired_macro_bootstrap(left, right, bootstrap=10, seed=1)

    def test_paired_comparison_rejects_prediction_age_mismatch(self):
        base = {"patient_index": 1, "disease_id": "D", "horizon_years": 1.0, "sex": "female", "age_start_years": 50.0, "label": 0, "score": 0.2}
        left = [{**base, "prediction_age_days": 18500.0}]
        right = [{**base, "prediction_age_days": 18501.0}]
        with self.assertRaises(ValueError):
            paired_macro_bootstrap(left, right, bootstrap=10, seed=1)

    def test_holm_adjustment_is_monotone_and_bounded(self):
        adjusted = holm_adjust([0.001, 0.02, 0.5])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertLessEqual(adjusted[0], adjusted[1])

    def test_memory_efficient_comparison_matches_standard_path(self):
        rows_a = []
        rows_b = []
        for age in (50.0, 55.0):
            for patient, label in enumerate((0, 0, 1, 1)):
                base = {
                    "patient_index": patient,
                    "disease_id": "D",
                    "horizon_years": 5.0,
                    "sex": "female",
                    "age_start_years": age,
                    "prediction_age_days": age * 365.25,
                    "label": label,
                    "score_is_probability": True,
                }
                rows_a.append({**base, "score": 0.1 + 0.7 * label})
                rows_b.append({**base, "score": 0.3 + 0.2 * label})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for name, rows in (("A", rows_a), ("B", rows_b)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(rows), encoding="utf-8")
                paths.append(path)
            standard = root / "standard"
            compact = root / "compact"
            common = [
                "--input", f"A={paths[0]}", "--input", f"B={paths[1]}",
                "--bootstrap", "20", "--seed", "7",
            ]
            compare_main([*common, "--out-dir", str(standard)])
            compare_main([*common, "--out-dir", str(compact), "--memory-efficient"])
            standard_payload = json.loads((standard / "comparison.json").read_text(encoding="utf-8"))
            compact_payload = json.loads((compact / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(standard_payload["models"], compact_payload["models"])
            self.assertEqual(standard_payload["paired_bootstrap"], compact_payload["paired_bootstrap"])

    def test_factorial_interaction_uses_only_the_registered_four_cells(self):
        rows = []
        for patient, label in enumerate((0, 0, 1, 1)):
            rows.append({
                "patient_index": patient,
                "disease_id": "D",
                "horizon_years": 5.0,
                "sex": "female",
                "age_start_years": 50.0,
                "prediction_age_days": 50.0 * 365.25,
                "label": label,
                "score_is_probability": True,
            })
        scores = {
            "A1S": (0.9, 0.8, 0.2, 0.1),
            "A0S": (0.1, 0.2, 0.8, 0.9),
            "A1N": (0.1, 0.2, 0.8, 0.9),
            "A0N": (0.1, 0.2, 0.8, 0.9),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            specs = []
            for name, values in scores.items():
                path = Path(temp_dir) / f"{name}.json"
                path.write_text(json.dumps([{**row, "score": values[index]} for index, row in enumerate(rows)]), encoding="utf-8")
                specs.append(f"{name}={path}")
            compact = _load_compact_inputs(specs)
            summaries, _calibration = _compact_metrics(compact, bins=10)
            bootstrap = _compact_bootstrap_macro_values(compact, bootstrap=20, seed=3)
            result = factorial_auc_interaction(compact, summaries, bootstrap, "A1S,A0S,A1N,A0N")
            self.assertLess(result["interaction_auc"], 0.0)
            self.assertEqual(result["common_patients"], 4)

    def test_shared_landmarks_preserve_frozen_position(self):
        landmarks = select_shared_age_landmarks(
            np.asarray([[2, 3, 4]], dtype=np.int64),
            np.asarray([[18000.0, 18000.0, 19000.0]], dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            {0: [{"age_start_years": 49.0, "target_age_days": 18000.0, "position": 0}]},
        )
        self.assertEqual(landmarks["positions"].tolist(), [0])

    def test_linear_mode_loads_survival_trunk_checkpoint(self):
        config = SurvivalHorizonConfig(
            block_size=4, vocab_size=6, n_layer=1, n_head=1, n_embd=8,
            static_dim=1, static_hidden_dim=4, num_tte_tasks=2, num_horizons=3,
            survival_bins=2,
        )
        model = SurvivalHorizonMedTrajectory(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "survival.pt"
            torch.save({"model_args": config.__dict__, "model": model.state_dict()}, path)
            loaded, _ = load_model("linear", path, "cpu")
            self.assertIsInstance(loaded, SurvivalHorizonMedTrajectory)

    def test_explicit_mode_loads_carope_checkpoint(self):
        config = CARoPEConfig(
            block_size=8, vocab_size=32, n_layer=1, n_head=4, n_embd=32,
            static_dim=1, static_hidden_dim=32, num_tte_tasks=2,
            num_horizons=2, horizon_years=(1.0, 5.0),
        )
        model = CARoPEHorizonMedTrajectory(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "car_rope.pt"
            torch.save({"model_args": config.__dict__, "model": model.state_dict(), "model_family": "CARoPE_v1"}, path)
            loaded, _ = load_model("explicit", path, "cpu")
            self.assertIsInstance(loaded, CARoPEHorizonMedTrajectory)

    def test_official_evaluator_loads_carope_checkpoint(self):
        config = CARoPEConfig(
            block_size=8, vocab_size=32, n_layer=1, n_head=4, n_embd=32,
            static_dim=1, static_hidden_dim=32, num_tte_tasks=2,
            num_horizons=3, horizon_years=(1.0, 5.0, 10.0),
        )
        model = CARoPEHorizonMedTrajectory(config)
        checkpoint = {
            "model_args": config.__dict__,
            "model": model.state_dict(),
            "model_family": "CARoPE_v1",
        }
        loaded = load_calibration_model(checkpoint, "cpu")
        self.assertIsInstance(loaded, CARoPEHorizonMedTrajectory)


if __name__ == "__main__":
    unittest.main()
