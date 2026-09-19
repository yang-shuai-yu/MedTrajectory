from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
for path in (REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_paper_protocol_longitudinal import (  # noqa: E402
    build_cutoff_landmark_batch,
    cutoff_landmark_positions,
    load_outcomes,
)
from semantic_delphi_ukb import evaluate_paper_protocol_medical_auc  # noqa: E402
from semantic_delphi_ukb.paper_run import TwoStageRiskScheduler  # noqa: E402
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import load_token_codes  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import load_sex_mapping  # noqa: E402
from training_monitor import RunMonitor  # noqa: E402


class PaperTrainingContractTests(unittest.TestCase):
    def test_token_vocab_path_is_relative_to_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            vocab_dir = data_dir / "vocab"
            vocab_dir.mkdir()
            (data_dir / "prepare_manifest.json").write_text(
                json.dumps({"vocab_csv": "vocab/dynamic_token_vocab.csv"}),
                encoding="utf-8",
            )
            with (vocab_dir / "dynamic_token_vocab.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["token_id", "event_type", "code_norm"])
                writer.writeheader()
                writer.writerow({"token_id": 7, "event_type": "diagnosis", "code_norm": "i10"})
                writer.writerow({"token_id": 8, "event_type": "medication", "code_norm": "ignored"})

            self.assertEqual(load_token_codes(data_dir), {7: "I10"})

    def test_sex_mapping_falls_back_for_sex_only_paper_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            vocab_dir = data_dir / "vocab"
            vocab_dir.mkdir()
            (data_dir / "prepare_manifest.json").write_text(
                json.dumps({"static_feature_order": ["sex_id"]}),
                encoding="utf-8",
            )
            vocab_csv = vocab_dir / "dynamic_token_vocab.csv"
            vocab_csv.write_text("token_id,event_type,code_norm\n0,padding,\n", encoding="utf-8")

            self.assertEqual(load_sex_mapping(vocab_csv), {"female": 0, "male": 1})

    def test_medical_wrapper_locks_paper_defaults(self):
        observed = {}

        def fake_main(arguments):
            observed["arguments"] = arguments
            return 0

        with mock.patch.object(evaluate_paper_protocol_medical_auc, "evaluate_main", fake_main):
            self.assertEqual(evaluate_paper_protocol_medical_auc.main(["--checkpoint", "model.pt"]), 0)
        self.assertEqual(observed["arguments"][-4:], ["--split", "val", "--age-groups", "50,55,60,65,70,75"])

    def test_protocol_config_locks_required_repairs(self):
        common = json.loads((REPO_DIR / "configs/paper_protocol_v1/common.json").read_text(encoding="utf-8"))
        self.assertEqual(common["split_seed"], 1337)
        self.assertEqual(common["training_seeds"], [42, 43, 44])
        self.assertEqual(common["pretraining"]["encoder_family"], "causal_delphi")
        self.assertEqual(common["pretraining"]["objective"], "all_position_next_event_and_time")
        self.assertEqual(common["posttraining"]["horizons_years"], [1, 5, 10])
        self.assertEqual(common["posttraining"]["next_event_loss_weight"], 0.2)
        self.assertTrue(common["posttraining"]["monotonic_horizon_risk"])
        self.assertEqual(common["survival_horizon"]["enabled_experiments"], ["P0", "P3"])
        for key in ("P0", "P1", "P2", "P3", "P4"):
            experiment = json.loads((REPO_DIR / f"configs/paper_protocol_v1/{key}.json").read_text(encoding="utf-8"))
            self.assertEqual(experiment["encoder"], "causal_delphi")

    def test_monotonic_parameterization_has_gradients_for_every_horizon(self):
        config = HorizonRiskConfig(num_horizons=3, num_tte_tasks=2, monotonic_horizon_risk=True)
        model = object.__new__(HorizonRiskMedTrajectory)
        model.config = config
        raw = torch.randn(2, 4, 3, 2, requires_grad=True)
        logits = HorizonRiskMedTrajectory._horizon_risk_logits(model, raw)
        self.assertTrue(bool(torch.all(logits[:, :, 1:] > logits[:, :, :-1])))
        logits.sum().backward()
        self.assertTrue(bool(torch.all(raw.grad.abs().sum(dim=(0, 1, 3)) > 0)))

    def test_two_stage_scheduler_freezes_then_uses_lower_trunk_lr(self):
        trunk = torch.nn.Parameter(torch.ones(1))
        head = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([{"params": [trunk]}, {"params": [head]}])
        scheduler = TwoStageRiskScheduler(optimizer, 5, 20, 3e-5, 3e-4, 5e-4)
        scheduler.step(0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        self.assertEqual(optimizer.param_groups[1]["lr"], 5e-4)
        scheduler.step(5)
        self.assertLess(optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])

    def test_longitudinal_outcomes_ignore_rows_beyond_smoke_subset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "outcomes.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_index", "token_id"])
                writer.writerows([[0, 10], [3, 10], [1, 99]])
            outcomes = load_outcomes(path, [10], patient_count=2)
            np.testing.assert_array_equal(outcomes[:, 0], [1, 0])

    def test_longitudinal_scores_at_explicit_cutoff_landmark(self):
        data = np.asarray(
            [
                [10, 100, 4],
                [10, 200, 7],
                [10, 300, 0],
                [11, 150, 5],
                [11, 350, 0],
            ],
            dtype=np.uint32,
        )
        p2i = np.asarray([[0, 3], [3, 2]], dtype=np.int64)
        static = np.asarray([[0.0], [1.0]], dtype=np.float32)
        patient_ids = np.asarray([0, 1], dtype=np.int64)
        x, age, _ = build_cutoff_landmark_batch(patient_ids, data, p2i, static, block_size=3, device="cpu")
        keep, positions = cutoff_landmark_positions(
            x,
            age,
            np.asarray([300.0, 350.0], dtype=np.float32),
            np.asarray([True, False]),
        )
        self.assertEqual(keep.tolist(), [True, False])
        self.assertEqual(positions.tolist(), [2, 2])
        self.assertEqual(int(x[0, positions[0]]), 1)

    def test_monitor_writes_resumable_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = RunMonitor(temp_dir, {"protocol": "paper_protocol_v1"}, enable_tensorboard=False)
            monitor.mark_running(global_step=0)
            state = {
                "model": {}, "optimizer": {}, "scheduler": {}, "scaler": None,
                "iteration": 0, "global_step": 0, "best_val_loss": 1.0,
                "config": {}, "random_state": monitor.capture_random_state(),
            }
            monitor.save_checkpoint(state, "last.pt")
            monitor.save_checkpoint(state, "best_val_loss.pt")
            monitor.log_epoch({"iteration": 0, "global_step": 0, "val_loss": 1.0})
            monitor.close()
            root = Path(temp_dir)
            self.assertTrue((root / "resolved_config.json").exists())
            self.assertTrue((root / "status.json").exists())
            self.assertTrue((root / "metrics.jsonl").exists())
            self.assertTrue((root / "checkpoints/last.pt").exists())
            self.assertTrue((root / "checkpoints/best_val_loss.pt").exists())


if __name__ == "__main__":
    unittest.main()
