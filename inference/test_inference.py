"""Behavioral checks using the released trained model and synthetic inputs only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import predict


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle = predict.load_bundle()
        cls.example = predict.read_json(predict.ROOT / "examples/synthetic_history.json")

    def test_real_checkpoint_deterministic_and_valid_trajectory(self):
        left = predict.run(self.example, self.bundle)
        right = predict.run(self.example, self.bundle)
        self.assertEqual(left, right)
        self.assertTrue(left["trained_model"])
        self.assertEqual(left["history_events_used"], 8)
        total = 0
        for trajectory in left["trajectories"]:
            previous = left["prediction_origin_age_days"]
            total += trajectory["event_count"]
            for i, event in enumerate(trajectory["events"]):
                self.assertGreater(event["age_days"], previous)
                self.assertLessEqual(event["age_days"], left["horizon_end_age_days"])
                self.assertEqual(self.bundle[1]["dynamic_labels"][event["token_id"]], event["code"])
                if event["event_type"] == "death":
                    self.assertEqual(i, len(trajectory["events"]) - 1)
                    self.assertEqual(trajectory["stop_reason"], "death")
                previous = event["age_days"]
        self.assertGreater(total, 0)

    def test_invalid_inputs_fail(self):
        cases = []
        for field, value in [("code", "diag:NOT_A_CODE"), ("age_days", float("nan")), ("age_days", True)]:
            example = copy.deepcopy(self.example)
            example["history"][0][field] = value
            cases.append(example)
        example = copy.deepcopy(self.example)
        example["history"] = list(reversed(example["history"]))
        cases.append(example)
        example = copy.deepcopy(self.example)
        example["static"]["anchor_age_days"] = 30000
        cases.append(example)
        example = copy.deepcopy(self.example)
        example["history"][0]["code"] = next(x for x in self.bundle[1]["dynamic_labels"] if x.startswith("death:"))
        cases.append(example)
        example = copy.deepcopy(self.example)
        example["generation"]["temperature"] = 1.2
        cases.append(example)
        example = copy.deepcopy(self.example)
        example["generation"]["num_trajectories"] = 0
        cases.append(example)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict.validate_input(value, self.bundle[1])

    def test_horizon_and_event_limits(self):
        value = copy.deepcopy(self.example)
        value["generation"]["horizon_days"] = 0.1
        result = predict.run(value, self.bundle)
        for trajectory in result["trajectories"]:
            self.assertEqual(trajectory["stop_reason"], "horizon")
            self.assertEqual(trajectory["events"], [])
        value["generation"].update(horizon_days=3652.5, max_new_events=1)
        result = predict.run(value, self.bundle)
        self.assertTrue(all(t["event_count"] <= 1 for t in result["trajectories"]))

    def test_death_stops_and_does_not_emit_later_events(self):
        token = next(i for i, code in enumerate(self.bundle[1]["dynamic_labels"]) if code.startswith("death:"))
        with patch("semantic_delphi_ukb.track_g_generation.sample_event_and_wait_with_diagnostics",
                   return_value=(token, 1.0, False)):
            result = predict.run(self.example, self.bundle)
        for trajectory in result["trajectories"]:
            self.assertEqual(trajectory["event_count"], 1)
            self.assertEqual(trajectory["stop_reason"], "death")

    def test_history_truncation_and_rolling_context(self):
        value = copy.deepcopy(self.example)
        value["history"] = [{"code": "diag:I10", "age_days": 18000.0 + i * 10} for i in range(140)]
        value["static"]["anchor_age_days"] = 18000
        value["generation"].update(num_trajectories=1, max_new_events=3)
        token = self.bundle[1]["dynamic_labels"].index("diag:I10")
        with patch("semantic_delphi_ukb.track_g_generation.sample_event_and_wait_with_diagnostics",
                   return_value=(token, 1.0, False)):
            result = predict.run(value, self.bundle)
        self.assertEqual(result["history_events_dropped"], 12)
        self.assertEqual(result["trajectories"][0]["event_count"], 3)
        self.assertEqual(result["trajectories"][0]["stop_reason"], "event_limit")

    def test_corrupt_bundle_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(self.bundle[2]), encoding="utf-8")
            (root / "model.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                predict.load_bundle(root)


if __name__ == "__main__":
    unittest.main()
