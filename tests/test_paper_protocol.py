from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_delphi_ukb.paper_medical_auc import (  # noqa: E402
    longitudinal_delong_rows,
    official_medical_auc_rows,
)
from semantic_delphi_ukb.paper_protocol import (  # noqa: E402
    exact_random_split,
    is_longitudinal_outcome,
    is_training_input_event,
)
from semantic_delphi_ukb.paper_targets import censor_aware_horizon_target  # noqa: E402
from semantic_delphi_ukb.prepare_paper_protocol_dataset import main as prepare_main  # noqa: E402


class PaperProtocolTests(unittest.TestCase):
    def test_exact_random_split_is_disjoint_complete_and_reproducible(self):
        participant_ids = np.arange(101, dtype=np.int64)
        first = exact_random_split(participant_ids, seed=17)
        second = exact_random_split(participant_ids, seed=17)
        self.assertEqual(first, second)
        self.assertFalse(first.train & first.validation)
        self.assertEqual(first.train | first.validation, frozenset(participant_ids.tolist()))
        self.assertEqual(len(first.validation), 21)

    def test_calendar_windows_exclude_gap(self):
        self.assertTrue(is_training_input_event(date(2020, 6, 30)))
        self.assertFalse(is_training_input_event(date(2020, 7, 1)))
        self.assertFalse(is_longitudinal_outcome(date(2021, 6, 30)))
        self.assertTrue(is_longitudinal_outcome(date(2021, 7, 1)))
        self.assertTrue(is_longitudinal_outcome(date(2022, 7, 1)))

    def test_short_followup_non_case_is_censored(self):
        label, eligible, duration = censor_aware_horizon_target(1000.0, [], 2000.0, 5.0)
        self.assertEqual(label, 0.0)
        self.assertEqual(eligible, 0.0)
        self.assertGreater(duration, 0.0)

    def test_case_is_eligible_even_when_followup_is_short(self):
        label, eligible, _ = censor_aware_horizon_target(1000.0, [1200.0], 1300.0, 5.0)
        self.assertEqual((label, eligible), (1.0, 1.0))

    def test_medical_auc_uses_age_sex_cells(self):
        years = 365.25
        input_ages = np.asarray([[50 * years, 52 * years]] * 8)
        target_ages = np.asarray([[51 * years, 53 * years]] * 8)
        targets = np.asarray([[9, 2], [9, 2], [9, 8], [9, 8]] * 2)
        scores = np.asarray([[0.2, 0.9], [0.2, 0.8], [0.1, 0.2], [0.1, 0.3]] * 2)
        rows = official_medical_auc_rows(
            targets,
            input_ages,
            target_ages,
            scores,
            token_id=2,
            patient_ids=np.arange(8),
            sex_values=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            sex_mapping={"female": 0, "male": 1},
            age_groups=(50, 55),
            seed=5,
        )
        self.assertEqual({row["sex"] for row in rows}, {"female", "male"})
        self.assertTrue(all(row["metric"] == "delphi2m_medical_auc" for row in rows))
        self.assertTrue(all(row["auc_delong"] == 1.0 for row in rows))

    def test_longitudinal_auc_applies_case_threshold(self):
        outcomes = np.zeros((30, 2), dtype=np.int8)
        outcomes[:25, 0] = 1
        outcomes[:24, 1] = 1
        scores = outcomes.astype(np.float64)
        rows = longitudinal_delong_rows(scores, outcomes, ["eligible", "too_few"], minimum_cases=25)
        self.assertEqual([row["disease_id"] for row in rows], ["eligible"])
        self.assertEqual(rows[0]["auc_delong"], 1.0)


class PaperDatasetBuilderTests(unittest.TestCase):
    @staticmethod
    def _static_payload(sex: int, ethnicity: str, age_recruit: float = 60.0) -> dict:
        return {
            "sex": {"field_id": 31, "value_raw": str(sex), "value_id": sex},
            "ethnicity": {"field_id": 21000, "value_raw": ethnicity, "value_id": ethnicity},
            "height_cm": {"field_id": 50, "value": None, "missing": True},
            "weight_kg": {"field_id": 21002, "value": None, "missing": True},
            "bmi": {"field_id": 21001, "value": None, "missing": True},
            "age_recruit": {"field_id": 21022, "value": age_recruit, "missing": False},
        }

    @staticmethod
    def _event(event_type: str, token_key: str, event_date: str, age_days: int) -> dict:
        return {
            "event_type": event_type,
            "token_key": token_key,
            "event_date_raw": event_date,
            "age_days": age_days,
        }

    def test_builder_retains_zero_event_and_excludes_ethnicity_from_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            current = root / "current"
            output = root / "paper"
            source.mkdir()
            current.mkdir()
            static_csv = source / "static.csv"
            canonical = source / "canonical.jsonl"
            vocab = source / "vocab.csv"
            semantic = source / "semantic.npy"

            with static_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["eid"])
                writer.writerows([[101], [102], [103], [104]])
            with vocab.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["token_id", "token_key", "event_type", "code_norm", "embedding_strategy"])
                writer.writerows(
                    [
                        [0, "Padding", "padding", "", "none"],
                        [1, "No event", "no_event", "", "none"],
                        [2, "diag:A01", "diagnosis", "A01", "icd10_64d_pca"],
                        [3, "proc:X01", "procedure", "X01", "learned_vocab"],
                        [4, "death:U00", "death", "U00", "icd10_64d_pca"],
                    ]
                )
            np.save(semantic, np.arange(10, dtype=np.float32).reshape(5, 2))

            records = [
                {
                    "eid": 101,
                    "age_anchor": {"birth_ordinal_inferred": 700000.0},
                    "static": self._static_payload(0, "1001"),
                    "dynamic": [
                        self._event("diagnosis", "diag:A01", "2020-06-30", 20000),
                        self._event("diagnosis", "diag:A01", "2021-07-01", 20366),
                    ],
                },
                {
                    "eid": 102,
                    "age_anchor": {"birth_ordinal_inferred": 700100.0},
                    "static": self._static_payload(1, "3001"),
                    "dynamic": [self._event("death", "death:U00", "2020-06-01", 19000)],
                },
                {
                    "eid": 103,
                    "age_anchor": {"birth_ordinal_inferred": None},
                    "static": self._static_payload(0, "4001"),
                    "dynamic": [],
                },
                {
                    "eid": 104,
                    "age_anchor": {"birth_ordinal_inferred": 700200.0},
                    "static": self._static_payload(1, "2001"),
                    "dynamic": [
                        self._event("procedure", "proc:X01", "2021-01-01", 18000),
                        self._event("diagnosis", "diag:A01", "2022-07-01", 18500),
                    ],
                },
            ]
            canonical.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            for split, ids in {"train": [101], "val": [103], "test": []}.items():
                with (current / f"{split}_patient_index.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["eid"])
                    writer.writerows([[value] for value in ids])

            result = prepare_main(
                [
                    "--canonical-jsonl", str(canonical),
                    "--static-csv", str(static_csv),
                    "--source-vocab-csv", str(vocab),
                    "--source-semantic-npy", str(semantic),
                    "--current-data-dir", str(current),
                    "--output-dir", str(output),
                    "--seed", "3",
                ]
            )
            self.assertEqual(result, 0)
            manifest = json.loads((output / "multitype" / "prepare_manifest.json").read_text(encoding="utf-8"))
            split_total = manifest["split_summaries"]["train"]["patients"] + manifest["split_summaries"]["val"]["patients"]
            self.assertEqual(split_total, 4)
            self.assertEqual(manifest["split_summaries"]["longitudinal"]["patients"], 3)
            self.assertFalse(manifest["ethnicity_model_input"])
            self.assertEqual(np.load(output / "multitype" / "train_static.npy").shape[1], 1)
            with (output / "multitype" / "longitudinal_patient_index.csv").open(encoding="utf-8") as handle:
                longitudinal_index = list(csv.DictReader(handle))
            self.assertTrue(any(row["birth_anchor_available"] == "0" for row in longitudinal_index))
            longitudinal_data = np.fromfile(
                output / "multitype" / "longitudinal.bin", dtype=np.uint32
            ).reshape(-1, 3)
            patient_101 = longitudinal_data[longitudinal_data[:, 0] == 101]
            self.assertEqual(patient_101[-1, 2], 0)
            self.assertEqual(
                patient_101[-1, 1],
                date(2020, 6, 30).toordinal() - 700000,
            )
            self.assertEqual(manifest["split_summaries"]["longitudinal"]["real_events"], 1)

            locked_test = root / "locked_test_eids.csv"
            locked_test.write_text("eid\n103\n", encoding="utf-8")
            locked_output = root / "paper_protocol_locked"
            result = prepare_main(
                [
                    "--canonical-jsonl", str(canonical),
                    "--static-csv", str(static_csv),
                    "--source-vocab-csv", str(vocab),
                    "--source-semantic-npy", str(semantic),
                    "--current-data-dir", str(current),
                    "--output-dir", str(locked_output),
                    "--test-eids-csv", str(locked_test),
                    "--seed", "3",
                ]
            )
            self.assertEqual(result, 0)
            locked_manifest = json.loads((locked_output / "multitype" / "prepare_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(locked_manifest["split_summaries"]["test"]["patients"], 1)
            self.assertEqual(
                locked_manifest["split_summaries"]["train"]["patients"]
                + locked_manifest["split_summaries"]["val"]["patients"],
                3,
            )
            self.assertEqual(locked_manifest["split"]["test_participants"], 1)


if __name__ == "__main__":
    unittest.main()
