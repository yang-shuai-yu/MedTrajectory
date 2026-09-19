from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.evaluate_medical_control_tasks import _summary  # noqa: E402
from semantic_delphi_ukb.paper_protocol_audit import (  # noqa: E402
    audit_disjoint_splits,
    audit_split,
    build_freeze_manifest,
    eid_sequence_sha256,
    eid_set_sha256,
)


class PaperProtocolAuditTest(unittest.TestCase):
    def test_split_audit_checks_alignment_and_overlap(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for split, eids in (("train", [1, 2]), ("val", [3]), ("test", [4])):
                np.asarray([[eid, 100, 2] for eid in eids], dtype=np.uint32).tofile(root / f"{split}.bin")
                np.save(root / f"{split}_static.npy", np.zeros((len(eids), 1), dtype=np.float32))
                np.save(root / f"{split}_followup_end_age_days.npy", np.ones(len(eids), dtype=np.float32))
                (root / f"{split}_patient_index.csv").write_text(
                    "row_index,eid,num_events\n" + "".join(f"{idx},{eid},1\n" for idx, eid in enumerate(eids)), encoding="utf-8"
                )
            audits = {split: audit_split(root, split) for split in ("train", "val", "test")}
            self.assertTrue(audit_disjoint_splits(audits)["ok"])
            self.assertEqual(audits["train"]["eid_set_sha256"], eid_set_sha256([1, 2]))
            audits["test"]["eids"] = [2]
            self.assertFalse(audit_disjoint_splits(audits)["ok"])

    def test_split_audit_rejects_trajectory_index_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            np.asarray([[99, 100, 2]], dtype=np.uint32).tofile(root / "test.bin")
            np.save(root / "test_static.npy", np.zeros((1, 1), dtype=np.float32))
            np.save(root / "test_followup_end_age_days.npy", np.ones(1, dtype=np.float32))
            (root / "test_patient_index.csv").write_text("row_index,eid,num_events\n0,4,1\n", encoding="utf-8")
            audit = audit_split(root, "test", require_nonempty=True)
            self.assertFalse(audit["ok"])
            self.assertIn("trajectory_eids_do_not_match_patient_index", audit["errors"])

    def test_eid_hash_distinguishes_sequence_from_set(self):
        self.assertNotEqual(eid_sequence_sha256([1, 2]), eid_sequence_sha256([2, 1]))
        self.assertEqual(eid_set_sha256([1, 2]), eid_set_sha256([2, 1]))

    def test_freeze_manifest_records_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            disease = root / "diseases.yaml"
            checkpoint = root / "ckpt.pt"
            disease.write_text("diseases: []\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            manifest = build_freeze_manifest(root, {"test": {"ok": True}}, {"p3": checkpoint}, disease, {"seed": 1337})
            self.assertEqual(manifest["protocol"]["seed"], 1337)
            self.assertEqual(len(manifest["checkpoints"]["p3"]["sha256"]), 64)

    def test_medical_summary_separates_lm_calibration(self):
        rows = [
            {"sex": "female", "disease_id": "d", "horizon_years": 1.0, "age_start_years": 50.0, "label": 0, "score": -1.0},
            {"sex": "female", "disease_id": "d", "horizon_years": 1.0, "age_start_years": 50.0, "label": 1, "score": 1.0},
        ]
        summary = _summary(rows, "lm", 10)
        self.assertEqual(summary["valid_auc_cells"], 1)
        self.assertEqual(summary["macro"]["auc"], 1.0)
        self.assertTrue(np.isnan(summary["macro"]["brier"]))


if __name__ == "__main__":
    unittest.main()
