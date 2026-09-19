from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_car_rope_locked_retrain import (  # noqa: E402
    select_variants,
    validate_horizon_stage,
    validate_spec,
    variant_command,
)
from repartition_paper_protocol_data import main as repartition_main  # noqa: E402
from build_paper_protocol_freeze_manifest import main as build_freeze_main  # noqa: E402
from audit_paper_protocol_splits import main as split_audit_main  # noqa: E402
from verify_paper_protocol_freeze_manifest import main as verify_freeze_main  # noqa: E402
from check_paper_medical_control_consistency import main as consistency_main  # noqa: E402
from build_validation_official_aggregate_bundle import main as bundle_main  # noqa: E402
from check_car_rope_validation_gate import main as validation_gate_main  # noqa: E402
from run_locked_test_paper_protocol import (  # noqa: E402
    _pairing_audit_compact,
    main as locked_test_main,
)
from semantic_delphi_ukb.paper_run import write_source_manifest  # noqa: E402


class CARoPELockedWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(
            (ROOT / "configs/paper_protocol_v1/CARoPE_locked_retrain_v2.json").read_text(encoding="utf-8")
        )

    def test_reviewed_minimum_matrix_contract(self):
        contract = validate_spec(self.spec, require_inputs=False)
        self.assertEqual(self.spec["seeds"], [42])
        self.assertEqual(self.spec["horizons_years"], [1.0, 5.0, 10.0])
        self.assertIn("data_dir", contract["paths"])

    def test_horizon_command_uses_matching_pretraining_checkpoint(self):
        variant = self.spec["variants"][3]
        command = variant_command(self.spec, variant, 42, "horizon", "cuda")
        init_index = command.index("--init-from-ckpt") + 1
        self.assertIn("seed42", command[init_index])
        self.assertIn("A3_full_carope", command[init_index])
        self.assertTrue(command[init_index].endswith("pretraining\\checkpoints\\best_val_loss.pt") or command[init_index].endswith("pretraining/checkpoints/best_val_loss.pt"))
        self.assertEqual(command[command.index("--horizons") + 1], "1.0,5.0,10.0")

    def test_horizon_stage_preflights_the_complete_matrix(self):
        with tempfile.TemporaryDirectory() as raw:
            spec = dict(self.spec)
            spec["output_root"] = raw
            with self.assertRaises(RuntimeError):
                validate_horizon_stage(spec)

    def test_variant_selection_supports_disjoint_parallel_queues(self):
        selected = select_variants(self.spec, "A1_car_rope_trunk,A3_full_carope")
        self.assertEqual([variant["name"] for variant in selected], [
            "A1_car_rope_trunk", "A3_full_carope"
        ])
        with self.assertRaises(ValueError):
            select_variants(self.spec, "A1_car_rope_trunk,A1_car_rope_trunk")
        with self.assertRaises(ValueError):
            select_variants(self.spec, "A9_unknown")
            for variant in spec["variants"]:
                checkpoint = Path(raw) / "seed42" / variant["name"] / "pretraining/checkpoints/best_val_loss.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
            self.assertTrue(validate_horizon_stage(spec)["horizon_outputs_empty"])
            occupied = Path(raw) / "seed42/A0_sincos_control/horizon/status.json"
            occupied.parent.mkdir(parents=True)
            occupied.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_horizon_stage(spec)

    def test_repartition_writes_new_version_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            profile = source / "multitype"
            profile.mkdir(parents=True)
            (source / "prepare_manifest.json").write_text('{"profiles": {}}', encoding="utf-8")
            (profile / "prepare_manifest.json").write_text('{"vocab_size": 8}', encoding="utf-8")
            for split, eids in (("train", [1, 2]), ("val", [3, 4])):
                np.asarray([[eid, 100 + eid, 2] for eid in eids], dtype=np.uint32).tofile(profile / f"{split}.bin")
                np.save(profile / f"{split}_static.npy", np.zeros((len(eids), 1), dtype=np.float32))
                np.save(profile / f"{split}_followup_end_age_days.npy", np.ones(len(eids), dtype=np.float32))
                (profile / f"{split}_patient_index.csv").write_text(
                    "row_index,eid,num_events,model_eligible,birth_anchor_available,anchor_only,sex_raw,ethnicity_raw\n"
                    + "".join(f"{index},{eid},1,1,1,0,0,0\n" for index, eid in enumerate(eids)),
                    encoding="utf-8",
                )
            eids_csv = root / "test_eids.csv"
            eids_csv.write_text("eid\n4\n", encoding="utf-8")
            output = root / "paper_protocol_v2_locked_test"
            argv = [
                "--source-dir", str(source), "--output-dir", str(output),
                "--test-eids-csv", str(eids_csv), "--profiles", "multitype",
                "--validation-fraction", "0.34",
            ]
            self.assertEqual(repartition_main(argv), 0)
            manifest = json.loads((output / "multitype/prepare_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol_id"], "paper_protocol_v2_locked_test")
            self.assertEqual(manifest["split"]["test_participants"], 1)
            with self.assertRaises(FileExistsError):
                repartition_main(argv)

    def test_v2_consistency_rejects_historical_two_horizon_input(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "medical_protocol": "paper_medical_control_v1",
                "horizons_years": [1, 5, 10],
                "age_groups_years": [50, 55],
                "age_bin_width_years": 5,
                "sex_stratified": True,
                "case_control": "paper_censor_aware_v1",
            }), encoding="utf-8")
            landmark = root / "landmarks.json"
            landmark.write_text("{}", encoding="utf-8")
            control = root / "control.json"
            control.write_text(json.dumps({
                "protocol": "paper_medical_control_v1", "split": "val",
                "checkpoint": str(root / "checkpoint.pt"), "data_dir": str(root / "data"),
                "horizons_years": [1, 5], "age_groups": [50, 55],
                "protocol_signature": {
                    "age_groups_years": [50, 55], "age_bin_width_years": 5,
                    "sex_stratified": True, "case_control": "paper_censor_aware_v1",
                    "shared_landmark_manifest": str(landmark),
                },
                "summary": {"cells": [
                    {"disease_id": "d", "horizon_years": horizon, "auc": 0.5}
                    for horizon in (1, 5)
                ]},
            }), encoding="utf-8")
            official = root / "official.json"
            official.write_text(json.dumps({"horizon_risk_age_sex_macro": [
                {"disease_id": "d", "horizon_years": horizon, "auc": 0.5}
                for horizon in (1, 5)
            ]}), encoding="utf-8")
            out = root / "consistency.json"
            self.assertEqual(consistency_main([
                "--control", f"A0={control}", "--official-aggregates", str(official),
                "--protocol-json", str(protocol), "--out", str(out),
            ]), 2)
            binding = json.loads(out.read_text(encoding="utf-8"))["protocol_binding"]
            self.assertFalse(binding["ok"])
            self.assertIn("official_aggregate_horizons_mismatch", binding["errors"])

    def test_freeze_v2_and_rehash_verification_close_the_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_dir = root / "data"
            data_dir.mkdir()
            test_eids = root / "test_eids.csv"
            test_eids.write_text("eid\n3\n", encoding="utf-8")
            for split, eid in (("train", 1), ("val", 2), ("test", 3)):
                np.asarray([[eid, 100, 2]], dtype=np.uint32).tofile(data_dir / f"{split}.bin")
                np.save(data_dir / f"{split}_static.npy", np.zeros((1, 1), dtype=np.float32))
                np.save(data_dir / f"{split}_followup_end_age_days.npy", np.ones(1, dtype=np.float32))
                (data_dir / f"{split}_patient_index.csv").write_text(
                    f"row_index,eid,num_events\n0,{eid},1\n", encoding="utf-8"
                )
            from semantic_delphi_ukb.paper_protocol_audit import eid_set_sha256
            data_manifest = {
                "protocol_id": "paper_protocol_v2_locked_test",
                "split": {"test_participants": 1},
                "locked_test_source": {"eid_set_sha256": eid_set_sha256([3])},
            }
            (data_dir / "prepare_manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")
            audit_path = root / "split_audit.json"
            self.assertEqual(split_audit_main([
                "--data-dir", str(data_dir), "--splits", "train,val,test", "--require-test",
                "--test-eids-csv", str(test_eids), "--out", str(audit_path),
            ]), 0)
            run_dir = root / "run"
            checkpoint = run_dir / "checkpoints/best_val_horizon_auc.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            pretraining_dir = root / "pretraining"
            pretraining_checkpoint = pretraining_dir / "checkpoints/best_val_loss.pt"
            pretraining_checkpoint.parent.mkdir(parents=True)
            pretraining_checkpoint.write_bytes(b"pretraining")
            source_files = (
                ROOT / "src/semantic_delphi_ukb/car_rope_model.py",
                ROOT / "src/semantic_delphi_ukb/train_car_rope.py",
                ROOT / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
                ROOT / "src/semantic_delphi_ukb/multitype_batch.py",
                ROOT / "src/semantic_delphi_ukb/train_architecture_risk_heads.py",
                ROOT / "src/semantic_delphi_ukb/tte_targets.py",
                ROOT / "scripts/training_monitor.py",
            )
            write_source_manifest(
                pretraining_dir, ROOT, ROOT / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
                extra_paths=source_files,
            )
            write_source_manifest(
                run_dir, ROOT, ROOT / "src/semantic_delphi_ukb/train_car_rope.py",
                extra_paths=source_files,
            )
            (run_dir / "resolved_config.json").write_text(
                json.dumps({"config": {
                    "data_dir": str(data_dir.resolve()),
                    "init_from_ckpt": str(pretraining_checkpoint.resolve()),
                }}), encoding="utf-8"
            )
            diseases = root / "diseases.yaml"
            diseases.write_text("diseases: []\n", encoding="utf-8")
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "age_groups_years": [50, 55], "bootstrap": 10, "bootstrap_seed": 42,
                "calibration_bins": 10, "checkpoint_selection_rule": "max_val_auc",
                "horizons_years": [1, 5, 10], "medical_protocol": "paper_medical_control_v1",
                "age_bin_width_years": 5, "sex_stratified": True,
                "case_control": "paper_censor_aware_v1", "data_protocol_id": "paper_protocol_v2_locked_test",
            }), encoding="utf-8")
            assets = {}
            for name in ("medical_control_evaluator", "comparison_evaluator", "vocab", "shared_validation_landmarks", "shared_test_landmarks"):
                path = root / f"{name}.txt"
                path.write_text(name, encoding="utf-8")
                assets[name] = path
            control_summary = root / "A0_validation_summary.json"
            control_summary.write_text(json.dumps({
                "protocol": "paper_medical_control_v1",
                "split": "val",
                "checkpoint": str(checkpoint.resolve()),
                "data_dir": str(data_dir.resolve()),
                "horizons_years": [1, 5, 10],
                "age_groups": [50, 55],
                "protocol_signature": {
                    "age_groups_years": [50, 55],
                    "age_bin_width_years": 5,
                    "sex_stratified": True,
                    "case_control": "paper_censor_aware_v1",
                    "shared_landmark_manifest": str(assets["shared_validation_landmarks"].resolve()),
                },
                "summary": {"cells": [
                    {"disease_id": "d", "horizon_years": horizon, "auc": 0.5}
                    for horizon in (1, 5, 10)
                ]},
            }), encoding="utf-8")
            official = root / "validation_official_aggregates.json"
            official.write_text(json.dumps({"horizon_risk_age_sex_macro": [
                {"disease_id": "d", "horizon_years": horizon, "auc": 0.5}
                for horizon in (1, 5, 10)
            ]}), encoding="utf-8")
            consistency = root / "validation_consistency.json"
            self.assertEqual(consistency_main([
                "--control", f"A0={control_summary}", "--require", "A0",
                "--official-aggregates", str(official), "--protocol-json", str(protocol),
                "--out", str(consistency),
            ]), 0)
            assets["validation_consistency"] = consistency
            assets["validation_official_aggregates"] = official
            freeze_path = root / "freeze.json"
            argv = [
                "--split-audit", str(audit_path), "--data-dir", str(data_dir),
                "--diseases-yaml", str(diseases), "--checkpoint", f"A0={checkpoint}",
                "--model-data-dir", f"A0={data_dir}", "--model-split-audit", f"A0={audit_path}",
                "--protocol-json", str(protocol), "--test-eids-csv", str(test_eids),
                "--confirm-checkpoint-excludes-test", "--out", str(freeze_path),
            ]
            for name, path in assets.items():
                argv.extend(["--asset", f"{name}={path}"])
            self.assertEqual(build_freeze_main(argv), 0)
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            self.assertEqual(freeze["manifest_version"], "paper_medical_control_locked_test_v2")
            verification = root / "verification.json"
            self.assertEqual(verify_freeze_main(["--freeze-manifest", str(freeze_path), "--out", str(verification)]), 0)
            self.assertTrue(json.loads(verification.read_text(encoding="utf-8"))["ok"])

    def test_consistency_uses_model_specific_official_aggregates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "medical_protocol": "paper_medical_control_v1",
                "horizons_years": [1, 5, 10], "age_groups_years": [50, 55],
                "age_bin_width_years": 5, "sex_stratified": True,
                "case_control": "paper_censor_aware_v1",
            }), encoding="utf-8")
            landmark = root / "landmarks.json"
            landmark.write_text("{}", encoding="utf-8")
            controls = []
            official_inputs = []
            for name, auc in (("A0", 0.6), ("A1", 0.8)):
                control = root / f"{name}_summary.json"
                control.write_text(json.dumps({
                    "protocol": "paper_medical_control_v1", "split": "val",
                    "checkpoint": str(root / f"{name}.pt"), "data_dir": str(root / "data"),
                    "horizons_years": [1, 5, 10], "age_groups": [50, 55],
                    "protocol_signature": {
                        "age_groups_years": [50, 55], "age_bin_width_years": 5,
                        "sex_stratified": True, "case_control": "paper_censor_aware_v1",
                        "shared_landmark_manifest": str(landmark),
                    },
                    "summary": {"cells": [
                        {"disease_id": "d", "horizon_years": horizon, "auc": auc}
                        for horizon in (1, 5, 10)
                    ]},
                }), encoding="utf-8")
                official = root / f"{name}_official.json"
                official.write_text(json.dumps({"horizon_risk_age_sex_macro": [
                    {"disease_id": "d", "horizon_years": horizon, "auc": auc}
                    for horizon in (1, 5, 10)
                ]}), encoding="utf-8")
                controls.extend(["--control", f"{name}={control}", "--require", name])
                official_inputs.extend(["--input", f"{name}={official}"])
            bundle = root / "bundle.json"
            self.assertEqual(bundle_main([*official_inputs, "--out", str(bundle)]), 0)
            out = root / "consistency.json"
            self.assertEqual(consistency_main([
                *controls, "--official-aggregates", str(bundle),
                "--protocol-json", str(protocol), "--out", str(out),
            ]), 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["protocol_binding"]["ok"])
            self.assertEqual(set(payload["protocol_binding"]["official_aggregates"]["model_sources"]), {"A0", "A1"})

    def test_validation_effect_gate_requires_paired_significance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            comparison = root / "comparison.json"
            comparison.write_text(json.dumps({
                "models": {
                    "A0": {"auc": 0.67, "auprc": 0.07},
                    "A3": {"auc": 0.70, "auprc": 0.08},
                },
                "paired_bootstrap": {
                    "A0 - A3": {
                        "delta_auc": -0.03, "ci95_low": -0.04,
                        "ci95_high": -0.02, "p_holm": 0.01,
                    },
                },
            }), encoding="utf-8")
            out = root / "gate.json"
            self.assertEqual(validation_gate_main([
                "--comparison", str(comparison), "--out", str(out),
            ]), 0)
            self.assertTrue(json.loads(out.read_text(encoding="utf-8"))["ok"])

    def test_memory_efficient_pairing_audit_checks_all_models(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [
                {
                    "patient_index": patient,
                    "disease_id": "d",
                    "horizon_years": 1.0,
                    "sex": "female",
                    "age_start_years": 50.0,
                    "prediction_age_days": 50.0 * 365.25,
                    "label": patient % 2,
                    "score": 0.2 + patient * 0.1,
                    "score_is_probability": True,
                }
                for patient in range(4)
            ]
            inputs = []
            for name in ("A0", "A3"):
                path = root / f"{name}.json"
                path.write_text(json.dumps(rows), encoding="utf-8")
                inputs.append((name, str(path)))
            audit = _pairing_audit_compact(inputs)
            self.assertEqual(audit["models"]["A0"]["rows"], 4)
            self.assertEqual(audit["comparisons"]["A0 - A3"]["label_mismatch_rows"], 0)

    def test_locked_test_runner_reaches_gate_without_hashlib_scope_error(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            freeze = root / "freeze.json"
            verification = root / "verification.json"
            split_audit = root / "split_audit.json"
            consistency = root / "consistency.json"
            freeze.write_text("{}", encoding="utf-8")
            verification.write_text("{}", encoding="utf-8")
            split_audit.write_text("{}", encoding="utf-8")
            consistency.write_text("{}", encoding="utf-8")
            out = root / "out"
            self.assertEqual(locked_test_main([
                "--freeze-manifest", str(freeze),
                "--freeze-verification", str(verification),
                "--split-audit", str(split_audit),
                "--consistency", str(consistency),
                "--input", f"A0={root / 'missing_rows.json'}",
                "--out-dir", str(out),
            ]), 2)
            self.assertTrue((out / "locked_test_gate.json").is_file())


if __name__ == "__main__":
    unittest.main()
