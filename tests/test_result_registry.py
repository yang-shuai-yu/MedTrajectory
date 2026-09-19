import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("result_registry_builder", ROOT / "scripts/build_result_registry.py")
auditor = _load("result_registry_auditor", ROOT / "scripts/audit_result_registry.py")
summary_builder = _load("result_registry_summary_builder", ROOT / "scripts/build_result_registry_summary.py")


def test_registry_builds_all_current_sources(tmp_path):
    config_path = ROOT / "configs/paper_protocol_v1/result_registry_sources.json"
    rows = builder.build(config_path, tmp_path)
    assert len(rows) > 100
    assert {row["result_class"] for row in rows} == {
        "lm_logits", "horizon_head", "other_medical", "literature_reference"
    }
    assert (tmp_path / "result_registry.jsonl").exists()
    assert (tmp_path / "result_registry.csv").exists()


def test_audit_flags_cross_category_table_mix():
    rows = [
        {
            "record_id": "lm", "result_class": "lm_logits", "task": "x", "metric": "auc",
            "value": 0.8, "split": "val", "dataset_profile": "x", "comparison_group": "same",
            "source_file": "x", "source_locator": "x", "status": "diagnostic", "comparability": "x",
        },
        {
            "record_id": "risk", "result_class": "horizon_head", "task": "x", "metric": "auc",
            "value": 0.7, "split": "val", "dataset_profile": "x", "comparison_group": "same",
            "source_file": "x", "source_locator": "x", "status": "diagnostic", "comparability": "x",
        },
    ]
    result = auditor.audit(rows)
    assert not result["ok"]
    assert any(item["rule"] == "cross_category_table_mix" for item in result["findings"])


def test_audit_accepts_isolated_literature_and_frozen_test():
    rows = [
        {
            "record_id": "test", "result_class": "horizon_head", "task": "x", "metric": "auc",
            "value": 0.7, "split": "test", "dataset_profile": "x", "comparison_group": "test_horizon",
            "source_file": "x", "source_locator": "x", "status": "primary", "comparability": "locked_test",
            "freeze_manifest": "freeze.json",
        },
        {
            "record_id": "ref", "result_class": "literature_reference", "task": "x", "metric": "auc",
            "value": 0.75, "split": "reference", "dataset_profile": "x", "comparison_group": "reference",
            "source_file": "x", "source_locator": "x", "status": "reference", "comparability": "external_reference",
        },
    ]
    result = auditor.audit(rows)
    assert result["ok"]


def test_compact_summary_writes_human_views(tmp_path):
    registry_dir = tmp_path / "registry"
    builder.build(ROOT / "configs/paper_protocol_v1/result_registry_sources.json", registry_dir)
    summary_dir = tmp_path / "compact"
    summary_builder.build(registry_dir / "result_registry.csv", summary_dir)
    assert (summary_dir / "registry_compact_summary.md").exists()
    assert (summary_dir / "locked_test_horizon_summary.csv").exists()
    assert (summary_dir / "locked_test_horizon_auc_auprc.svg").exists()


def test_topk_addendum_is_lm_logits_and_keeps_split_identity(tmp_path):
    rows = builder.build(ROOT / "configs/paper_protocol_v1/result_registry_sources.json", tmp_path)
    addendum = [row for row in rows if row["task"] == "next_diagnosis_topk"]
    assert addendum
    assert {row["result_class"] for row in addendum} == {"lm_logits"}
    assert {row["split"] for row in addendum} == {"val", "test"}
    assert len({row["record_id"] for row in addendum}) == len(addendum)
    assert all(row["freeze_manifest"] for row in addendum if row["split"] == "test")


def test_shared_cross_profile_topk_is_isolated_as_lm_logits(tmp_path):
    rows = builder.build(ROOT / "configs/paper_protocol_v1/result_registry_sources.json", tmp_path)
    shared = [row for row in rows if row["task"].startswith("shared_landmark_next_diagnosis_topk")]
    assert shared
    assert {row["result_class"] for row in shared} == {"lm_logits"}
    assert {row["split"] for row in shared} == {"val", "test"}
    assert all(row["freeze_manifest"] for row in shared if row["split"] == "test")
