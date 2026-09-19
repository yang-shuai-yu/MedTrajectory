import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cross_profile_topk", ROOT / "scripts/run_cross_profile_topk_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cross_profile_config_is_frozen_and_preserves_previous_outputs():
    config = json.loads((ROOT / "configs/paper_protocol_v1/cross_profile_topk_v1.json").read_text(encoding="utf-8"))
    assert config["canonical_profile"] == "diagnosis_death"
    assert config["previous_topk_untouched"].endswith("topk_addendum_20260810")
    assert {model["name"] for model in config["models"]} == {"P0_seed42", "P1_seed42", "P2_seed42", "P3_seed42"}


def test_paired_bootstrap_keeps_common_keys_and_reports_delta():
    rows = {
        "P0_seed42": [
            {"patient_index": 1, "age_start_years": 50.0, "target_code": "I10", "disease_id": "0", "exact_top1_hit": 0, "group_top1_hit": 0},
            {"patient_index": 2, "age_start_years": 55.0, "target_code": "I11", "disease_id": "0", "exact_top1_hit": 1, "group_top1_hit": 1},
        ],
        "P2_seed42": [
            {"patient_index": 1, "age_start_years": 50.0, "target_code": "I10", "disease_id": "0", "exact_top1_hit": 1, "group_top1_hit": 1},
            {"patient_index": 2, "age_start_years": 55.0, "target_code": "I11", "disease_id": "0", "exact_top1_hit": 1, "group_top1_hit": 1},
        ],
    }
    result = MODULE.paired_bootstrap(rows, [["P0_seed42", "P2_seed42"]], [1], seed=42, reps=20)
    assert result[0]["common_rows"] == 2
    assert result[0]["common_patients"] == 2
    assert result[0]["delta_mean"] == 0.5
    assert result[0]["p_two_sided"] > 0
