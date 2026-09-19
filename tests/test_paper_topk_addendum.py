import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_topk_addendum", ROOT / "scripts/run_paper_protocol_topk_addendum.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_config_has_four_models_and_two_splits():
    config = json.loads((ROOT / "configs/paper_protocol_v1/topk_addendum_v1.json").read_text(encoding="utf-8"))
    MODULE.validate_config(config)
    assert config["test_evaluation_once"] is True
    assert config["analysis_status"] == "secondary_posthoc_addendum"


def test_merge_keeps_profiles_and_splits(tmp_path):
    config = {
        "analysis_status": "secondary_posthoc_addendum",
        "splits": ["val", "test"],
        "models": [
            {"name": "P0_seed42", "data_profile": "diagnosis_death"},
            {"name": "P1_seed42", "data_profile": "diagnosis_death"},
            {"name": "P2_seed42", "data_profile": "multitype"},
            {"name": "P3_seed42", "data_profile": "multitype"},
        ],
    }
    fields = [
        "model", "disease_id", "n_next_diagnosis_targets",
        "diagnosis_exact_top1_hit_rate", "diagnosis_exact_top5_hit_rate",
        "diagnosis_exact_top10_hit_rate", "diagnosis_group_top10_hit_rate",
    ]
    for split in config["splits"]:
        for model in config["models"]:
            path = tmp_path / split / model["name"] / "topk_summary.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "model": model["name"], "disease_id": "overall_panel_targets",
                    "n_next_diagnosis_targets": 10, "diagnosis_exact_top1_hit_rate": 0.1,
                    "diagnosis_exact_top5_hit_rate": 0.2, "diagnosis_exact_top10_hit_rate": 0.3,
                    "diagnosis_group_top10_hit_rate": 0.4,
                })
    merged = MODULE.merge_outputs(config, tmp_path)
    assert len(merged) == 8
    assert {row["data_profile"] for row in merged} == {"diagnosis_death", "multitype"}
    assert (tmp_path / "topk_overall.md").exists()

