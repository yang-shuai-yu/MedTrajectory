import json
from pathlib import Path

import numpy as np

from scripts.run_track_r_carope_diagnostics import STAGE_ORDER, jobs, parser
from scripts.finalize_track_r_carope_diagnostics import jobs as finalize_jobs
from semantic_delphi_ukb.track_r_batch import load_track_r_static_features, track_r_static_dim
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_track_r_static_conditioning_encodings(tmp_path):
    manifest = {
        "static_token_ids": {
            "static:sex:female": 10,
            "static:sex:male": 11,
            "static:sex:missing": 12,
            "static:bmi:missing": 13,
            "static:bmi:gte30": 14,
        }
    }
    (tmp_path / "prepare_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(tmp_path / "val_static_prefix_token_ids.npy", np.asarray([[10, 14], [11, 13], [12, 13]]))

    legacy = load_track_r_static_features(tmp_path, "val", "legacy-sex-residual")
    categorical = load_track_r_static_features(tmp_path, "val", "categorical-residual")

    assert legacy.tolist() == [[0.0], [1.0], [-3.0]]
    assert categorical.shape == (3, 5)
    assert categorical.sum(axis=1).tolist() == [2.0, 2.0, 2.0]
    assert track_r_static_dim(tmp_path, "categorical-residual") == 5


def test_diagnostic_plan_is_validation_only_and_ordered():
    args = parser().parse_args(["--stage", "all", "--run-tag", "unit_test"])
    protocol = load_track_r_protocol(args.protocol)
    _out_root, stages = jobs(args, protocol)

    assert tuple(stages) == STAGE_ORDER
    assert [len(stages[name]) for name in STAGE_ORDER] == [1, 3, 3, 3, 3, 3]
    rendered = {name: [" ".join(item) for item in commands] for name, commands in stages.items()}
    assert all("--split test" not in item for commands in rendered.values() for item in commands)
    assert "last.pt" in rendered["checkpoint_drift"][0]
    assert "--use-age-encoding true" in rendered["a0_no_static"][0]
    assert "--use-age-rope false" in rendered["a0_no_static"][0]
    assert "--include-static-prefix false" in rendered["a0_no_static"][0]
    assert "--static-conditioning none" in rendered["a0_no_static"][0]
    assert "--static-conditioning legacy-sex-residual" in rendered["legacy_sex_bos"][0]
    assert "--static-conditioning categorical-residual" in rendered["post_trunk_static"][0]
    assert "--static-fusion-stage post_transformer" in rendered["post_trunk_static"][0]
    assert "--use-age-encoding true" in rendered["hybrid_age_poststatic"][0]
    assert "--use-age-rope true" in rendered["hybrid_age_poststatic"][0]
    assert "--include-static-prefix false" in rendered["hybrid_age_poststatic"][0]
    assert "--static-conditioning categorical-residual" in rendered["hybrid_age_poststatic"][0]
    assert "--static-fusion-stage post_transformer" in rendered["hybrid_age_poststatic"][0]
    assert "CARoPE-HybridAge-PostStatic" in rendered["hybrid_age_poststatic"][2]
    assert "--track-r-protocol" not in rendered["legacy_contract_rebuild"][0]


def test_finalize_plan_keeps_cross_contract_rebuild_out_of_paired_commands():
    args = parser().parse_args(["--run-tag", "unit_test"])
    protocol = load_track_r_protocol(args.protocol)
    finalize_args = type("Args", (), {
        "run_tag": "unit_test",
        "hybrid_run_tag": "unit_test_hybrid",
        "current_run_tag": "seed42_retry1",
        "bootstrap": 10,
        "seed": 42,
    })()
    plan = finalize_jobs(finalize_args, protocol)
    rendered = "\n".join(" ".join(plan[name]) for name in ("factorial", "extended", "checkpoint_drift"))
    assert "A1-LegacyContract-Rebuild" not in rendered
    assert "A1-LegacySex-BOS" in " ".join(plan["extended"])
    assert "CARoPE-HybridAge-PostStatic" in " ".join(plan["extended"])
    assert "--factorial-interaction" in plan["factorial"]
