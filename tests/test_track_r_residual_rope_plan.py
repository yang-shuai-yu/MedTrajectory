import json
from pathlib import Path

from scripts.run_track_r_residual_rope import STAGES, jobs, load_spec, parser
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_residual_rope_spec_locks_a0_fallback_and_provenance_only_origin():
    args = parser().parse_args(["--run-tag", "unit_test"])
    spec = load_spec(args)
    assert spec["fallback_target"]["model"] == "A0-TokenStatic"
    assert spec["fallback_target"]["retrain"] is False
    assert spec["architecture"]["inner_rope_gate"] is False
    assert "provenance" in spec["architecture"]["phase_origin_note"]
    assert "not a performance hypothesis" in spec["performance_claim_exclusion"]


def test_residual_rope_plan_is_validation_only_and_reuses_a0():
    args = parser().parse_args(["--stage", "all", "--run-tag", "unit_test"])
    protocol = load_track_r_protocol(args.protocol)
    _output, a0_checkpoint, stages = jobs(args, protocol)
    assert tuple(stages) == STAGES
    assert [len(stages[name]) for name in STAGES] == [4, 3]
    rendered = {name: [" ".join(item) for item in commands] for name, commands in stages.items()}
    assert all("--split test" not in item for commands in rendered.values() for item in commands)
    assert all("train_car_rope_pretraining.py" not in item for commands in rendered.values() for item in commands)
    assert all(str(a0_checkpoint) in rendered[name][0] for name in STAGES)
    assert "--residual-rope-mode learned" in rendered["learned"][0]
    assert "--residual-rope-mode fixed" in rendered["fixed1"][0]
    assert "--residual-rope-alpha-override 0.0" in rendered["learned"][2]
    assert "A1-ResidualRoPE-Learned-Alpha0" in rendered["learned"][2]
    assert all("--run-dir" in rendered[name][0] and f"A1-ResidualRoPE" in rendered[name][0] for name in STAGES)
