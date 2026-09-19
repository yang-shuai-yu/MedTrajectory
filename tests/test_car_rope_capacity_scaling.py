import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_car_rope_capacity_scaling import select_variants, validate_spec, variant_command


def test_capacity_spec_and_commands_include_transformer_dimensions():
    spec = json.loads((ROOT / "configs/paper_protocol_v1/CARoPE_capacity_scaling_v1.json").read_text())
    validate_spec(spec, require_inputs=False)
    variants = select_variants(spec, "A1-M,A1-L")
    command = variant_command(spec, variants[0], "pretraining", "cuda")
    assert command[command.index("--n-layer") + 1] == "12"
    assert command[command.index("--n-head") + 1] == "12"
    assert command[command.index("--n-embd") + 1] == "120"
    assert command[command.index("--use-relative-horizon-query") + 1] == "false"


def test_capacity_variant_selection_rejects_unknown_name():
    spec = json.loads((ROOT / "configs/paper_protocol_v1/CARoPE_capacity_scaling_v1.json").read_text())
    try:
        select_variants(spec, "A1-XL")
    except ValueError as exc:
        assert "unknown variants" in str(exc)
    else:
        raise AssertionError("unknown capacity variant was accepted")


def test_all_stage_can_render_horizon_command_before_checkpoint_exists():
    spec = json.loads((ROOT / "configs/paper_protocol_v1/CARoPE_capacity_scaling_v1.json").read_text())
    command = variant_command(spec, spec["variants"][0], "horizon", "cuda")
    assert "--init-from-ckpt" in command
