from scripts.finalize_track_r_residual_rope import parser, plan
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol


def test_residual_rope_finalize_is_paired_and_validation_only():
    args = parser().parse_args(["--run-tag", "unit_test", "--bootstrap", "10"])
    protocol = load_track_r_protocol(args.protocol)
    details = plan(args, protocol)
    rendered = " ".join(details["command"])
    assert tuple(details["inputs"]) == (
        "A0-TokenStatic",
        "A1-ResidualRoPE-Learned",
        "A1-ResidualRoPE-Learned-Alpha0",
        "A1-ResidualRoPE-Fixed1",
    )
    assert "--memory-efficient" in rendered
    assert "--bootstrap 10" in rendered
    assert "--split test" not in rendered
