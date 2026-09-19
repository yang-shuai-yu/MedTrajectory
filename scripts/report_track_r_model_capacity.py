"""Write exact Track R model-capacity reports after the vocabulary is frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.medbert_r_model import MedBERTR, MedBERTRConfig, parameter_report  # noqa: E402
from semantic_delphi_ukb.train_car_rope import model_size_payload  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    manifest = json.loads((args.data_dir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    embeddings = np.load(args.data_dir / manifest["semantic_output"]).astype(np.float32)
    vocab_size, semantic_dim = embeddings.shape
    variants = {}
    for item in protocol["transformer_variants"]:
        model = CARoPEHorizonMedTrajectory(
            CARoPEConfig(
                block_size=protocol["dynamic_context_length"] + 1 + (protocol["static_prefix"]["fixed_length"] if item["static_prefix"] else 0),
                vocab_size=vocab_size,
                n_layer=item["n_layer"],
                n_head=item["n_head"],
                n_embd=item["n_embd"],
                semantic_embedding_dim=semantic_dim,
                static_dim=0,
                static_hidden_dim=item["n_embd"],
                num_tte_tasks=10,
                num_horizons=len(protocol["horizons_years"]),
                horizon_years=tuple(protocol["horizons_years"]),
                use_age_encoding=item["age_encoding"] == "sincos",
                use_age_rope=item["age_encoding"] == "carope",
                use_relative_horizon_query=False,
            ),
            pretrained_token_embeddings=embeddings,
        )
        variants[item["name"]] = model_size_payload(model)
    for item in protocol["medbert_variants"]:
        model = MedBERTR(
            MedBERTRConfig(
                vocab_size=vocab_size,
                max_sequence_length=protocol["dynamic_context_length"] + protocol["static_prefix"]["fixed_length"] + 1,
                n_layer=item["n_layer"],
                n_head=item["n_head"],
                hidden_size=item["hidden_size"],
                intermediate_size=item["intermediate_size"],
                num_horizons=len(protocol["horizons_years"]),
            )
        )
        variants[item["name"]] = {**parameter_report(model), "capacity_role": item["capacity_role"]}
    reference = variants["A1-TokenStatic"]["parameter_groups"]["transformer_blocks"]
    matched = variants["Med-BERT-Matched-S"]["transformer_block_parameters"]
    relative_error = abs(matched - reference) / reference
    tolerance = next(item for item in protocol["medbert_variants"] if item["name"] == "Med-BERT-Matched-S")["matching_tolerance_fraction"]
    payload = {
        "protocol_id": protocol["protocol_id"],
        "vocab_size": vocab_size,
        "semantic_embedding_dim": semantic_dim,
        "variants": variants,
        "matched_block_relative_error": relative_error,
        "matched_block_tolerance": tolerance,
        "matched_block_passed": relative_error <= tolerance,
        "comparison_rule": "Report total trainable and transformer-block parameters separately.",
    }
    if not payload["matched_block_passed"]:
        raise ValueError(f"Med-BERT-Matched-S block mismatch is {relative_error:.3%}, above {tolerance:.3%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "matched_block_relative_error": relative_error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
