from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from semantic_delphi_ukb.evaluate_multitype import (
    build_event_type_lookup,
    evaluate_split,
    load_split,
    load_token_event_types,
    parse_event_types,
    parse_topk,
    resolve_ckpt_path,
    resolve_data_dir,
    resolve_token_vocab_csv,
)
from semantic_delphi_ukb.type_softmax_model import TypeSoftmaxMultitypeConfig, TypeSoftmaxMultitypeDelphi
from semantic_delphi_ukb.type_softmax_utils import load_token_type_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MedTrajectory event-type softmax checkpoint.")
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--token-vocab-csv", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument("--target-event-types", type=str, default=None)
    parser.add_argument("--candidate-event-types", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = resolve_ckpt_path(args)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    data_dir = resolve_data_dir(args, checkpoint_config)

    conf = TypeSoftmaxMultitypeConfig(**checkpoint["model_args"])
    token_type_ids, _ = load_token_type_ids(data_dir, int(conf.vocab_size))
    model = TypeSoftmaxMultitypeDelphi(conf, token_type_ids=token_type_ids)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    token_vocab_csv = resolve_token_vocab_csv(args, data_dir)
    token_event_types = load_token_event_types(token_vocab_csv, int(conf.vocab_size))
    target_lookup = build_event_type_lookup(token_event_types, parse_event_types(args.target_event_types), args.device)
    candidate_lookup = build_event_type_lookup(token_event_types, parse_event_types(args.candidate_event_types), args.device)
    topk_values = parse_topk(args.topk)

    metrics = []
    for split_name in [item.strip() for item in args.splits.split(",") if item.strip()]:
        data, p2i = load_split(data_dir / f"{split_name}.bin")
        static_matrix = np.load(data_dir / f"{split_name}_static.npy").astype(np.float32)
        metrics.append(
            evaluate_split(
                model=model,
                data=data,
                p2i=p2i,
                static_matrix=static_matrix,
                split_name=split_name,
                batch_size=args.batch_size,
                block_size=int(conf.block_size),
                device=args.device,
                no_event_token_rate=int(checkpoint_config.get("no_event_token_rate", 5)),
                ignore_tokens=list(conf.ignore_tokens),
                max_patients=args.max_patients,
                topk_values=topk_values,
                target_lookup=target_lookup,
                candidate_lookup=candidate_lookup,
            )
        )

    target_event_types = parse_event_types(args.target_event_types)
    candidate_event_types = parse_event_types(args.candidate_event_types)
    payload = {
        "ckpt_path": str(ckpt_path),
        "data_dir": str(data_dir),
        "token_vocab_csv": None if token_vocab_csv is None else str(token_vocab_csv),
        "target_event_types": None if target_event_types is None else sorted(target_event_types),
        "candidate_event_types": None if candidate_event_types is None else sorted(candidate_event_types),
        "model_family": checkpoint_config.get("model_family", "medtrajectory_type_softmax"),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "metrics": metrics,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
