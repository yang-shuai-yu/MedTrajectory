from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np
import torch

from semantic_delphi_ukb.evaluate_modern_multitype import build_parser
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
from semantic_delphi_ukb.tte_model import TTEConfig, TTEMultitaskDelphi


class TTELogitsOnly(torch.nn.Module):
    def __init__(self, model: TTEMultitaskDelphi):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, *args, **kwargs):
        logits, loss, att, _ = self.model(*args, **kwargs)
        return logits, loss, att


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = resolve_ckpt_path(args)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    data_dir = resolve_data_dir(args, checkpoint_config)

    conf = TTEConfig(**checkpoint["model_args"])
    model = TTEMultitaskDelphi(conf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = TTELogitsOnly(model.to(args.device))
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

    payload = {
        "ckpt_path": str(ckpt_path),
        "data_dir": str(data_dir),
        "token_vocab_csv": None if token_vocab_csv is None else str(token_vocab_csv),
        "model_family": checkpoint_config.get("model_family", "medtrajectory_tte_multitask"),
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
