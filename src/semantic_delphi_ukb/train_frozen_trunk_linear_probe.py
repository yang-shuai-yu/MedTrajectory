"""Train a linear fixed-horizon probe while keeping the trajectory trunk frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.evaluate_horizon_control_tasks import captured_hidden, load_model  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    build_horizon_targets,
    build_patient_disease_ages,
    load_followup_end_ages,
    load_selected_disease_token_groups,
    load_split,
    parse_horizons,
    resolve_diseases_yaml,
)
from semantic_delphi_ukb.evaluate_horizon_risk_locked_test import patient_batches  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-checkpoint", type=Path, required=True)
    p.add_argument("--diseases-yaml", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--horizons", default="1,5,10")
    p.add_argument("--max-iters", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p


@torch.no_grad()
def evaluate_probe_loss(model, probe, data, p2i, static, patient_ages, followup_end, diseases, horizons, args):
    model.eval()
    probe.eval()
    total_loss = 0.0
    total_count = 0
    for ix in patient_batches(len(p2i), args.batch_size):
        x, age, y, _target_age, static_batch = get_batch(
            ix, data, p2i, static, select="left", padding="regular", block_size=args.block_size,
            device=args.device, cut_batch=False,
        )
        keep, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(
            ix, x, age, y, patient_ages, followup_end, horizons, args.device, censor_aware=True
        )
        hidden = captured_hidden(model, x, age, static_batch)
        batch_index = torch.arange(x.size(0), device=args.device)
        logits = probe(hidden[batch_index, pos]).view(x.size(0), len(horizons), len(diseases))
        valid = mask.bool() & keep[:, None, None]
        if valid.any():
            values = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="sum")
            total_loss += float(values.item())
            total_count += int(valid.sum().item())
    probe.train()
    return total_loss / max(1, total_count)


def main():
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    horizons = parse_horizons(args.horizons)
    data, p2i, static = load_split(args.data_dir, "train", 0)
    val_data, val_p2i, val_static = load_split(args.data_dir, "val", 0)
    diseases, token_groups = load_selected_disease_token_groups(resolve_diseases_yaml(args.diseases_yaml), args.data_dir)
    manifest = json.loads((args.data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    vocab_size = int(manifest.get("vocab_size", int(data[:, 2].max()) + 2))
    patient_ages, observed_last = build_patient_disease_ages(data, p2i, token_groups, vocab_size)
    followup_end = load_followup_end_ages(args.data_dir, "train", observed_last, 0)
    val_patient_ages, val_observed_last = build_patient_disease_ages(val_data, val_p2i, token_groups, vocab_size)
    val_followup_end = load_followup_end_ages(args.data_dir, "val", val_observed_last, 0)
    model, _ = load_model("linear", args.base_checkpoint, args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(model.config.n_embd)
    probe = torch.nn.Linear(hidden_dim, len(horizons) * len(diseases)).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    batches = patient_batches(len(p2i), args.batch_size)
    if not batches:
        raise ValueError("train split has no patients")
    probe.train()
    best_val_loss = float("inf")
    best_state = None
    for step in range(args.max_iters):
        if step % len(batches) == 0:
            batch_order = rng.permutation(len(batches))
        ix = batches[int(batch_order[step % len(batches)])]
        x, age, y, target_age, static_batch = get_batch(
            ix, data, p2i, static, select="left", padding="regular", block_size=args.block_size,
            device=args.device, cut_batch=False,
        )
        keep, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(
            ix, x, age, y, patient_ages, followup_end, horizons, args.device, censor_aware=True
        )
        with torch.no_grad():
            hidden = captured_hidden(model, x, age, static_batch)
        logits = probe(hidden[torch.arange(x.size(0), device=args.device), pos]).view(x.size(0), len(horizons), len(diseases))
        valid = mask.bool() & keep[:, None, None]
        if not valid.any():
            continue
        loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, args.eval_interval) == 0 or step + 1 == args.max_iters:
            val_loss = evaluate_probe_loss(
                model, probe, val_data, val_p2i, val_static, val_patient_ages, val_followup_end,
                diseases, horizons, args,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {name: value.detach().cpu().clone() for name, value in probe.state_dict().items()}
    if best_state is not None:
        probe.load_state_dict(best_state, strict=True)
    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": probe.state_dict(),
        "hidden_dim": hidden_dim,
        "horizons": horizons,
        "disease_ids": [d.disease_id for d in diseases],
        "base_checkpoint": str(args.base_checkpoint),
        "seed": args.seed,
        "train_steps": args.max_iters,
        "best_val_loss": best_val_loss,
    }, args.out_checkpoint)


if __name__ == "__main__":
    main()
