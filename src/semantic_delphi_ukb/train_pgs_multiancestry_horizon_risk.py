from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import DATA_ROOT, EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import DATA_ROOT, EXTERNAL_ROOT

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.pgs_horizon import (  # noqa: E402
    PGSResidualAdapter,
    adapter_logits,
    collect_final_context_cache,
    load_base_model,
    load_split,
    prediction_metrics,
)
from semantic_delphi_ukb.train_architecture_risk_heads import parse_horizons  # noqa: E402
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages, load_selected_disease_token_groups  # noqa: E402
from training_monitor import RunMonitor  # noqa: E402


DEFAULT_DATA_DIR = Path(str(EXTERNAL_ROOT / "data" / "ukb_semantic_multitype_explicit_split"))
DEFAULT_PGS_DIR = Path(str(DATA_ROOT / "pgs_i21_euasian_v1"))
DEFAULT_CKPT = REPO_DIR / "results" / "monotonic_horizon" / "monotonic_gated_rope_gate100" / "ckpt.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen-base PGS adapter with balanced European and Asian samples.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pgs-dir", type=Path, default=DEFAULT_PGS_DIR)
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "docs" / "selected_diseases.yaml")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-disease-ids", default="ischemic_heart_disease,myocardial_infarction")
    parser.add_argument("--horizons", default="5,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def load_token_groups_compat(diseases_yaml: Path, data_dir: Path):
    try:
        return load_selected_disease_token_groups(diseases_yaml, data_dir)
    except TypeError as exc:
        if "positional argument" not in str(exc):
            raise
        return load_selected_disease_token_groups(diseases_yaml)


def resolve_targets(diseases, spec: str) -> tuple[list[str], list[int]]:
    target_ids = [item.strip() for item in spec.split(",") if item.strip()]
    lookup = {disease.disease_id: index for index, disease in enumerate(diseases)}
    missing = [disease_id for disease_id in target_ids if disease_id not in lookup]
    if missing:
        raise ValueError(f"Unknown target disease IDs: {missing}")
    return target_ids, [lookup[disease_id] for disease_id in target_ids]


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cache_group(base_model, data_dir, pgs_dir, split, row_mask, token_groups, horizons, target_indices, args):
    data, p2i, static, pgs, available = load_split(data_dir, pgs_dir, split)
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(
        data, p2i, token_groups, int(base_model.config.vocab_size)
    )
    return collect_final_context_cache(
        base_model, data, p2i, static, pgs, available, patient_disease_ages, patient_last_ages,
        horizons, target_indices, device=args.device, block_size=int(base_model.config.block_size),
        batch_size=args.base_batch_size, available_only=True, row_mask=row_mask,
    )


def group_metrics(adapter, cache, args):
    fused = adapter_logits(adapter, cache, args.device, args.train_batch_size)
    return prediction_metrics(cache.base_logits, cache.labels, cache.mask), prediction_metrics(fused, cache.labels, cache.mask)


def main() -> int:
    args = build_parser().parse_args()
    if args.run_dir.exists() and args.resume is None:
        unexpected = [path.name for path in args.run_dir.iterdir() if path.name != "stdout.log"]
        if unexpected:
            raise FileExistsError(f"Run directory contains training artifacts: {sorted(unexpected)}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    horizons = parse_horizons(args.horizons)
    diseases, token_groups = load_token_groups_compat(args.diseases_yaml, args.data_dir)
    target_ids, target_indices = resolve_targets(diseases, args.target_disease_ids)
    manifest = json.loads((args.pgs_dir / "manifest.json").read_text(encoding="utf-8"))
    base_model, base_checkpoint = load_base_model(args.base_checkpoint, args.device)
    config = {
        **vars(args), "horizons": horizons, "target_disease_ids": target_ids,
        "target_disease_indices": target_indices, "pgs_feature_order": manifest["feature_order"],
        "base_checkpoint_best_val_auc_mean": base_checkpoint.get("best_val_auc_mean"),
        "base_model_frozen": True, "sampling": "equal European and Asian complete-case rows per epoch",
        "selection_metric": "mean(European val AUPRC, Asian val AUPRC)",
    }
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    global_step = 0
    best_metric = -float("inf")
    best_epoch = -1
    patience_used = 0
    last_epoch = -1
    try:
        masks = {
            "eu_train": np.load(args.pgs_dir / "train_adapter_mask.npy"),
            "eu_val": np.load(args.pgs_dir / "val_adapter_mask.npy"),
            "asian_train": np.load(args.pgs_dir / "asian_train_adapter_mask.npy"),
            "asian_val": np.load(args.pgs_dir / "asian_val_adapter_mask.npy"),
        }
        caches = {}
        specs = {"eu_train": ("train", masks["eu_train"]), "eu_val": ("train", masks["eu_val"]),
                 "asian_train": ("val", masks["asian_train"]), "asian_val": ("val", masks["asian_val"])}
        for name, (split, mask) in specs.items():
            monitor.mark_running(phase=f"caching_{name}", epoch=0, global_step=0)
            caches[name] = cache_group(base_model, args.data_dir, args.pgs_dir, split, mask, token_groups, horizons, target_indices, args)

        adapter = PGSResidualAdapter(caches["eu_train"].pgs_features.shape[1], args.hidden_dim, len(horizons), len(target_ids), args.dropout).to(args.device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        start_epoch = 0
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
            adapter.load_state_dict(checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_metric"])
            best_epoch = int(checkpoint["best_epoch"])
            patience_used = int(checkpoint["patience_used"])
            restore_random_state(checkpoint["random_state"])
        group_size = min(len(caches["eu_train"].base_logits), len(caches["asian_train"].base_logits))
        monitor.update_status(status="running", phase="training", train_patients_per_ancestry=group_size,
                              eu_val_patients=len(caches["eu_val"].base_logits), asian_val_patients=len(caches["asian_val"].base_logits))
        for epoch in range(start_epoch, args.epochs):
            last_epoch = epoch
            epoch_start = time.perf_counter()
            eu_rows = np.random.choice(len(caches["eu_train"].base_logits), group_size, replace=False)
            asian_rows = np.random.choice(len(caches["asian_train"].base_logits), group_size, replace=False)
            groups = np.concatenate([np.zeros(group_size, dtype=np.int8), np.ones(group_size, dtype=np.int8)])
            rows = np.concatenate([eu_rows, asian_rows])
            order = np.random.permutation(len(rows))
            epoch_loss = 0.0
            adapter.train()
            for start in range(0, len(order), args.train_batch_size):
                step_start = time.perf_counter()
                selected = order[start:start + args.train_batch_size]
                base_parts, feature_parts, available_parts, label_parts, mask_parts = [], [], [], [], []
                for group_id, cache_name in ((0, "eu_train"), (1, "asian_train")):
                    keep = selected[groups[selected] == group_id]
                    if not len(keep):
                        continue
                    index = rows[keep]
                    cache = caches[cache_name]
                    base_parts.append(cache.base_logits[index]); feature_parts.append(cache.pgs_features[index])
                    available_parts.append(cache.pgs_available[index]); label_parts.append(cache.labels[index]); mask_parts.append(cache.mask[index])
                base = torch.as_tensor(np.concatenate(base_parts), device=args.device)
                features = torch.as_tensor(np.concatenate(feature_parts), device=args.device)
                available = torch.as_tensor(np.concatenate(available_parts), dtype=torch.float32, device=args.device)
                labels = torch.as_tensor(np.concatenate(label_parts), device=args.device)
                valid = torch.as_tensor(np.concatenate(mask_parts), dtype=torch.float32, device=args.device)
                fused, delta, gate = adapter(base, features, available)
                loss = masked_loss(fused, labels, valid)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0); optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(selected); global_step += 1
                if global_step % args.log_every == 0:
                    elapsed = time.perf_counter() - step_start
                    monitor.log_step(global_step, {"loss/train_total": loss, "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                        "optimization/gradient_norm": grad_norm, "model/pgs_gate_mean": gate.mean(),
                        "representation_norm/pgs_delta": torch.linalg.vector_norm(delta.flatten(1), dim=1).mean(),
                        "performance/step_time_seconds": elapsed, "performance/samples_per_second": len(selected) / max(elapsed, 1e-9)})

            eu_base, eu_fused = group_metrics(adapter, caches["eu_val"], args)
            as_base, as_fused = group_metrics(adapter, caches["asian_val"], args)
            selection = float(np.mean([eu_fused["auprc_macro"], as_fused["auprc_macro"]]))
            improved = selection > best_metric
            if improved: best_metric, best_epoch, patience_used = selection, epoch, 0
            else: patience_used += 1
            metrics = {"epoch": epoch, "global_step": global_step, "train_loss": epoch_loss / len(order),
                "eu_val_base_auroc": eu_base["auroc_macro"], "eu_val_fused_auroc": eu_fused["auroc_macro"],
                "eu_val_base_auprc": eu_base["auprc_macro"], "eu_val_fused_auprc": eu_fused["auprc_macro"],
                "eu_val_base_brier": eu_base["brier_macro"], "eu_val_fused_brier": eu_fused["brier_macro"],
                "asian_val_base_auroc": as_base["auroc_macro"], "asian_val_fused_auroc": as_fused["auroc_macro"],
                "asian_val_base_auprc": as_base["auprc_macro"], "asian_val_fused_auprc": as_fused["auprc_macro"],
                "asian_val_base_brier": as_base["brier_macro"], "asian_val_fused_brier": as_fused["brier_macro"],
                "val_balanced_auprc": selection, "best_val_balanced_auprc": best_metric, "best_epoch": best_epoch,
                "patience_used": patience_used, "pgs_gate_mean": float(torch.sigmoid(adapter.gate_logits).mean().detach().cpu()),
                "epoch_time_seconds": time.perf_counter() - epoch_start}
            monitor.log_epoch(metrics); monitor.update_status(status="running", phase="training", **metrics, **monitor.system_metrics())
            state = {"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None, "scaler": None,
                     "epoch": epoch, "global_step": global_step, "best_metric": best_metric, "best_epoch": best_epoch,
                     "patience_used": patience_used, "config": config, "random_state": monitor.capture_random_state()}
            monitor.save_checkpoint(state, "last.pt")
            if improved: monitor.save_checkpoint(state, "best_auprc.pt")
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if patience_used >= args.patience: break
        monitor.mark_finished(phase="finished", epoch=last_epoch, global_step=global_step, best_metric=best_metric, best_epoch=best_epoch)
    except BaseException as exc:
        monitor.mark_failed(exc, epoch=last_epoch, global_step=global_step); raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
