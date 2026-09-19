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
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    load_selected_disease_token_groups,
)
from training_monitor import RunMonitor  # noqa: E402


DEFAULT_DATA_DIR = Path(str(EXTERNAL_ROOT / "data" / "ukb_semantic_multitype_explicit_split"))
DEFAULT_PGS_DIR = Path(str(DATA_ROOT / "pgs_i21_v1"))
DEFAULT_CKPT = REPO_DIR / "results" / "monotonic_horizon" / "monotonic_gated_rope_gate100" / "ckpt.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PGS residual risk adapter over frozen Monotonic Gated RoPE logits.")
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def resolve_targets(diseases, spec: str) -> tuple[list[str], list[int]]:
    target_ids = [item.strip() for item in spec.split(",") if item.strip()]
    disease_lookup = {disease.disease_id: index for index, disease in enumerate(diseases)}
    missing = [disease_id for disease_id in target_ids if disease_id not in disease_lookup]
    if missing:
        raise ValueError(f"Unknown target disease IDs: {missing}")
    return target_ids, [disease_lookup[disease_id] for disease_id in target_ids]


def load_token_groups_compat(diseases_yaml: Path, data_dir: Path):
    try:
        return load_selected_disease_token_groups(diseases_yaml, data_dir)
    except TypeError as exc:
        if "positional argument" not in str(exc):
            raise
        return load_selected_disease_token_groups(diseases_yaml)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def masked_numpy_loss(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    raw = np.logaddexp(0.0, logits.astype(np.float64)) - labels * logits
    return float(raw[mask].mean())


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def self_test(args: argparse.Namespace) -> int:
    adapter = PGSResidualAdapter(16, args.hidden_dim, 2, 2, args.dropout).to(args.device)
    base = torch.tensor([[[0.1, -0.2], [0.3, 0.0]]], device=args.device)
    features = torch.randn(1, 16, device=args.device)
    available = torch.ones(1, device=args.device)
    fused, delta, _ = adapter(base, features, available)
    if not torch.allclose(base, fused) or not torch.allclose(delta, torch.zeros_like(delta)):
        raise RuntimeError("Zero-initialized adapter must reproduce the frozen base logits")
    loss = masked_loss(fused, torch.ones_like(fused), torch.ones_like(fused))
    loss.backward()
    print(json.dumps({"self_test": True, "loss": float(loss.detach().cpu())}))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test(args)
    if args.run_dir.exists() and args.resume is None:
        unexpected = [path.name for path in args.run_dir.iterdir() if path.name != "stdout.log"]
        if unexpected:
            raise FileExistsError(
                f"Run directory already contains training artifacts: {args.run_dir} ({sorted(unexpected)})"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    horizons = parse_horizons(args.horizons)
    diseases, token_groups = load_token_groups_compat(args.diseases_yaml, args.data_dir)
    target_ids, target_indices = resolve_targets(diseases, args.target_disease_ids)
    manifest = json.loads((args.pgs_dir / "manifest.json").read_text(encoding="utf-8"))
    base_model, base_checkpoint = load_base_model(args.base_checkpoint, args.device)
    config = {
        **vars(args),
        "horizons": horizons,
        "target_disease_ids": target_ids,
        "target_disease_indices": target_indices,
        "pgs_feature_order": manifest["feature_order"],
        "base_checkpoint_best_val_auc_mean": base_checkpoint.get("best_val_auc_mean"),
    }

    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    global_step = 0
    last_epoch = -1
    best_metric = -float("inf")
    best_epoch = -1
    patience_used = 0
    try:
        monitor.mark_running(phase="loading_base", epoch=0, global_step=0)

        caches = {}
        for split in ("train", "val"):
            monitor.update_status(status="running", phase=f"caching_{split}", global_step=global_step)
            data, p2i, static, pgs, available = load_split(args.data_dir, args.pgs_dir, split)
            patient_disease_ages, patient_last_ages = build_patient_disease_ages(
                data, p2i, token_groups, int(base_model.config.vocab_size)
            )
            caches[split] = collect_final_context_cache(
                base_model,
                data,
                p2i,
                static,
                pgs,
                available,
                patient_disease_ages,
                patient_last_ages,
                horizons,
                target_indices,
                device=args.device,
                block_size=int(base_model.config.block_size),
                batch_size=args.base_batch_size,
                available_only=True,
            )

        train_cache = caches["train"]
        val_cache = caches["val"]
        adapter = PGSResidualAdapter(
            train_cache.pgs_features.shape[1],
            args.hidden_dim,
            len(horizons),
            len(target_ids),
            args.dropout,
        ).to(args.device)
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

        train_count = len(train_cache.base_logits)
        monitor.update_status(
            status="running",
            phase="training",
            train_patients=train_count,
            val_patients=len(val_cache.base_logits),
            epoch=start_epoch,
            global_step=global_step,
        )
        for epoch in range(start_epoch, args.epochs):
            last_epoch = epoch
            epoch_start = time.perf_counter()
            adapter.train()
            order = torch.randperm(train_count)
            epoch_loss = 0.0
            epoch_observations = 0
            for start in range(0, train_count, args.train_batch_size):
                step_start = time.perf_counter()
                batch_index = order[start : start + args.train_batch_size].numpy()
                base = torch.as_tensor(train_cache.base_logits[batch_index], device=args.device)
                features = torch.as_tensor(train_cache.pgs_features[batch_index], device=args.device)
                available = torch.as_tensor(
                    train_cache.pgs_available[batch_index], dtype=torch.float32, device=args.device
                )
                labels = torch.as_tensor(train_cache.labels[batch_index], device=args.device)
                mask = torch.as_tensor(train_cache.mask[batch_index], dtype=torch.float32, device=args.device)
                fused, delta, gate = adapter(base, features, available)
                loss = masked_loss(fused, labels, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                optimizer.step()
                batch_size = len(batch_index)
                epoch_loss += float(loss.detach().cpu()) * batch_size
                epoch_observations += batch_size
                global_step += 1
                if global_step % args.log_every == 0:
                    step_time = time.perf_counter() - step_start
                    monitor.log_step(
                        global_step,
                        {
                            "loss/train_total": loss,
                            "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                            "optimization/gradient_norm": grad_norm,
                            "model/pgs_gate_mean": gate.mean(),
                            "representation_norm/pgs_delta": torch.linalg.vector_norm(
                                delta.flatten(1), dim=1
                            ).mean(),
                            "performance/step_time_seconds": step_time,
                            "performance/samples_per_second": batch_size / max(step_time, 1e-9),
                            **monitor.system_metrics(),
                        },
                    )

            fused_val = adapter_logits(adapter, val_cache, args.device, args.train_batch_size)
            base_metrics = prediction_metrics(val_cache.base_logits, val_cache.labels, val_cache.mask)
            fused_metrics = prediction_metrics(fused_val, val_cache.labels, val_cache.mask)
            selection_metric = fused_metrics["auprc_macro"]
            improved = selection_metric > best_metric
            if improved:
                best_metric = selection_metric
                best_epoch = epoch
                patience_used = 0
            else:
                patience_used += 1
            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": epoch_loss / max(epoch_observations, 1),
                "val_base_loss": masked_numpy_loss(val_cache.base_logits, val_cache.labels, val_cache.mask),
                "val_fused_loss": masked_numpy_loss(fused_val, val_cache.labels, val_cache.mask),
                "val_base_auroc": base_metrics["auroc_macro"],
                "val_fused_auroc": fused_metrics["auroc_macro"],
                "val_base_auprc": base_metrics["auprc_macro"],
                "val_fused_auprc": fused_metrics["auprc_macro"],
                "val_base_brier": base_metrics["brier_macro"],
                "val_fused_brier": fused_metrics["brier_macro"],
                "val_fused_capture": fused_metrics["top_decile_capture_macro"],
                "pgs_gate_mean": float(torch.sigmoid(adapter.gate_logits).mean().detach().cpu()),
                "epoch_time_seconds": time.perf_counter() - epoch_start,
                "best_val_auprc": best_metric,
                "best_epoch": best_epoch,
                "patience_used": patience_used,
            }
            monitor.log_epoch(epoch_metrics)
            monitor.update_status(status="running", phase="training", **epoch_metrics, **monitor.system_metrics())
            state = {
                "adapter": adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None,
                "scaler": None,
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "patience_used": patience_used,
                "config": config,
                "random_state": monitor.capture_random_state(),
            }
            monitor.save_checkpoint(state, "last.pt")
            if improved:
                monitor.save_checkpoint(state, "best_auprc.pt")
            print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
            if patience_used >= args.patience:
                break

        monitor.mark_finished(
            phase="finished",
            epoch=last_epoch,
            global_step=global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
    except BaseException as exc:
        monitor.mark_failed(exc, epoch=last_epoch, global_step=global_step)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
