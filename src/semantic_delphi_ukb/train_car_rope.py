"""Validation-only training entrypoint for the CARoPE candidate.

This entrypoint is deliberately separate from the frozen paper_protocol_v1
training scripts. It writes only to the requested CARoPE run directory and
does not read or create a locked test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    STATIC_CONDITIONING_MODES,
    get_track_r_batch,
    load_track_r_assets,
    load_track_r_bos_token_id,
    load_track_r_static_features,
    track_r_static_dim,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402
from semantic_delphi_ukb.track_r_v2_2 import load_wavelength_manifest  # noqa: E402
from semantic_delphi_ukb.paper_run import cosine_warmup_scheduler, seed_everything, write_source_manifest  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    binary_auc,
    build_horizon_targets,
    load_followup_end_ages,
)
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    disease_specs_payload,
    load_selected_disease_token_groups,
)
from training_monitor import RunMonitor  # noqa: E402
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CARoPE on train and monitor validation only.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data/paper_protocol_v1/multitype")
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-from-ckpt", type=Path, default=None,
                        help="CARoPE pretraining checkpoint; must use the same CARoPE model_args")
    parser.add_argument(
        "--init-from-a0-ckpt",
        type=Path,
        default=None,
        help="upgrade a frozen A0-TokenStatic horizon checkpoint with residual-RoPE parameters",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--next-event-loss-weight", type=float, default=0.2)
    parser.add_argument("--horizon-risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--time-gap-loss-weight", type=float, default=0.2)
    parser.add_argument("--use-age-encoding", choices=("true", "false"), default="false")
    parser.add_argument("--use-age-rope", choices=("true", "false"), default="true")
    parser.add_argument("--use-relative-horizon-query", choices=("true", "false"), default="true")
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--rope-scales", default="0.25,1,4")
    parser.add_argument("--rope-initial-gate", type=float, default=0.1)
    parser.add_argument("--age-rope-variant", choices=("legacy", "additive_v2_2"), default="legacy")
    parser.add_argument("--rope-wavelengths-manifest", type=Path, default=None)
    parser.add_argument("--rope-max-scale", type=float, default=2.0)
    parser.add_argument("--residual-rope-mode", choices=("none", "learned", "fixed"), default="none")
    parser.add_argument("--residual-rope-initial-alpha", type=float, default=0.01)
    parser.add_argument("--residual-rope-fixed-alpha", type=float, default=1.0)
    parser.add_argument("--residual-rope-phase-origin", choices=("recruitment_age",), default="recruitment_age")
    parser.add_argument("--horizons", default="1,5")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--track-r-protocol", type=Path, default=None)
    parser.add_argument("--include-static-prefix", choices=("true", "false"), default="false")
    parser.add_argument("--static-conditioning", choices=STATIC_CONDITIONING_MODES, default="none")
    parser.add_argument(
        "--static-fusion-stage",
        choices=("pre_transformer", "post_transformer"),
        default="pre_transformer",
    )
    return parser


def parse_horizons(value: str) -> tuple[float, ...]:
    horizons = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("--horizons must contain positive comma-separated years")
    if tuple(sorted(horizons)) != horizons:
        raise ValueError("--horizons must be sorted in ascending order")
    return horizons


def parse_positive_floats(value: str, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive comma-separated values")
    return values


def load_rope_wavelengths(
    path: Optional[Path],
    wavelength_contract: Optional[Mapping] = None,
    *,
    require_expected: bool = False,
) -> tuple[float, ...]:
    if path is None:
        return ()
    contract = dict(wavelength_contract or {})
    expected_sha256 = contract.get("expected_sha256")
    if require_expected and not expected_sha256:
        raise ValueError("frozen v2.2 training requires wavelength_contract.expected_sha256")
    manifest = load_wavelength_manifest(
        path,
        expected_sha256=expected_sha256,
        scale_factor_bounds=contract.get("scale_factor_bounds"),
        require_log_scale_match=bool(contract.get("require_manifest_log_scale_match", False)),
    )
    return tuple(float(value) for value in manifest["wavelengths_days_head_major"])


def resolve_diseases_yaml(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    for candidate in (REPO_DIR / "selected_diseases.yaml", REPO_DIR / "docs/selected_diseases.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("selected disease YAML was not found")


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i, static = p2i[:max_patients], static[:max_patients]
    return data, p2i, static


def make_model(args: argparse.Namespace, data_dir: Path, num_diseases: int) -> CARoPEHorizonMedTrajectory:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(data_dir / manifest["semantic_output"]).astype(np.float32)
    n_embd = int(args.n_embd or embeddings.shape[1])
    config = CARoPEConfig(
        block_size=args.block_size,
        vocab_size=int(embeddings.shape[0]),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=n_embd,
        semantic_embedding_dim=int(embeddings.shape[1]),
        dropout=args.dropout,
        static_dim=(
            track_r_static_dim(data_dir, args.static_conditioning)
            if args.track_r_protocol is not None
            else len(manifest["static_feature_order"])
        ),
        static_hidden_dim=n_embd,
        static_fusion_stage=args.static_fusion_stage,
        num_tte_tasks=num_diseases,
        num_horizons=len(args.horizons),
        time_gap_loss_weight=args.time_gap_loss_weight,
        horizon_risk_loss_weight=args.horizon_risk_loss_weight,
        horizon_years=args.horizons,
        use_age_encoding=args.use_age_encoding == "true",
        use_age_rope=args.use_age_rope == "true",
        use_relative_horizon_query=args.use_relative_horizon_query == "true",
        rope_base=args.rope_base,
        rope_scales=args.rope_scales,
        rope_initial_gate=args.rope_initial_gate,
        age_rope_variant=args.age_rope_variant,
        rope_wavelengths_days=getattr(args, "rope_wavelengths_days", ()),
        rope_max_scale=args.rope_max_scale,
        residual_rope_mode=args.residual_rope_mode,
        residual_rope_initial_alpha=args.residual_rope_initial_alpha,
        residual_rope_fixed_alpha=args.residual_rope_fixed_alpha,
        residual_rope_phase_origin=args.residual_rope_phase_origin,
    )
    return CARoPEHorizonMedTrajectory(config, pretrained_token_embeddings=embeddings)


def build_training_batch(
    ix,
    data,
    p2i,
    static,
    args,
    *,
    select: str,
    padding: str,
    prefix_ids=None,
    anchor_ages=None,
):
    if args.track_r_protocol is None:
        x, age, y, target_age, static_features = get_batch(
            ix,
            data,
            p2i,
            static,
            block_size=args.block_size,
            device=args.device,
            padding=padding,
            select=select,
            no_event_token_rate=args.no_event_token_rate,
            cut_batch=True,
        )
        return x, age, y, target_age, static_features, None, None, None, None, None
    return get_track_r_batch(
        ix,
        data,
        p2i,
        static,
        prefix_ids,
        anchor_ages,
        include_static_prefix=args.include_static_prefix == "true",
        dynamic_context_length=args.dynamic_context_length,
        device=args.device,
        select=select,
        padding=padding,
        no_event_token_rate=args.no_event_token_rate,
        cut_batch=True,
        bos_token_id=args.track_r_bos_token_id,
        static_conditioning=args.static_conditioning,
    )


def model_size_payload(model: CARoPEHorizonMedTrajectory) -> dict:
    groups = {
        "token_embedding": sum(p.numel() for p in model.transformer.wte.parameters()),
        "token_input_projection": sum(p.numel() for p in model.token_input_proj.parameters()),
        "transformer_blocks": sum(p.numel() for p in model.transformer.h.parameters()),
    }
    total = sum(p.numel() for p in model.parameters())
    groups["other"] = total - sum(groups.values())
    return {
        "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": total,
        "parameter_groups": groups,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "semantic_embedding_dim": model.config.semantic_embedding_dim or model.config.n_embd,
    }


def write_model_size_manifest(run_dir: Path, model: CARoPEHorizonMedTrajectory) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_size.json").write_text(
        json.dumps(model_size_payload(model), indent=2), encoding="utf-8"
    )


def residual_rope_metrics(model: CARoPEHorizonMedTrajectory) -> dict[str, float]:
    diagnostics = model.residual_rope_diagnostics()
    layers = diagnostics["layers"]
    if not layers:
        return {}
    alpha = np.asarray([value for layer in layers for value in layer["alpha_by_head"]], dtype=np.float64)
    scales = np.asarray(
        [value for layer in layers for head in layer["scale_by_head_and_frequency"] for value in head],
        dtype=np.float64,
    )
    return {
        "residual_rope_alpha_mean": float(alpha.mean()),
        "residual_rope_alpha_min": float(alpha.min()),
        "residual_rope_alpha_max": float(alpha.max()),
        "residual_rope_scale_mean": float(scales.mean()),
        "residual_rope_scale_min": float(scales.min()),
        "residual_rope_scale_max": float(scales.max()),
    }


def write_residual_rope_diagnostics(run_dir: Path, model: CARoPEHorizonMedTrajectory) -> None:
    diagnostics = model.residual_rope_diagnostics()
    if diagnostics["layers"]:
        (run_dir / "residual_rope_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )


def write_additive_rope_diagnostics(run_dir: Path, model: CARoPEHorizonMedTrajectory) -> None:
    diagnostics = model.additive_rope_v2_2_diagnostics()
    if diagnostics["layers"]:
        (run_dir / "additive_rope_v2_2_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )


def load_frozen_a0_base(
    model: CARoPEHorizonMedTrajectory,
    checkpoint_path: Path,
    device: str,
) -> dict:
    if model.config.residual_rope_mode == "none":
        raise ValueError("--init-from-a0-ckpt requires residual RoPE mode")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_config = checkpoint.get("model_args")
    if source_config is None:
        raise ValueError("A0 checkpoint is missing model_args")
    expected = {
        "use_age_encoding": True,
        "use_age_rope": False,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "block_size": model.config.block_size,
        "vocab_size": model.config.vocab_size,
        "num_horizons": model.config.num_horizons,
        "num_tte_tasks": model.config.num_tte_tasks,
        "use_relative_horizon_query": model.config.use_relative_horizon_query,
    }
    mismatches = {
        key: {"expected": value, "checkpoint": source_config.get(key)}
        for key, value in expected.items()
        if source_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"A0 checkpoint contract mismatch: {mismatches}")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        name
        for name, _ in model.named_parameters()
        if ".residual_rope." in name or name.endswith(".residual_gate_logit")
    }
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            "A0 upgrade changed non-residual parameters: "
            f"missing={missing}, allowed_missing={sorted(allowed_missing)}, unexpected={unexpected}"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in allowed_missing
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A0 upgrade left no trainable residual parameters")
    return checkpoint


def _sequence_targets(labels: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor, seq_len: int):
    batch, horizons, diseases = labels.shape
    events = torch.zeros((batch, seq_len, horizons, diseases), device=labels.device)
    masks = torch.zeros_like(events)
    rows = torch.arange(batch, device=labels.device)
    valid_pos = positions.clamp(0, seq_len - 1)
    events[rows, valid_pos] = labels
    masks[rows, valid_pos] = mask
    return events, masks


def compose_training_loss(parts, next_event_weight: float, horizon_weight: float, gap_weight: float):
    next_loss = parts.get("loss_ce", 0.0) + parts.get("loss_dt", 0.0)
    reference = next(value for value in parts.values() if torch.is_tensor(value))
    zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
    gap_loss = parts.get("loss_gap", zero)
    risk_loss = parts.get("loss_horizon", zero)
    return next_event_weight * next_loss + gap_weight * gap_loss + horizon_weight * risk_loss


def _batch_loss(model, batch, patient_ages, patient_last, horizons, args, validation_loss_mode=False):
    (
        ix, x, age, y, target_age, static, static_mask, bos_mask,
        next_event_mask, time_loss_mask, static_feature_mask,
    ) = batch
    horizon_y = y if bos_mask is None else y.masked_fill(static_mask | bos_mask, -1)
    labels, mask, _ = build_horizon_targets(
        ix, x, age, horizon_y, patient_ages, patient_last, horizons, args.device, censor_aware=True
    )
    _, positions = last_prediction_positions(x, horizon_y)
    horizon_event, horizon_mask = _sequence_targets(labels, mask, positions, x.size(1))
    logits, parts, _, _, risk_logits = model(
        x, age, static, y, target_age, validation_loss_mode=validation_loss_mode,
        horizon_event=horizon_event, horizon_mask=horizon_mask,
        static_token_mask=static_mask, next_event_mask=next_event_mask, time_loss_mask=time_loss_mask,
        static_feature_mask=static_feature_mask,
        bos_token_mask=bos_mask,
    )
    if parts is None:
        raise RuntimeError("CARoPE produced no loss for a non-empty training batch")
    risk_loss = parts.get("loss_horizon", torch.zeros((), device=x.device))
    # The auxiliary gap coefficient is applied once here. This keeps A3's
    # configured 0.2 contribution at 0.2 rather than 0.2 * next_event_weight.
    loss = compose_training_loss(
        parts,
        next_event_weight=args.next_event_loss_weight,
        horizon_weight=args.horizon_risk_loss_weight,
        gap_weight=model.config.time_gap_loss_weight,
    )
    return loss, risk_loss, logits, risk_logits, labels, mask, positions


@torch.no_grad()
def evaluate(model, data, p2i, static, patient_ages, patient_last, horizons, args, prefix_ids=None, anchor_ages=None):
    model.eval()
    losses, risks, labels, masks = [], [], [], []
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        batch = (ix, *build_training_batch(
            ix, data, p2i, static, args, padding="regular", select="random",
            prefix_ids=prefix_ids, anchor_ages=anchor_ages,
        ))
        loss, risk_loss, _, risk_logits, y_true, mask, positions = _batch_loss(
            model, batch, patient_ages, patient_last, horizons, args, validation_loss_mode=True
        )
        rows = torch.arange(len(ix), device=args.device)
        risks.append(torch.sigmoid(risk_logits[rows, positions]).cpu().numpy())
        labels.append(y_true.cpu().numpy())
        masks.append(mask.cpu().numpy().astype(bool))
        losses.append(float(loss.cpu()))
    model.train()
    score = np.concatenate(risks, axis=0)
    target = np.concatenate(labels, axis=0)
    observed = np.concatenate(masks, axis=0)
    aucs = []
    for h_idx in range(score.shape[1]):
        for d_idx in range(score.shape[2]):
            valid = observed[:, h_idx, d_idx] & np.isfinite(score[:, h_idx, d_idx])
            y = target[valid, h_idx, d_idx]
            if len(y) and 0 < int(y.sum()) < len(y):
                aucs.append(binary_auc(score[valid, h_idx, d_idx], y))
    return {"val_loss": float(np.mean(losses)), "val_horizon_auc_mean": float(np.mean(aucs)) if aucs else float("nan")}


def self_test(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    if isinstance(args.horizons, str):
        args.horizons = parse_horizons(args.horizons)
    config = CARoPEConfig(block_size=8, vocab_size=32, n_layer=2, n_head=4, n_embd=32,
                          static_dim=1, static_hidden_dim=32, num_tte_tasks=3,
                          num_horizons=len(args.horizons), horizon_years=args.horizons)
    model = CARoPEHorizonMedTrajectory(config).to(args.device)
    x = torch.randint(2, 32, (4, 8), device=args.device)
    age = torch.sort(torch.rand(4, 8, device=args.device) * 30000, dim=1).values
    y = torch.randint(2, 32, (4, 8), device=args.device)
    target_age = age + 30
    static = torch.randn(4, 1, device=args.device)
    horizon_event = torch.zeros(4, 8, len(args.horizons), 3, device=args.device)
    horizon_mask = torch.zeros_like(horizon_event)
    horizon_event[:, -1] = torch.randint(0, 2, (4, len(args.horizons), 3), device=args.device).float()
    horizon_mask[:, -1] = 1.0
    _, parts, _, _, _ = model(
        x, age, static, y, target_age, horizon_event=horizon_event, horizon_mask=horizon_mask
    )
    loss = compose_training_loss(
        parts,
        next_event_weight=args.next_event_loss_weight,
        horizon_weight=args.horizon_risk_loss_weight,
        gap_weight=model.config.time_gap_loss_weight,
    )
    loss.backward()
    print(json.dumps({"self_test": True, "loss_finite": bool(torch.isfinite(loss)), "horizons": args.horizons}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.rope_scales = parse_positive_floats(args.rope_scales, "--rope-scales")
    args.rope_wavelengths_days = ()
    if args.self_test:
        return self_test(args)
    args.horizons = parse_horizons(args.horizons)
    args.dynamic_context_length = args.block_size
    protocol = None
    if args.track_r_protocol is not None:
        protocol = load_track_r_protocol(args.track_r_protocol)
        args.track_r_protocol_sha256 = protocol.get("protocol_manifest_sha256")
        validate_track_r_data_manifest(args.data_dir, protocol)
        args.dynamic_context_length = int(protocol["dynamic_context_length"])
        args.track_r_bos_token_id = load_track_r_bos_token_id(args.data_dir)
        args.block_size = args.dynamic_context_length + 1 + (
            int(protocol["static_prefix"]["fixed_length"]) if args.include_static_prefix == "true" else 0
        )
        if args.static_conditioning != "none" and args.include_static_prefix == "true":
            raise ValueError("static residual conditioning cannot be combined with static prefix tokens")
    elif args.static_conditioning != "none":
        raise ValueError("--static-conditioning is only valid with --track-r-protocol")
    args.rope_wavelengths_days = load_rope_wavelengths(
        args.rope_wavelengths_manifest,
        protocol.get("wavelength_contract") if protocol is not None else None,
        require_expected=args.age_rope_variant == "additive_v2_2",
    )
    if args.init_from_ckpt is not None and args.init_from_a0_ckpt is not None:
        raise ValueError("--init-from-ckpt and --init-from-a0-ckpt are mutually exclusive")
    if args.residual_rope_mode != "none":
        if args.track_r_protocol is None or args.include_static_prefix != "true":
            raise ValueError("residual RoPE is registered only for Track R TokenStatic")
        if args.use_age_encoding != "true" or args.use_age_rope != "false":
            raise ValueError("residual RoPE requires A0 Sin/Cos base and disables legacy inner RoPE")
        if args.init_from_a0_ckpt is None:
            raise ValueError("residual RoPE training must upgrade the frozen A0 checkpoint")
    if args.age_rope_variant == "additive_v2_2":
        if args.track_r_protocol is None or args.include_static_prefix != "true":
            raise ValueError("additive v2.2 RoPE is registered only for Track R TokenStatic")
        if not args.rope_wavelengths_days:
            raise ValueError("additive v2.2 RoPE requires --rope-wavelengths-manifest")
    seed_everything(args.seed)
    if args.run_dir.exists() and any(p.name != "stdout.log" for p in args.run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    diseases, token_groups = load_selected_disease_token_groups(resolve_diseases_yaml(args.diseases_yaml), args.data_dir)
    train_data, train_p2i, train_static = load_split(args.data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(args.data_dir, "val", args.max_patients)
    train_prefix = train_anchor = val_prefix = val_anchor = None
    if protocol is not None:
        train_prefix, train_anchor = load_track_r_assets(args.data_dir, "train", args.max_patients)
        val_prefix, val_anchor = load_track_r_assets(args.data_dir, "val", args.max_patients)
        train_static = load_track_r_static_features(
            args.data_dir, "train", args.static_conditioning, args.max_patients
        )
        val_static = load_track_r_static_features(
            args.data_dir, "val", args.static_conditioning, args.max_patients
        )
    model = make_model(args, args.data_dir, len(diseases)).to(args.device)
    a0_checkpoint = None
    if args.init_from_a0_ckpt is not None:
        a0_checkpoint = load_frozen_a0_base(model, args.init_from_a0_ckpt, args.device)
    elif args.init_from_ckpt is not None:
        checkpoint = torch.load(args.init_from_ckpt, map_location=args.device, weights_only=False)
        state_dict = dict(checkpoint["model"])
        current_state = model.state_dict()
        skipped = []
        for key in list(state_dict.keys()):
            if key in current_state and tuple(state_dict[key].shape) != tuple(current_state[key].shape):
                skipped.append(key)
                state_dict.pop(key)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected or set(missing) - set(skipped):
            raise RuntimeError(f"incompatible CARoPE pretraining checkpoint: missing={missing}, unexpected={unexpected}, skipped={skipped}")
    train_ages, train_last = build_patient_disease_ages(train_data, train_p2i, token_groups, model.config.vocab_size)
    val_ages, val_last = build_patient_disease_ages(val_data, val_p2i, token_groups, model.config.vocab_size)
    train_followup = load_followup_end_ages(args.data_dir, "train", train_last, args.max_patients)
    val_followup = load_followup_end_ages(args.data_dir, "val", val_last, args.max_patients)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), "cuda" if "cuda" in args.device else "cpu")
    scheduler = cosine_warmup_scheduler(
        optimizer, args.warmup_iters, args.max_iters, args.min_learning_rate / args.learning_rate
    )
    config = {**vars(args), "data_dir": str(args.data_dir), "model_args": model.config.__dict__.copy(), "diseases": disease_specs_payload(diseases), "stage": "risk"}
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    write_model_size_manifest(args.run_dir, model)
    write_source_manifest(args.run_dir, REPO_DIR, Path(__file__), extra_paths=(
        REPO_DIR / "src/semantic_delphi_ukb/car_rope_model.py",
        REPO_DIR / "src/semantic_delphi_ukb/train_car_rope.py",
        REPO_DIR / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
        REPO_DIR / "src/semantic_delphi_ukb/multitype_batch.py",
        REPO_DIR / "src/semantic_delphi_ukb/track_r_batch.py",
        REPO_DIR / "src/semantic_delphi_ukb/track_r_contract.py",
        REPO_DIR / "src/semantic_delphi_ukb/train_architecture_risk_heads.py",
        REPO_DIR / "src/semantic_delphi_ukb/tte_targets.py",
        REPO_DIR / "scripts/training_monitor.py",
    ))
    best_auc = -float("inf")
    last_iter = -1
    try:
        monitor.mark_running(phase="validation_training", iteration=0, global_step=0)
        for iteration in range(args.max_iters + 1):
            last_iter = iteration
            if iteration % args.eval_interval == 0:
                metrics = evaluate(
                    model, val_data, val_p2i, val_static, val_ages, val_followup, args.horizons, args,
                    prefix_ids=val_prefix, anchor_ages=val_anchor,
                )
                current_auc = metrics["val_horizon_auc_mean"]
                improved = bool(np.isfinite(current_auc) and current_auc > best_auc)
                if improved:
                    best_auc = current_auc
                row = {
                    "iteration": iteration,
                    **metrics,
                    **residual_rope_metrics(model),
                    "best_val_horizon_auc_mean": best_auc,
                }
                monitor.log_epoch(row)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "iteration": iteration,
                         "model_args": model.config.__dict__.copy(),
                         "model_family": (
                             "CARoPE_additive_v2_2" if model.config.age_rope_variant == "additive_v2_2"
                             else "CARoPE_residual_v1" if model.config.residual_rope_mode != "none"
                             else "CARoPE_v1"
                         ),
                         "frozen_a0_source": str(args.init_from_a0_ckpt) if a0_checkpoint is not None else None,
                         "stage": "risk",
                         "protocol_manifest_sha256": getattr(args, "track_r_protocol_sha256", None),
                         "config": config}
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_horizon_auc.pt")
                monitor.update_status(status="running", **row)
                print(json.dumps(row, sort_keys=True), flush=True)
                write_residual_rope_diagnostics(args.run_dir, model)
                write_additive_rope_diagnostics(args.run_dir, model)
            if iteration == args.max_iters:
                break
            ix = torch.randint(len(train_p2i), (args.batch_size,))
            batch = (ix, *build_training_batch(
                ix, train_data, train_p2i, train_static, args, padding="regular", select="random",
                prefix_ids=train_prefix, anchor_ages=train_anchor,
            ))
            loss, risk_loss, _, _, _, _, _ = _batch_loss(model, batch, train_ages, train_followup, args.horizons, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if iteration % args.log_every == 0:
                monitor.log_step(iteration, {"loss/train_total": loss, "loss/train_horizon": risk_loss,
                                              "optimization/gradient_norm": grad_norm, **monitor.system_metrics()})
        monitor.mark_finished(phase="finished", iteration=last_iter, global_step=last_iter, best_val_horizon_auc_mean=best_auc)
    except BaseException as exc:
        monitor.mark_failed(exc, iteration=last_iter, global_step=last_iter)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
