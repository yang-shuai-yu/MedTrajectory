"""Deterministic Track R v2.2 pretraining-fidelity evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory
from semantic_delphi_ukb.track_r_batch import (
    load_track_r_assets,
    load_track_r_bos_token_id,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol
from semantic_delphi_ukb.track_r_rows import write_json_gzip
from semantic_delphi_ukb.track_r_v2_2 import (
    deterministic_patient_batch,
    event_time_nll_rows,
    load_wavelength_manifest,
)
from utils import get_p2i


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--model-name", required=True)
    value.add_argument("--split", choices=("train", "val"), default="val")
    value.add_argument("--out-dir", type=Path, required=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--max-patients", type=int, default=0)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _topk_hits(logits: torch.Tensor, target: int, ignored: list[int]) -> dict[str, bool]:
    filtered = logits.clone()
    filtered[ignored] = -torch.inf
    order = torch.topk(filtered, k=min(10, filtered.numel())).indices.tolist()
    return {f"hit_at_{k}": int(target in order[:k]) for k in (1, 5, 10)}


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[list[dict], dict]:
    protocol = load_track_r_protocol(args.protocol)
    if protocol["protocol_id"] != "track_r_v2_2":
        raise ValueError("pretraining-fidelity evaluator requires Track R v2.2")
    validate_track_r_data_manifest(args.data_dir, protocol)
    wavelength_contract = protocol["wavelength_contract"]
    if not wavelength_contract.get("expected_sha256"):
        raise ValueError("v2.2 evaluation requires wavelength_contract.expected_sha256")
    manifest_path = Path(wavelength_contract["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    wavelength_manifest = load_wavelength_manifest(
        manifest_path,
        expected_sha256=wavelength_contract.get("expected_sha256"),
        scale_factor_bounds=wavelength_contract["scale_factor_bounds"],
        require_log_scale_match=bool(wavelength_contract["require_manifest_log_scale_match"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get("stage") != "pretraining":
        raise ValueError("main NLL must use a pretraining checkpoint")
    if checkpoint.get("protocol_manifest_sha256") != protocol["protocol_manifest_sha256"]:
        raise ValueError("checkpoint was not trained under this frozen v2.2 protocol manifest")
    model = CARoPEHorizonMedTrajectory(CARoPEConfig(**checkpoint["model_args"]))
    if model.config.age_rope_variant == "additive_v2_2":
        checkpoint_wavelengths = tuple(float(value) for value in model.config.rope_wavelengths_days)
        manifest_wavelengths = tuple(float(value) for value in wavelength_manifest["wavelengths_days_head_major"])
        if checkpoint_wavelengths != manifest_wavelengths:
            raise ValueError("checkpoint RoPE wavelengths differ from the frozen manifest")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    if abs(float(model.config.t_min) - float(protocol["pretraining_fidelity"]["t_min"])) > 1e-12:
        raise ValueError("checkpoint t_min differs from the frozen v2.2 evaluator contract")

    data = np.memmap(args.data_dir / f"{args.split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    prefix_ids, anchor_ages = load_track_r_assets(args.data_dir, args.split)
    patient_count = len(p2i) if args.max_patients <= 0 else min(len(p2i), args.max_patients)
    bos_token_id = load_track_r_bos_token_id(args.data_dir)
    rows: list[dict] = []
    patient_means = []
    diagnostic_values = [[] for _ in model.transformer.h]
    ignored = list(model.config.ignore_tokens) + [1]
    atol = float(protocol["pretraining_fidelity"]["nll_equivalence_atol"])
    diagnostic_patient_count = int(protocol["mechanism_gate"]["diagnostic_patient_prefix_count"])

    for patient_index in range(patient_count):
        batch = deterministic_patient_batch(
            data,
            p2i,
            prefix_ids,
            anchor_ages,
            patient_index,
            dynamic_context_length=int(protocol["dynamic_context_length"]),
            bos_token_id=bos_token_id,
            device=args.device,
        )
        if not batch:
            continue
        model.set_additive_diagnostics_capture(patient_index < diagnostic_patient_count)
        logits, parts, *_ = model(
            batch["x"],
            batch["age"],
            None,
            batch["targets"],
            batch["targets_age"],
            validation_loss_mode=True,
            static_token_mask=batch["static_mask"],
            bos_token_mask=batch["bos_mask"],
            next_event_mask=batch["next_event_mask"],
            time_loss_mask=batch["time_loss_mask"],
        )
        if patient_index < diagnostic_patient_count:
            for layer_index, block in enumerate(model.transformer.h):
                values = block.attn.last_additive_relative_values
                if values is not None and values.numel():
                    diagnostic_values[layer_index].append(values)
        attn_mask = model.build_track_r_attention_mask(
            batch["x"], batch["age"], batch["targets_age"], batch["static_mask"]
        )
        nll, time_valid = event_time_nll_rows(
            logits,
            batch["x"],
            batch["age"],
            batch["targets"],
            batch["targets_age"],
            attn_mask,
            ignore_tokens=model.config.ignore_tokens,
            t_min=model.config.t_min,
            mask_ties=model.config.mask_ties,
            next_event_mask=batch["next_event_mask"],
            time_loss_mask=batch["time_loss_mask"],
        )
        if not bool(time_valid.any()):
            continue
        evaluator_mean = nll[time_valid].mean()
        if not torch.isclose(evaluator_mean, parts["loss_dt"], rtol=0.0, atol=atol):
            raise AssertionError(
                f"patient {patient_index}: evaluator NLL differs from loss_dt by more than atol={atol}"
            )
        patient_means.append(float(evaluator_mean.cpu()))
        for local, source_ordinal, target_ordinal in zip(
            batch["local_positions"], batch["source_ordinals"], batch["target_ordinals"]
        ):
            local = int(local)
            if not bool(time_valid.view_as(batch["x"])[0, local]):
                continue
            source_age = float(batch["age"][0, local].cpu())
            target_age = float(batch["targets_age"][0, local].cpu())
            target_token = int(batch["targets"][0, local].cpu())
            row = {
                "patient_index": patient_index,
                "original_source_event_ordinal": int(source_ordinal),
                "original_target_event_ordinal": int(target_ordinal),
                "context_window_start": int(batch["context_window_start"]),
                "local_source_position": local,
                "source_token_id": int(batch["x"][0, local].cpu()),
                "source_age_days": source_age,
                "target_token_id": target_token,
                "target_age_days": target_age,
                "positive_gap_days": target_age - source_age,
                "nll": float(nll.view_as(batch["x"])[0, local].cpu()),
                **_topk_hits(logits[0, local], target_token, ignored),
            }
            rows.append(row)

    keys = [(
        row["patient_index"], row["original_source_event_ordinal"],
        row["original_target_event_ordinal"], row["context_window_start"],
        row["local_source_position"],
    ) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("event evaluation row key is not unique")
    rope_diagnostics = model.additive_rope_v2_2_diagnostics()
    relative_values = [value for layer in diagnostic_values for value in layer]
    relative_median = (
        float(torch.cat(relative_values).median()) if relative_values else None
    )
    mechanism = protocol["mechanism_gate"]
    rope_diagnostics.update({
        "diagnostic_patient_prefix_count": diagnostic_patient_count,
        "relative_logit_amplitude_median": relative_median,
        "scale_gate_passed": (
            rope_diagnostics["saturated_fraction"]
            < float(mechanism["maximum_saturated_fraction_exclusive"])
            if rope_diagnostics["saturated_fraction"] is not None else None
        ),
        "amplitude_gate_passed": (
            relative_median > float(mechanism["minimum_relative_logit_amplitude_exclusive"])
            if relative_median is not None else None
        ),
    })
    summary = {
        "protocol_id": protocol["protocol_id"],
        "model_name": args.model_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "patient_count": patient_count,
        "patients_with_positive_gap": len(patient_means),
        "event_row_count": len(rows),
        "patient_mean_nll": float(np.mean(patient_means)) if patient_means else None,
        "event_mean_nll": float(np.mean([row["nll"] for row in rows])) if rows else None,
        "hit_at_1": float(np.mean([row["hit_at_1"] for row in rows])) if rows else None,
        "hit_at_5": float(np.mean([row["hit_at_5"] for row in rows])) if rows else None,
        "hit_at_10": float(np.mean([row["hit_at_10"] for row in rows])) if rows else None,
        "nll_equivalence_atol": atol,
        "window_select": "right",
        "padding": "none",
        "no_event_token_rate": 0,
        "additive_rope_diagnostics": rope_diagnostics,
    }
    return rows, summary


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    rows, summary = evaluate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = write_json_gzip(args.out_dir / "rows.json.gz", rows)
    summary["rows_sha256"] = _file_sha256(rows_path)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "status.json").write_text(
        json.dumps({"status": "finished", "event_row_count": len(rows)}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
