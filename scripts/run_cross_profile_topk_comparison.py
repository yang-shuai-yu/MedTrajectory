#!/usr/bin/env python3
"""Run paired LM disease Top-K evaluation on shared patient-age-disease targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.calibration_auc import select_shared_age_landmarks  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import build_official_left_batch  # noqa: E402
from semantic_delphi_ukb.evaluate_expanded_disease_topk import (  # noqa: E402
    build_disease_membership,
    load_medtrajectory,
    load_token_event_types,
    load_split,
)
from semantic_delphi_ukb.tte_targets import load_token_codes  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def fixed_padding_matrix(manifest: dict[str, Any], patient_count: int) -> np.ndarray:
    entries = defaultdict(list)
    for entry in manifest["landmarks"]:
        entries[int(entry["patient_index"])].append(entry)
    count = max((len(values) for values in entries.values()), default=0)
    matrix = np.full((patient_count, count), -10000.0, dtype=np.float32)
    for patient_id, values in entries.items():
        if patient_id >= patient_count:
            continue
        for index, entry in enumerate(values):
            matrix[patient_id, index] = float(entry["target_age_days"])
    return matrix


def manifest_targets(
    canonical_data: np.ndarray,
    canonical_p2i: np.ndarray,
    canonical_codes: dict[int, str],
    canonical_event_types: list[str],
    canonical_membership: dict[int, list[int]],
    diseases: list[Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = []
    by_patient = defaultdict(list)
    for entry in manifest["landmarks"]:
        by_patient[int(entry["patient_index"])].append(entry)
    for patient_id, entries in by_patient.items():
        if patient_id >= len(canonical_p2i):
            continue
        start, length = map(int, canonical_p2i[patient_id])
        segment = canonical_data[start : start + length]
        for entry in entries:
            target_age = float(entry["target_age_days"])
            for raw_row in segment:
                if float(raw_row[1]) <= target_age + 0.25:
                    continue
                model_token = int(raw_row[2]) + 1
                if model_token >= len(canonical_event_types) or canonical_event_types[model_token] != "diagnosis":
                    continue
                if model_token not in canonical_membership:
                    continue
                code = canonical_codes.get(model_token, "")
                for disease_idx in canonical_membership[model_token]:
                    targets.append({
                        "patient_index": patient_id,
                        "age_start_years": float(entry["age_start_years"]),
                        "target_age_days": target_age,
                        "target_code": code,
                        "disease_idx": int(disease_idx),
                        "disease_id": diseases[int(disease_idx)].disease_id,
                    })
                break
    return targets


def model_rows(
    model_spec: dict[str, Any],
    profile_dir: Path,
    split: str,
    manifest: dict[str, Any],
    targets: list[dict[str, Any]],
    diseases_yaml: Path,
    block_size: int,
    batch_size: int,
    topk: list[int],
    device: str,
) -> list[dict[str, Any]]:
    data, p2i, static = load_split(profile_dir, split, 0)
    vocab_path = profile_dir / "vocab" / "dynamic_token_vocab.csv"
    token_codes = load_token_codes(profile_dir)
    code_to_token = {code: int(token_id) for token_id, code in token_codes.items() if code}
    model, checkpoint = load_medtrajectory(Path(model_spec["checkpoint"]), device)
    model.eval()
    vocab_size = int(model.config.vocab_size)
    event_types = load_token_event_types(vocab_path, vocab_size)
    _, _, token_sets, _ = build_disease_membership(profile_dir, diseases_yaml)
    diagnosis_mask = torch.tensor(
        [idx < len(event_types) and event_types[idx] == "diagnosis" for idx in range(vocab_size)],
        dtype=torch.bool,
        device=device,
    )
    fixed_padding = fixed_padding_matrix(manifest, len(p2i))
    patient_ids, full_batch = build_official_left_batch(
        data, p2i, static, block_size, no_event_token_rate=5, fixed_padding_ages=fixed_padding
    )
    shared = defaultdict(list)
    for entry in manifest["landmarks"]:
        shared[int(entry["patient_index"])].append(entry)
    resolved = select_shared_age_landmarks(full_batch[0].numpy(), full_batch[1].numpy(), patient_ids, shared)
    index = {
        (int(pid), float(age)): (int(row), int(pos))
        for pid, age, row, pos in zip(
            resolved["patient_ids"], resolved["age_start_years"], resolved["row_indices"], resolved["positions"]
        )
    }
    target_lookup = defaultdict(list)
    keys_by_patient = defaultdict(list)
    for target in targets:
        key = (int(target["patient_index"]), float(target["age_start_years"]))
        profile_token = code_to_token.get(target["target_code"])
        if profile_token is None or profile_token >= len(event_types) or event_types[profile_token] != "diagnosis":
            continue
        target_lookup[key].append((target, int(profile_token)))
        if key not in keys_by_patient[int(target["patient_index"])]:
            keys_by_patient[int(target["patient_index"])].append(key)
    rows = []
    with torch.no_grad():
        for start in range(0, len(p2i), batch_size):
            stop = min(start + batch_size, len(p2i))
            x, age, y, target_age, static_batch = [tensor[start:stop].to(device) for tensor in full_batch]
            logits, _, _, _, _ = model(x, age, static_batch, y, target_age, validation_loss_mode=True)
            for local in range(stop - start):
                patient_id = start + local
                keys = [key for key in keys_by_patient.get(patient_id, []) if key in index]
                if not keys:
                    continue
                for key in keys:
                    _, pos = index[key]
                    final_logits = logits[local, pos].clone()
                    final_logits[~diagnosis_mask] = -torch.inf
                    values = torch.topk(final_logits, k=min(max(topk), int(diagnosis_mask.sum().item()))).indices.cpu().numpy().tolist()
                    for target, profile_token in target_lookup[key]:
                        disease_tokens = token_sets[int(target["disease_idx"])]
                        row = {
                            "model": model_spec["name"],
                            "profile": model_spec["profile"],
                            "split": split,
                            "patient_index": patient_id,
                            "age_start_years": key[1],
                            "target_age_days": target["target_age_days"],
                            "target_code": target["target_code"],
                            "disease_id": target["disease_id"],
                            "target_token": profile_token,
                        }
                        for k in topk:
                            row[f"exact_top{k}_hit"] = int(profile_token in values[:k])
                            row[f"group_top{k}_hit"] = int(bool(set(values[:k]) & set(disease_tokens)))
                        rows.append(row)
    return rows


def paired_bootstrap(
    rows_by_model: dict[str, list[dict[str, Any]]],
    comparisons: list[list[str]],
    topk: list[int],
    seed: int,
    reps: int,
) -> list[dict[str, Any]]:
    maps = {}
    for model, rows in rows_by_model.items():
        maps[model] = {
            (int(r["patient_index"]), float(r["age_start_years"]), r["target_code"], r["disease_id"]): r
            for r in rows
        }
    output = []
    for baseline, challenger in comparisons:
        common = sorted(set(maps[baseline]) & set(maps[challenger]))
        row_patients = np.asarray([key[0] for key in common], dtype=np.int64)
        patients, inverse = np.unique(row_patients, return_inverse=True)
        rng = np.random.default_rng(seed + sum(map(ord, challenger)))
        for metric in [f"exact_top{k}_hit" for k in topk] + [f"group_top{k}_hit" for k in topk]:
            delta = np.asarray([int(maps[challenger][key][metric]) - int(maps[baseline][key][metric]) for key in common], dtype=np.float64)
            patient_sums = np.bincount(inverse, weights=delta, minlength=len(patients))
            patient_counts = np.bincount(inverse, minlength=len(patients))
            boot = np.empty(reps, dtype=np.float64)
            for i in range(reps):
                sampled = rng.integers(0, len(patients), size=len(patients))
                boot[i] = float(patient_sums[sampled].sum() / patient_counts[sampled].sum())
            left = (float(np.sum(boot <= 0)) + 1.0) / (reps + 1.0)
            right = (float(np.sum(boot >= 0)) + 1.0) / (reps + 1.0)
            p = min(1.0, 2.0 * min(left, right))
            output.append({
                "comparison": f"{challenger}_minus_{baseline}",
                "metric": metric,
                "common_rows": len(common),
                "common_patients": len(patients),
                "delta_mean": float(delta.mean()) if len(delta) else math.nan,
                "ci95_low": float(np.quantile(boot, 0.025)) if len(delta) else math.nan,
                "ci95_high": float(np.quantile(boot, 0.975)) if len(delta) else math.nan,
                "p_two_sided": p if len(delta) else math.nan,
            })
    for metric in {row["metric"] for row in output}:
        rows = [row for row in output if row["metric"] == metric]
        order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided"])
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(rows) - rank) * rows[idx]["p_two_sided"])
            rows[idx]["p_holm"] = min(1.0, running)
    return output


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return int(row["patient_index"]), float(row["age_start_years"]), row["target_code"], row["disease_id"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_resolved_manifest(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    assets = {
        "config": config_path,
        "evaluator": Path(__file__).resolve(),
        "diseases_yaml": Path(config["diseases_yaml"]),
    }
    for split, path in config["split_manifests"].items():
        assets[f"{split}_landmarks"] = Path(path)
    for model in config["models"]:
        assets[f"{model['name']}_checkpoint"] = Path(model["checkpoint"])
    payload = {
        "protocol": config["protocol"],
        "analysis_status": config["analysis_status"],
        "target_definition": "first selected-panel diagnosis strictly after a shared patient-age landmark",
        "candidate_space": "profile-specific logits masked to diagnosis; target aligned by code_norm",
        "previous_topk_untouched": config["previous_topk_untouched"],
        "primary_results_untouched": config["primary_results_untouched"],
        "assets": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in assets.items()
        },
    }
    atomic_json(output_dir / "resolved_manifest.json", payload)


def write_markdown(summary: list[dict[str, Any]], output_dir: Path) -> None:
    wanted = {"exact_top1_hit", "exact_top5_hit", "exact_top10_hit", "group_top10_hit"}
    grouped = defaultdict(dict)
    for row in summary:
        grouped[(row["split"], row["model"], row["profile"])][row["metric"]] = row
    lines = [
        "# Shared Cross-Profile Top-K",
        "",
        "Status: `secondary_posthoc_cross_profile`",
        "",
        "| split | model | profile | targets | Exact@1 | Exact@5 | Exact@10 | Group@10 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (split, model, profile), metrics in grouped.items():
        if not wanted.issubset(metrics):
            continue
        lines.append(
            f"| {split} | {model} | {profile} | {metrics['exact_top1_hit']['n_targets']} | "
            f"{float(metrics['exact_top1_hit']['value']):.6f} | {float(metrics['exact_top5_hit']['value']):.6f} | "
            f"{float(metrics['exact_top10_hit']['value']):.6f} | {float(metrics['group_top10_hit']['value']):.6f} |"
        )
    (output_dir / "shared_topk_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_existing(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    rows_by_split = {}
    for split in config["split_manifests"]:
        rows_by_split[split] = {
            model["name"]: load_csv(output_dir / split / model["name"] / "shared_topk_raw.csv")
            for model in config["models"]
        }
    paired = []
    for split, rows_by_model in rows_by_split.items():
        for row in paired_bootstrap(rows_by_model, list(config["comparisons"]), list(config["topk"]), int(config["seed"]), int(config["bootstrap_replicates"])):
            row["split"] = split
            paired.append(row)
    write_csv(output_dir / "paired_bootstrap.csv", paired)
    summary = load_csv(output_dir / "shared_topk_summary.csv")
    write_markdown(summary, output_dir)
    write_resolved_manifest(config, config_path, output_dir)


def run(config: dict[str, Any], config_path: Path, output_dir: Path, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "completed_jobs": [], "errors": []}
    atomic_json(output_dir / "status.json", status)
    write_resolved_manifest(config, config_path, output_dir)
    diseases_yaml = Path(config["diseases_yaml"])
    canonical_dir = Path(config["profiles"][config["canonical_profile"]])
    models = config["models"]
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    all_summary = []
    pairing_audit = {"ok": True, "splits": {}}
    metrics_path = output_dir / "metrics.jsonl"
    for split, manifest_path_raw in config["split_manifests"].items():
        manifest_path = Path(manifest_path_raw)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        canonical_data, canonical_p2i, _ = load_split(canonical_dir, split, 0)
        canonical_codes = load_token_codes(canonical_dir)
        canonical_vocab = canonical_dir / "vocab" / "dynamic_token_vocab.csv"
        canonical_events = load_token_event_types(canonical_vocab, max(canonical_codes) + 1)
        diseases, _, _, canonical_to_diseases = build_disease_membership(canonical_dir, diseases_yaml)
        targets = manifest_targets(canonical_data, canonical_p2i, canonical_codes, canonical_events, canonical_to_diseases, diseases, manifest)
        split_key_sets = {}
        for model_spec in models:
            if model_spec["profile"] not in config["profiles"]:
                raise ValueError(f"unknown profile: {model_spec['profile']}")
            profile_dir = Path(config["profiles"][model_spec["profile"]])
            rows = model_rows(model_spec, profile_dir, split, manifest, targets, diseases_yaml, int(config["block_size"]), int(config["batch_size"]), list(config["topk"]), device)
            key = f"{split}/{model_spec['name']}"
            path = output_dir / split / model_spec["name"] / "shared_topk_raw.csv"
            write_csv(path, rows)
            rows_by_model[key] = rows
            split_key_sets[model_spec["name"]] = set(map(row_key, rows))
            status["completed_jobs"].append(key)
            atomic_json(output_dir / "status.json", status)
            for metric in [f"exact_top{k}_hit" for k in config["topk"]] + [f"group_top{k}_hit" for k in config["topk"]]:
                values = [int(row[metric]) for row in rows]
                summary_row = {"split": split, "model": model_spec["name"], "profile": model_spec["profile"], "metric": metric, "n_targets": len(values), "value": float(np.mean(values)) if values else math.nan}
                all_summary.append(summary_row)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary_row) + "\n")
        reference_name = models[0]["name"]
        reference_keys = split_key_sets[reference_name]
        model_audit = {
            name: {"rows": len(keys), "missing_vs_reference": len(reference_keys - keys), "extra_vs_reference": len(keys - reference_keys)}
            for name, keys in split_key_sets.items()
        }
        split_ok = all(values["missing_vs_reference"] == 0 and values["extra_vs_reference"] == 0 for values in model_audit.values())
        pairing_audit["splits"][split] = {"ok": split_ok, "canonical_targets": len(targets), "models": model_audit}
        pairing_audit["ok"] = pairing_audit["ok"] and split_ok
        if not split_ok:
            raise ValueError(f"shared target rows are not identical for split={split}")
    write_csv(output_dir / "shared_topk_summary.csv", all_summary)
    paired = []
    for split in config["split_manifests"]:
        split_rows = {model: rows for key, rows in rows_by_model.items() if key.startswith(split + "/") for model in [key.split("/", 1)[1]]}
        baseline = {"P0_seed42": split_rows.get("P0_seed42", []), "P1_seed42": split_rows.get("P1_seed42", []), "P2_seed42": split_rows.get("P2_seed42", []), "P3_seed42": split_rows.get("P3_seed42", [])}
        for row in paired_bootstrap(baseline, list(config["comparisons"]), list(config["topk"]), int(config["seed"]), int(config["bootstrap_replicates"])):
            row["split"] = split; paired.append(row)
    write_csv(output_dir / "paired_bootstrap.csv", paired)
    atomic_json(output_dir / "pairing_audit.json", pairing_audit)
    write_markdown(all_summary, output_dir)
    status["state"] = "finished"
    status["summary_rows"] = len(all_summary)
    status["paired_rows"] = len(paired)
    status["config_sha256"] = sha256_file(config_path)
    atomic_json(output_dir / "status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.finalize_existing:
        finalize_existing(config, args.config, args.output_dir)
    else:
        run(config, args.config, args.output_dir, args.device)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
