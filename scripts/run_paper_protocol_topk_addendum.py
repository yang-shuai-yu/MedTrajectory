#!/usr/bin/env python3
"""Run and merge P0-P3 next-diagnosis Top-K addendum evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "src/semantic_delphi_ukb/evaluate_expanded_disease_topk.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(values[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def resolved_manifest(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    assets = {
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "evaluator": {"path": str(EVALUATOR), "sha256": sha256(EVALUATOR)},
        "diseases_yaml": {"path": config["diseases_yaml"], "sha256": sha256(Path(config["diseases_yaml"]))},
    }
    models = {}
    for model in config["models"]:
        data_dir = Path(model["data_dir"])
        checkpoint = Path(model["checkpoint"])
        vocab = data_dir / "vocab/dynamic_token_vocab.csv"
        models[model["name"]] = {
            **model,
            "checkpoint_sha256": sha256(checkpoint),
            "prepare_manifest_sha256": sha256(data_dir / "prepare_manifest.json"),
            "vocab_path": str(vocab),
            "vocab_sha256": sha256(vocab),
        }
    return {**config, "resolved_at": utc_now(), "assets": assets, "resolved_models": models}


def command_for(config: Dict[str, Any], model: Dict[str, Any], split: str, out_dir: Path,
                device: str, max_patients: int) -> List[str]:
    data_dir = Path(model["data_dir"])
    return [
        sys.executable, "-u", str(EVALUATOR),
        "--data-dir", str(data_dir),
        "--token-vocab-csv", str(data_dir / "vocab/dynamic_token_vocab.csv"),
        "--diseases-yaml", config["diseases_yaml"],
        "--counts-wide", str(out_dir / "not_used_counts.csv"),
        "--out-dir", str(out_dir),
        "--prefix", "topk",
        "--split", split,
        "--device", device,
        "--batch-size", str(config["batch_size"]),
        "--block-size", str(config["block_size"]),
        "--max-patients", str(max_patients),
        "--topk", ",".join(map(str, config["topk"])),
        "--selection", config["selection"],
        "--padding", config["padding"],
        "--medtrajectory-checkpoint", model["checkpoint"],
        "--medtrajectory-name", model["name"],
        "--seed", str(config["seed"]),
    ]


def merge_outputs(config: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    merged = []
    model_map = {item["name"]: item for item in config["models"]}
    for split in config["splits"]:
        for model_name, model in model_map.items():
            summary_path = output_dir / split / model_name / "topk_summary.csv"
            for row in read_csv(summary_path):
                merged.append({
                    "split": split,
                    "model": model_name,
                    "data_profile": model["data_profile"],
                    "analysis_status": config["analysis_status"],
                    **{key: value for key, value in row.items() if key != "model"},
                })
    write_csv(output_dir / "topk_summary_all.csv", merged)
    overall = [row for row in merged if row.get("disease_id") == "overall_panel_targets"]
    write_csv(output_dir / "topk_overall.csv", overall)
    fields = [
        "split", "model", "data_profile", "n_next_diagnosis_targets",
        "diagnosis_exact_top1_hit_rate", "diagnosis_exact_top5_hit_rate",
        "diagnosis_exact_top10_hit_rate", "diagnosis_group_top10_hit_rate",
    ]
    lines = [
        "# P0-P3 Top-K Addendum", "",
        f"Status: `{config['analysis_status']}`", "",
        "Direct comparisons are P0 vs P1 and P2 vs P3. Cross-profile comparisons are descriptive.", "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in overall:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if field.endswith("_rate") and value:
                value = f"{float(value):.6f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "topk_overall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    names = [item["name"] for item in config["models"]]
    if names != ["P0_seed42", "P1_seed42", "P2_seed42", "P3_seed42"]:
        raise ValueError(f"expected P0-P3 seed42 in order, got {names}")
    if config["splits"] != ["val", "test"]:
        raise ValueError("addendum must run val then test")
    if config["analysis_status"] != "secondary_posthoc_addendum":
        raise ValueError("test addendum must remain secondary_posthoc_addendum")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_protocol_v1/topk_addendum_v1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output_dir = Path(config["output_dir"])
    if args.dry_run:
        for split in config["splits"]:
            for model in config["models"]:
                print(" ".join(command_for(config, model, split, output_dir / split / model["name"], args.device, args.max_patients)))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status = {"state": "running", "started_at": utc_now(), "current_job": None, "completed_jobs": []}
    write_json_atomic(status_path, status)
    write_json_atomic(output_dir / "resolved_manifest.json", resolved_manifest(config, args.config))
    try:
        for split in config["splits"]:
            for model in config["models"]:
                job_name = f"{split}/{model['name']}"
                job_dir = output_dir / split / model["name"]
                job_dir.mkdir(parents=True, exist_ok=True)
                status["current_job"] = job_name
                write_json_atomic(status_path, status)
                command = command_for(config, model, split, job_dir, args.device, args.max_patients)
                (job_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
                with (job_dir / "stdout.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                status["completed_jobs"].append(job_name)
                write_json_atomic(status_path, status)
        rows = merge_outputs(config, output_dir)
        status.update({"state": "finished", "finished_at": utc_now(), "current_job": None, "summary_rows": len(rows)})
        write_json_atomic(status_path, status)
    except Exception as exc:
        status.update({"state": "failed", "failed_at": utc_now(), "error": repr(exc)})
        write_json_atomic(status_path, status)
        raise


if __name__ == "__main__":
    main()
