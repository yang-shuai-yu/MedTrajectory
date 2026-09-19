"""Aggregate Track R v2.2 validation outputs under the frozen multi-seed gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.compare_horizon_control_tasks import (  # noqa: E402
    _compact_bootstrap_macro_values,
    _compact_metrics,
    _load_compact_inputs,
)
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402
from semantic_delphi_ukb.track_r_rows import read_json  # noqa: E402
from semantic_delphi_ukb.track_r_v2_2 import load_wavelength_manifest  # noqa: E402


MODELS = ("A0", "A2", "A2-noAge")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--out-dir", type=Path, required=True)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_key(row: Mapping) -> tuple:
    return (
        int(row["patient_index"]),
        int(row["original_source_event_ordinal"]),
        int(row["original_target_event_ordinal"]),
        int(row["context_window_start"]),
        int(row["local_source_position"]),
    )


def patient_nll_payload(rows: Sequence[Mapping]) -> dict:
    identity_digest = hashlib.sha256()
    sums = {}
    counts = {}
    previous_key = None
    for row in rows:
        key = event_key(row)
        if previous_key is not None and key <= previous_key:
            raise ValueError("pretraining-fidelity rows must be strictly ordered by the frozen event key")
        previous_key = key
        identity = key + (
            int(row["source_token_id"]),
            float(row["source_age_days"]),
            int(row["target_token_id"]),
            float(row["target_age_days"]),
            float(row["positive_gap_days"]),
        )
        identity_digest.update(repr(identity).encode("ascii"))
        patient = key[0]
        sums[patient] = sums.get(patient, 0.0) + float(row["nll"])
        counts[patient] = counts.get(patient, 0) + 1
    patients = np.asarray(sorted(sums), dtype=np.int64)
    means = np.asarray([sums[int(patient)] / counts[int(patient)] for patient in patients], dtype=np.float64)
    if not len(patients) or not np.isfinite(means).all():
        raise ValueError("pretraining-fidelity rows contain no finite patient NLL means")
    return {
        "identity_sha256": identity_digest.hexdigest(),
        "row_count": len(rows),
        "patients": patients,
        "patient_mean_nll": means,
    }


def load_nll_matrix(paths: Mapping[int, Mapping[str, Path]]) -> tuple[np.ndarray, dict, dict]:
    reference_identity = None
    reference_patients = None
    values = {}
    inputs = {}
    for seed in sorted(paths):
        values[seed] = {}
        inputs[str(seed)] = {}
        for model in MODELS:
            path = Path(paths[seed][model])
            rows = read_json(path)
            payload = patient_nll_payload(rows)
            del rows
            gc.collect()
            identity = (payload["identity_sha256"], payload["row_count"])
            if reference_identity is None:
                reference_identity = identity
                reference_patients = payload["patients"]
            elif identity != reference_identity or not np.array_equal(payload["patients"], reference_patients):
                raise ValueError(f"NLL evaluation rows are not exactly paired: seed={seed}, model={model}")
            values[seed][model] = payload["patient_mean_nll"]
            inputs[str(seed)][model] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "row_count": payload["row_count"],
            }
    return reference_patients, values, inputs


def contrast_matrix(values: Mapping[int, Mapping[str, np.ndarray]], left: str, right: str) -> tuple[list[int], np.ndarray]:
    seeds = sorted(values)
    matrix = np.stack([values[seed][left] - values[seed][right] for seed in seeds], axis=0)
    return seeds, matrix


def bootstrap_patient_mean(
    delta_by_seed: np.ndarray,
    *,
    replicates: int,
    seed: int,
    percentile: Sequence[float],
) -> dict:
    if delta_by_seed.ndim != 2 or not delta_by_seed.shape[1]:
        raise ValueError("bootstrap requires [seed,patient] deltas")
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=np.float64)
    batch_size = 8
    for start in range(0, int(replicates), batch_size):
        count = min(batch_size, int(replicates) - start)
        sampled = rng.integers(0, delta_by_seed.shape[1], size=(count, delta_by_seed.shape[1]))
        per_seed_delta = np.column_stack([
            delta_by_seed[seed_index][sampled].mean(axis=1)
            for seed_index in range(delta_by_seed.shape[0])
        ])
        values[start: start + count] = per_seed_delta.mean(axis=1)
    low, high = np.percentile(values, percentile)
    return {
        "delta": float(delta_by_seed.mean(axis=1).mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": int(replicates),
    }


def seed_averaged_bootstrap_delta(
    bootstrap_macro: np.ndarray,
    names: Sequence[str],
    seeds: Sequence[int],
    left: str,
    right: str,
    percentile: Sequence[float],
) -> tuple[np.ndarray, dict]:
    index = {name: position for position, name in enumerate(names)}
    per_seed = np.column_stack([
        bootstrap_macro[:, index[f"seed{seed}:{left}"]]
        - bootstrap_macro[:, index[f"seed{seed}:{right}"]]
        for seed in seeds
    ])
    averaged = per_seed.mean(axis=1)
    if not np.isfinite(averaged).all():
        raise ValueError("clinical bootstrap did not produce every frozen replicate")
    low, high = np.percentile(averaged, percentile)
    return per_seed, {
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": int(len(averaged)),
    }


def resolve_inputs(protocol: dict) -> dict:
    output = ROOT / protocol["output_root"]
    nll = {}
    clinical = {}
    summaries = {}
    for seed in protocol["seeds"]:
        run = output / "runs" / protocol["run_tag_template"].format(seed=seed)
        nll[int(seed)] = {
            model: run / "pretraining_fidelity" / model / "rows.json.gz" for model in MODELS
        }
        clinical[int(seed)] = {
            model: run / "clinical" / model / "rows.json.gz" for model in MODELS
        }
        summaries[int(seed)] = {
            model: run / "pretraining_fidelity" / model / "summary.json" for model in MODELS
        }
    return {"nll": nll, "clinical": clinical, "summaries": summaries}


def mechanism_assessment(protocol: dict, summary_paths: Mapping[int, Mapping[str, Path]]) -> dict:
    gate = protocol["mechanism_gate"]
    per_seed = {}
    for seed in protocol["seeds"]:
        summary = json.loads(Path(summary_paths[int(seed)]["A2"]).read_text(encoding="utf-8-sig"))
        diagnostic = summary["additive_rope_diagnostics"]
        saturated = float(diagnostic["saturated_fraction"])
        amplitude = float(diagnostic["relative_logit_amplitude_median"])
        passed = (
            saturated < float(gate["maximum_saturated_fraction_exclusive"])
            and amplitude > float(gate["minimum_relative_logit_amplitude_exclusive"])
        )
        per_seed[str(seed)] = {
            "saturated_fraction": saturated,
            "relative_logit_amplitude_median": amplitude,
            "passed": bool(passed),
        }
    passed_count = sum(int(item["passed"]) for item in per_seed.values())
    return {
        "aggregation": gate["aggregation"],
        "required_seed_passes": int(gate["required_seed_passes"]),
        "passed_seed_count": passed_count,
        "per_seed": per_seed,
        "passed": passed_count == int(gate["required_seed_passes"]),
    }


def write_outputs(out_dir: Path, payload: dict) -> None:
    if out_dir.exists():
        raise FileExistsError(f"final assessment output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "assessment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Track R v2.2 Final Validation Assessment",
        "",
        f"Overall gate passed: **{str(payload['overall_gate_passed']).lower()}**",
        "",
        "| Criterion | Delta | 95% CI | Passed |",
        "|---|---:|---:|:---:|",
    ]
    for name in ("primary_nll", "clinical_noninferiority"):
        item = payload[name]
        lines.append(
            f"| {name} | {item['delta']:+.8f} | "
            f"[{item['ci95_low']:+.8f}, {item['ci95_high']:+.8f}] | {item['passed']} |"
        )
    lines.append(f"| mechanism_3_of_3 |  |  | {payload['mechanism']['passed']} |")
    lines += [
        "",
        "A2-noAge is descriptive only and is not part of the formal gate.",
        "",
        "Locked test read: `false`.",
    ]
    (out_dir / "assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "status.json").write_text(
        json.dumps({"status": "finished", "overall_gate_passed": payload["overall_gate_passed"]}, indent=2),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    if protocol["protocol_id"] != "track_r_v2_2" or protocol["status"] != "ready_for_training":
        raise ValueError("finalization requires the fully frozen ready_for_training protocol")
    contract = protocol["wavelength_contract"]
    manifest_path = Path(contract["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    load_wavelength_manifest(
        manifest_path,
        expected_sha256=contract["expected_sha256"],
        scale_factor_bounds=contract["scale_factor_bounds"],
        require_log_scale_match=bool(contract["require_manifest_log_scale_match"]),
    )
    inputs = resolve_inputs(protocol)
    missing = [
        str(path)
        for family in inputs.values()
        for paths_by_model in family.values()
        for path in paths_by_model.values()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Track R v2.2 finalization inputs are incomplete: {missing}")

    patients, nll_values, nll_inputs = load_nll_matrix(inputs["nll"])
    bootstrap = protocol["bootstrap_resample"]
    percentile = bootstrap["percentile"]
    seeds, primary_matrix = contrast_matrix(nll_values, "A2", "A0")
    primary = bootstrap_patient_mean(
        primary_matrix,
        replicates=bootstrap["replicates"],
        seed=bootstrap["seed"],
        percentile=percentile,
    )
    primary_seed_delta = primary_matrix.mean(axis=1)
    primary.update({
        "contrast": "A2 - A0",
        "per_seed_delta": {str(seed): float(value) for seed, value in zip(seeds, primary_seed_delta)},
        "seed_delta_sample_sd": float(np.std(primary_seed_delta, ddof=1)),
        "direction_consistent_seeds": int((primary_seed_delta < 0.0).sum()),
    })
    primary["passed"] = bool(
        primary["delta"] < 0.0
        and primary["ci95_high"] < 0.0
        and primary["direction_consistent_seeds"]
        == int(protocol["pretraining_fidelity"]["primary_contrast"]["required_seed_directions"])
    )

    _, ablation_matrix = contrast_matrix(nll_values, "A2", "A2-noAge")
    ablation = bootstrap_patient_mean(
        ablation_matrix,
        replicates=bootstrap["replicates"],
        seed=bootstrap["seed"],
        percentile=percentile,
    )
    ablation.update({
        "contrast": "A2 - A2-noAge",
        "formal_gate": False,
        "per_seed_delta": {
            str(seed): float(value) for seed, value in zip(seeds, ablation_matrix.mean(axis=1))
        },
    })

    clinical_specs = []
    clinical_inputs = {}
    for seed in seeds:
        clinical_inputs[str(seed)] = {}
        for model in ("A0", "A2"):
            name = f"seed{seed}:{model}"
            path = inputs["clinical"][seed][model]
            clinical_specs.append(f"{name}={path}")
            clinical_inputs[str(seed)][model] = {"path": str(path), "sha256": file_sha256(path)}
    compact = _load_compact_inputs(clinical_specs)
    clinical_summary, _ = _compact_metrics(compact, bins=10)
    bootstrap_macro = _compact_bootstrap_macro_values(
        compact, bootstrap=bootstrap["replicates"], seed=bootstrap["seed"]
    )
    _, clinical_ci = seed_averaged_bootstrap_delta(
        bootstrap_macro, compact["names"], seeds, "A2", "A0", percentile
    )
    clinical_seed_delta = np.asarray([
        clinical_summary[f"seed{seed}:A2"]["auc"] - clinical_summary[f"seed{seed}:A0"]["auc"]
        for seed in seeds
    ])
    clinical = {
        "contrast": "A2 - A0",
        "delta": float(clinical_seed_delta.mean()),
        **clinical_ci,
        "noninferiority_margin": float(protocol["clinical_gate"]["noninferiority_margin"]),
        "per_seed_delta": {str(seed): float(value) for seed, value in zip(seeds, clinical_seed_delta)},
        "seed_delta_sample_sd": float(np.std(clinical_seed_delta, ddof=1)),
        "model_metrics": clinical_summary,
    }
    clinical["passed"] = clinical["ci95_low"] > clinical["noninferiority_margin"]
    mechanism = mechanism_assessment(protocol, inputs["summaries"])

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "wavelength_manifest_sha256": contract["expected_sha256"],
        "locked_test_read": False,
        "patient_count": int(len(patients)),
        "bootstrap_resample": bootstrap,
        "primary_nll": primary,
        "clinical_noninferiority": clinical,
        "mechanism": mechanism,
        "a2_noage_descriptive": ablation,
        "overall_gate_passed": bool(primary["passed"] and clinical["passed"] and mechanism["passed"]),
        "result_integrity": protocol["result_integrity"],
        "inputs": {"nll": nll_inputs, "clinical": clinical_inputs},
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_gate_passed": payload["overall_gate_passed"],
        "assessment": str(args.out_dir / "assessment.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
