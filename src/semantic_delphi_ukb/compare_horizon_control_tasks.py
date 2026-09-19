"""Compare fixed-horizon prediction rows with patient-level paired bootstrap."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.horizon_control_metrics import average_precision, binary_auc, summarize_scores
from semantic_delphi_ukb.track_r_rows import read_json, read_json_bytes


AGGREGATION = "macro_over_sex_disease_horizon_age_cells"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True, help="model_name=rows.json; repeat for models")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--memory-efficient", action="store_true")
    p.add_argument(
        "--factorial-interaction",
        default=None,
        help="A1_STATIC,A0_STATIC,A1_NOSTATIC,A0_NOSTATIC; requires --memory-efficient",
    )
    return p


def load_inputs(specs):
    outputs = {}
    for spec in specs:
        name, raw_path = spec.split("=", 1)
        outputs[name] = read_json(Path(raw_path))
    return outputs


def key(row):
    """Return the full prediction identity used for paired comparisons."""
    age = row.get("age_start_years")
    return (
        int(row["patient_index"]),
        str(row["disease_id"]),
        float(row["horizon_years"]),
        str(row.get("sex", "")),
        None if age is None else float(age),
    )


def _index_rows(rows):
    indexed = {}
    for row in rows:
        item = key(row)
        if item in indexed:
            raise ValueError(f"duplicate paired prediction key: {item!r}")
        indexed[item] = row
    return indexed


def stratum_key(row):
    if row.get("sex") in (None, "") or row.get("age_start_years") is None:
        raise ValueError("medical comparison rows require sex and age_start_years")
    return (
        str(row["sex"]),
        str(row["disease_id"]),
        float(row["horizon_years"]),
        float(row["age_start_years"]),
    )


def macro_metrics(rows):
    groups = {}
    for row in rows:
        groups.setdefault(stratum_key(row), []).append(row)
    metrics = []
    for group_rows in groups.values():
        labels = np.asarray([int(row["label"]) for row in group_rows], dtype=np.int8)
        scores = np.asarray([float(row["score"]) for row in group_rows], dtype=np.float64)
        probability_scores = all(bool(row.get("score_is_probability", str(row.get("model", "")) != "lm")) for row in group_rows)
        values = summarize_scores(scores, labels, probability_scores=probability_scores)
        if np.isfinite(values["auc"]):
            metrics.append(values)
    if not metrics:
        return {"strata": 0, "auc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "ece": float("nan")}
    macro = {}
    for name in ("auc", "auprc", "brier", "ece"):
        values = np.asarray([row[name] for row in metrics], dtype=np.float64)
        macro[name] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")
    return {"strata": len(metrics), **macro}


def calibration_table(rows, bins=10):
    """Return fixed-width calibration bins for each disease/horizon cell."""
    grouped = {}
    for row in rows:
        if not bool(row.get("score_is_probability", str(row.get("model", "")) != "lm")):
            continue
        grouped.setdefault(stratum_key(row), []).append(row)
    output = []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for (sex, disease_id, horizon, age_start_years), group_rows in sorted(grouped.items()):
        scores = np.clip(np.asarray([float(row["score"]) for row in group_rows], dtype=np.float64), 0.0, 1.0)
        labels = np.asarray([int(row["label"]) for row in group_rows], dtype=np.int8)
        for index in range(int(bins)):
            mask = (scores >= edges[index]) & (scores <= edges[index + 1] if index == int(bins) - 1 else scores < edges[index + 1])
            if not mask.any():
                continue
            output.append({
                "sex": sex,
                "disease_id": disease_id,
                "horizon_years": horizon,
                "age_start_years": age_start_years,
                "bin": index,
                "count": int(mask.sum()),
                "score_min": float(scores[mask].min()),
                "score_max": float(scores[mask].max()),
                "mean_score": float(scores[mask].mean()),
                "event_rate": float(labels[mask].mean()),
                "abs_calibration_error": float(abs(scores[mask].mean() - labels[mask].mean())),
            })
    return output


def paired_macro_bootstrap(left_rows, right_rows, bootstrap, seed):
    left = _index_rows(left_rows)
    right = _index_rows(right_rows)
    if set(left) != set(right):
        raise ValueError(
            "paired prediction files must contain identical landmark keys: "
            f"left_only={len(set(left) - set(right))}, right_only={len(set(right) - set(left))}"
        )
    common = sorted(set(left) & set(right))
    if any(int(left[item]["label"]) != int(right[item]["label"]) for item in common):
        raise ValueError("paired prediction files disagree on at least one label")
    if any(
        float(left[item].get("prediction_age_days", float("nan")))
        != float(right[item].get("prediction_age_days", float("nan")))
        for item in common
        if left[item].get("prediction_age_days") is not None or right[item].get("prediction_age_days") is not None
    ):
        raise ValueError("paired prediction files disagree on at least one prediction age")
    patient_ids = np.asarray([item[0] for item in common], dtype=np.int64)
    patients, patient_inverse = np.unique(patient_ids, return_inverse=True)
    strata = {}
    for row_idx, item in enumerate(common):
        strata.setdefault(stratum_key(left[item]), []).append(row_idx)
    strata = {name: np.asarray(indices, dtype=np.int64) for name, indices in strata.items()}
    left_scores = np.asarray([float(left[item]["score"]) for item in common], dtype=np.float64)
    right_scores = np.asarray([float(right[item]["score"]) for item in common], dtype=np.float64)
    labels = np.asarray([int(left[item]["label"]) for item in common], dtype=np.int8)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(bootstrap)):
        selected_patients = rng.integers(0, len(patients), size=len(patients))
        multiplicity = np.bincount(selected_patients, minlength=len(patients))
        left_aucs = []
        right_aucs = []
        for indices in strata.values():
            repeated = np.repeat(indices, multiplicity[patient_inverse[indices]])
            if repeated.size == 0:
                continue
            left_auc = summarize_scores(left_scores[repeated], labels[repeated], probability_scores=False)["auc"]
            right_auc = summarize_scores(right_scores[repeated], labels[repeated], probability_scores=False)["auc"]
            if np.isfinite(left_auc) and np.isfinite(right_auc):
                left_aucs.append(left_auc)
                right_aucs.append(right_auc)
        if left_aucs and right_aucs:
            deltas.append(float(np.mean(left_aucs) - np.mean(right_aucs)))
    if not deltas:
        return {"delta_auc": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "p_two_sided": float("nan"), "bootstrap_used": 0, "common_rows": len(common)}
    below = (np.asarray(deltas) <= 0.0).sum()
    above = (np.asarray(deltas) >= 0.0).sum()
    p_two_sided = min(1.0, 2.0 * min((below + 1) / (len(deltas) + 1), (above + 1) / (len(deltas) + 1)))
    return {
        "delta_auc": float(macro_metrics(left_rows)["auc"] - macro_metrics(right_rows)["auc"]),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "p_two_sided": float(p_two_sided),
        "bootstrap_used": len(deltas),
        "common_rows": len(common),
        "common_patients": len(patients),
    }


def holm_adjust(pvalues):
    ordered = sorted(enumerate(pvalues), key=lambda item: (float(item[1]), item[0]))
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, (index, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(pvalue)))
        adjusted[index] = running
    return adjusted


def _load_compact_inputs(specs):
    try:
        import orjson
        json_loads = orjson.loads
    except ImportError:
        json_loads = json.loads

    names = []
    scores_by_model = {}
    patient_ids = labels = prediction_ages = stratum_ids = None
    stratum_keys = []
    stratum_lookup = {}
    probability_scores = {}
    row_count = None

    for model_index, spec in enumerate(specs):
        name, raw_path = spec.split("=", 1)
        names.append(name)
        rows = json_loads(read_json_bytes(Path(raw_path)))
        if row_count is None:
            row_count = len(rows)
            patient_ids = np.empty(row_count, dtype=np.int32)
            labels = np.empty(row_count, dtype=np.int8)
            prediction_ages = np.empty(row_count, dtype=np.float64)
            stratum_ids = np.empty(row_count, dtype=np.uint16)
        elif len(rows) != row_count:
            raise ValueError(f"paired prediction files have different row counts: {name}={len(rows)}, expected={row_count}")

        scores = np.empty(row_count, dtype=np.float64)
        is_probability = None
        for index, row in enumerate(rows):
            row_stratum = stratum_key(row)
            row_probability = bool(row.get("score_is_probability", str(row.get("model", "")) != "lm"))
            if is_probability is None:
                is_probability = row_probability
            elif row_probability != is_probability:
                raise ValueError(f"mixed probability and non-probability scores for {name}")
            scores[index] = float(row["score"])
            if model_index == 0:
                stratum_id = stratum_lookup.get(row_stratum)
                if stratum_id is None:
                    stratum_id = len(stratum_keys)
                    if stratum_id >= np.iinfo(np.uint16).max:
                        raise ValueError("too many comparison strata")
                    stratum_lookup[row_stratum] = stratum_id
                    stratum_keys.append(row_stratum)
                patient_ids[index] = int(row["patient_index"])
                labels[index] = int(row["label"])
                prediction_ages[index] = float(row.get("prediction_age_days", np.nan))
                stratum_ids[index] = stratum_id
            else:
                if int(row["patient_index"]) != int(patient_ids[index]):
                    raise ValueError(f"paired prediction patient mismatch for {name} at row {index}")
                if int(row["label"]) != int(labels[index]):
                    raise ValueError(f"paired prediction label mismatch for {name} at row {index}")
                if row_stratum != stratum_keys[int(stratum_ids[index])]:
                    raise ValueError(f"paired prediction stratum mismatch for {name} at row {index}")
                row_age = float(row.get("prediction_age_days", np.nan))
                if not (row_age == float(prediction_ages[index]) or (np.isnan(row_age) and np.isnan(prediction_ages[index]))):
                    raise ValueError(f"paired prediction age mismatch for {name} at row {index}")
        scores_by_model[name] = scores
        probability_scores[name] = bool(is_probability)
        del rows
        gc.collect()

    if row_count is None or patient_ids is None or labels is None or prediction_ages is None or stratum_ids is None:
        raise ValueError("no comparison rows loaded")
    combined_keys = patient_ids.astype(np.int64) * max(1, len(stratum_keys)) + stratum_ids.astype(np.int64)
    if np.unique(combined_keys).size != row_count:
        raise ValueError("duplicate paired prediction key")
    order = np.argsort(stratum_ids, kind="stable")
    boundaries = np.flatnonzero(np.diff(stratum_ids[order])) + 1
    groups = np.split(order, boundaries)
    return {
        "names": names,
        "scores": scores_by_model,
        "probability_scores": probability_scores,
        "patient_ids": patient_ids,
        "labels": labels,
        "stratum_ids": stratum_ids,
        "stratum_keys": stratum_keys,
        "groups": groups,
    }


def _compact_metrics(compact, bins):
    labels = compact["labels"]
    summaries = {}
    calibration = {}
    for name in compact["names"]:
        scores = compact["scores"][name]
        metrics = []
        calibration_rows = []
        for stratum, indices in zip(compact["stratum_keys"], compact["groups"]):
            group_scores = scores[indices]
            group_labels = labels[indices]
            values = {
                "auc": binary_auc(group_scores, group_labels),
                "auprc": average_precision(group_scores, group_labels),
                "brier": float(np.mean((group_scores - group_labels) ** 2)),
                "ece": 0.0,
            }
            if compact["probability_scores"][name]:
                edges = np.linspace(0.0, 1.0, bins + 1)
                for bin_index in range(bins):
                    mask = (
                        (group_scores >= edges[bin_index])
                        & (group_scores <= edges[bin_index + 1] if bin_index == bins - 1 else group_scores < edges[bin_index + 1])
                    )
                    if mask.any():
                        values["ece"] += float(mask.mean()) * abs(
                            float(group_scores[mask].mean()) - float(group_labels[mask].mean())
                        )
            else:
                values["brier"] = float("nan")
                values["ece"] = float("nan")
            if np.isfinite(values["auc"]):
                metrics.append(values)
            if compact["probability_scores"][name]:
                calibration_rows.extend(calibration_table([
                    {
                        "sex": stratum[0], "disease_id": stratum[1],
                        "horizon_years": stratum[2], "age_start_years": stratum[3],
                        "score": float(score), "label": int(label),
                        "score_is_probability": True,
                    }
                    for score, label in zip(scores[indices], labels[indices])
                ], bins=bins))
        summary = {"strata": len(metrics)}
        for metric in ("auc", "auprc", "brier", "ece"):
            values = np.asarray([row[metric] for row in metrics], dtype=np.float64)
            summary[metric] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")
        summaries[name] = summary
        calibration[name] = calibration_rows
    return summaries, calibration


def _weighted_auc_batch(scores, labels, patients, multiplicities):
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)
    weights = multiplicities[:, patients[order]].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    positives = np.add.reduceat(weights * sorted_labels, starts, axis=1)
    negatives = np.add.reduceat(weights * (1.0 - sorted_labels), starts, axis=1)
    negative_before = np.cumsum(negatives, axis=1) - negatives
    numerator = np.sum(positives * (negative_before + 0.5 * negatives), axis=1)
    denominator = np.sum(positives, axis=1) * np.sum(negatives, axis=1)
    return np.divide(numerator, denominator, out=np.full(len(multiplicities), np.nan), where=denominator > 0)


def _compact_bootstrap_macro_values(compact, bootstrap, seed):
    names = compact["names"]
    patient_ids = compact["patient_ids"]
    unique_patients, patient_inverse = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    batches = []
    batch_size = 8
    for batch_start in range(0, int(bootstrap), batch_size):
        count = min(batch_size, int(bootstrap) - batch_start)
        multiplicities = np.empty((count, len(unique_patients)), dtype=np.uint16)
        for index in range(count):
            selected = rng.integers(0, len(unique_patients), size=len(unique_patients))
            multiplicities[index] = np.bincount(selected, minlength=len(unique_patients))
        macro_sum = np.zeros((count, len(names)), dtype=np.float64)
        valid_count = np.zeros(count, dtype=np.int32)
        for indices in compact["groups"]:
            group_labels = compact["labels"][indices]
            group_patients = patient_inverse[indices]
            aucs = np.column_stack([
                _weighted_auc_batch(
                    compact["scores"][name][indices], group_labels, group_patients, multiplicities
                )
                for name in names
            ])
            valid = np.all(np.isfinite(aucs), axis=1)
            macro_sum[valid] += aucs[valid]
            valid_count[valid] += 1
        batches.append(np.divide(
            macro_sum, valid_count[:, None], out=np.full_like(macro_sum, np.nan),
            where=valid_count[:, None] > 0,
        ))
    return np.concatenate(batches, axis=0)


def _compact_pairwise_bootstrap(compact, summaries, bootstrap_macro):
    names = compact["names"]
    patient_ids = compact["patient_ids"]
    unique_patients = np.unique(patient_ids)
    comparisons = {}
    for left_index, left_name in enumerate(names):
        for right_index, right_name in enumerate(names[left_index + 1 :], start=left_index + 1):
            deltas = bootstrap_macro[:, left_index] - bootstrap_macro[:, right_index]
            deltas = deltas[np.isfinite(deltas)]
            below = int((deltas <= 0.0).sum())
            above = int((deltas >= 0.0).sum())
            p_two_sided = min(1.0, 2.0 * min((below + 1) / (len(deltas) + 1), (above + 1) / (len(deltas) + 1)))
            comparisons[f"{left_name} - {right_name}"] = {
                "delta_auc": float(summaries[left_name]["auc"] - summaries[right_name]["auc"]),
                "ci95_low": float(np.percentile(deltas, 2.5)),
                "ci95_high": float(np.percentile(deltas, 97.5)),
                "p_two_sided": float(p_two_sided),
                "bootstrap_used": len(deltas),
                "common_rows": int(len(patient_ids)),
                "common_patients": int(len(unique_patients)),
            }
    return comparisons


def _compact_bootstrap(compact, summaries, bootstrap, seed):
    bootstrap_macro = _compact_bootstrap_macro_values(compact, bootstrap, seed)
    return _compact_pairwise_bootstrap(compact, summaries, bootstrap_macro)


def factorial_auc_interaction(compact, summaries, bootstrap_macro, specification):
    names = [item.strip() for item in specification.split(",") if item.strip()]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("factorial interaction requires four distinct model names")
    missing = [name for name in names if name not in compact["names"]]
    if missing:
        raise ValueError(f"factorial interaction models are missing from inputs: {missing}")
    a1_static, a0_static, a1_no_static, a0_no_static = names
    indices = {name: compact["names"].index(name) for name in names}
    values = (
        bootstrap_macro[:, indices[a1_static]]
        - bootstrap_macro[:, indices[a0_static]]
        - bootstrap_macro[:, indices[a1_no_static]]
        + bootstrap_macro[:, indices[a0_no_static]]
    )
    values = values[np.isfinite(values)]
    below = int((values <= 0.0).sum())
    above = int((values >= 0.0).sum())
    p_two_sided = min(
        1.0,
        2.0 * min((below + 1) / (len(values) + 1), (above + 1) / (len(values) + 1)),
    )
    point = (
        summaries[a1_static]["auc"]
        - summaries[a0_static]["auc"]
        - summaries[a1_no_static]["auc"]
        + summaries[a0_no_static]["auc"]
    )
    return {
        "formula": f"({a1_static} - {a0_static}) - ({a1_no_static} - {a0_no_static})",
        "interaction_auc": float(point),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "p_two_sided": float(p_two_sided),
        "bootstrap_used": int(len(values)),
        "common_rows": int(len(compact["patient_ids"])),
        "common_patients": int(len(np.unique(compact["patient_ids"]))),
    }


def _write_outputs(args, summary, calibration, comparisons, factorial_interaction=None):
    comparison_values = list(comparisons.values())
    adjusted = holm_adjust([float(item.get("p_two_sided", 1.0)) for item in comparison_values])
    for item, pvalue in zip(comparison_values, adjusted):
        item["p_holm"] = float(pvalue)
    payload = {
        "aggregation": AGGREGATION,
        "models": summary,
        "paired_bootstrap": comparisons,
        "factorial_interaction": factorial_interaction,
        "calibration_bins": calibration,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "calibration_bins_requested": args.calibration_bins,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    calibration_rows = []
    for name, rows in calibration.items():
        for row in rows:
            calibration_rows.append({"model": name, **row})
    if calibration_rows:
        with (args.out_dir / "calibration_bins.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0]))
            writer.writeheader()
            writer.writerows(calibration_rows)
    lines = ["# Medical Horizon Control Comparison", "", f"Aggregation: `{AGGREGATION}`", "", "| Model | Strata | Macro AUC | Macro AUPRC | Macro Brier | Macro ECE |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in summary.items():
        lines.append(f"| {name} | {values['strata']} | {values['auc']:.6f} | {values['auprc']:.6f} | {values['brier']:.6f} | {values['ece']:.6f} |")
    lines += ["", "| Contrast | Delta AUC | 95% CI | Bootstrap p | Holm p | Common patients | Common rows |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in comparisons.items():
        lines.append(f"| {name} | {values['delta_auc']:+.6f} | [{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] | {values.get('p_two_sided', float('nan')):.6f} | {values.get('p_holm', float('nan')):.6f} | {values.get('common_patients', 0)} | {values['common_rows']} |")
    if factorial_interaction is not None:
        values = factorial_interaction
        lines += [
            "",
            "## Prespecified 2x2 Interaction",
            "",
            f"Formula: `{values['formula']}`",
            "",
            "| Interaction AUC | 95% CI | Bootstrap p | Common patients | Common rows |",
            "|---:|---:|---:|---:|---:|",
            f"| {values['interaction_auc']:+.6f} | [{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] | {values['p_two_sided']:.6f} | {values['common_patients']} | {values['common_rows']} |",
        ]
    (args.out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.memory_efficient:
        compact = _load_compact_inputs(args.input)
        summary, calibration = _compact_metrics(compact, args.calibration_bins)
        bootstrap_macro = _compact_bootstrap_macro_values(compact, args.bootstrap, args.seed)
        comparisons = _compact_pairwise_bootstrap(compact, summary, bootstrap_macro)
        interaction = (
            factorial_auc_interaction(compact, summary, bootstrap_macro, args.factorial_interaction)
            if args.factorial_interaction
            else None
        )
        _write_outputs(args, summary, calibration, comparisons, interaction)
        return
    if args.factorial_interaction:
        raise ValueError("--factorial-interaction requires --memory-efficient")
    inputs = load_inputs(args.input)
    summary = {name: macro_metrics(rows) for name, rows in inputs.items()}
    calibration = {name: calibration_table(rows, bins=args.calibration_bins) for name, rows in inputs.items()}
    comparisons = {}
    names = list(inputs)
    for idx, left_name in enumerate(names):
        for right_name in names[idx + 1 :]:
            comparisons[f"{left_name} - {right_name}"] = paired_macro_bootstrap(
                inputs[left_name], inputs[right_name], args.bootstrap, args.seed
            )
    _write_outputs(args, summary, calibration, comparisons)


if __name__ == "__main__":
    main()
