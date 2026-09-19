"""Aggregate per-disease x horizon risk metrics from evaluate_track_r.py rows.json.gz.

Streams the (large) rows.json.gz written by evaluate_track_r.py and emits compact
per-disease tables.  Point AUC is the tie-corrected Mann-Whitney estimator
(identical formula to ``horizon_control_metrics.binary_auc``); Brier/ECE reuse
``calibration_metrics``.  Two CI flavours are reported:

* ``auc_ci_low/high``      -- patient-level cluster bootstrap (primary; patients are
  the resampling unit because each patient contributes multiple landmark rows).
* ``auc_ci_low/high_delong``-- DeLong variance (unclustered reference).

The ranking / AUC / DeLong helpers are fully vectorised (C-speed), and the cluster
bootstrap for the main disease x horizon cells runs in a ProcessPoolExecutor.

Usage (on server):
  python scripts/summarize_disease_level_risk.py \
    --rows results/track_r_v2_2/runs/seed42_additiverope1/risk/A2_expanded38/validation/rows.json.gz \
    --diseases-yaml docs/expanded_disease_panel_ukb_icd10.yaml \
    --out-dir results/track_r_v2_2/runs/seed42_additiverope1/risk/A2_expanded38/validation/disease_level
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from semantic_delphi_ukb.horizon_control_metrics import (  # noqa: E402
    average_precision,
    calibration_metrics,
)

Z95 = 1.959963984540054

ICD10_CHAPTER = {
    "A": ("I", "Certain infectious and parasitic diseases"),
    "B": ("I", "Certain infectious and parasitic diseases"),
    "C": ("II", "Neoplasms"),
    "D": ("II/III", "Neoplasms / Blood"),
    "E": ("IV", "Endocrine, nutritional and metabolic diseases"),
    "F": ("V", "Mental and behavioural disorders"),
    "G": ("VI", "Diseases of the nervous system"),
    "H": ("VII/VIII", "Eye / Ear and mastoid"),
    "I": ("IX", "Diseases of the circulatory system"),
    "J": ("X", "Diseases of the respiratory system"),
    "K": ("XI", "Diseases of the digestive system"),
    "L": ("XII", "Diseases of the skin and subcutaneous tissue"),
    "M": ("XIII", "Diseases of the musculoskeletal system"),
    "N": ("XIV", "Diseases of the genitourinary system"),
    "O": ("XV", "Pregnancy, childbirth and the puerperium"),
    "P": ("XVI", "Certain conditions originating in the perinatal period"),
    "Q": ("XVII", "Congenital malformations"),
    "R": ("XVIII", "Symptoms, signs and abnormal findings"),
    "S": ("XIX", "Injury, poisoning"),
    "T": ("XIX", "Injury, poisoning"),
    "U": ("XXII", "Codes for special purposes"),
    "V": ("XX", "External causes of morbidity and mortality"),
    "W": ("XX", "External causes of morbidity and mortality"),
    "X": ("XX", "External causes of morbidity and mortality"),
    "Y": ("XX", "External causes of morbidity and mortality"),
    "Z": ("XXI", "Factors influencing health status"),
}


def iter_rows(path: Path):
    """Stream the JSON array of row dicts without loading the whole file."""
    decoder = json.JSONDecoder()
    buf = ""
    with gzip.open(path, "rb") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig")
        while True:
            chunk = text.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,[":
                    pos += 1
                if pos >= len(buf):
                    break
                if buf[pos] == "]":
                    pos += 1
                    break
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = end
            buf = buf[pos:]


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based average (mid) ranks with tie correction, fully vectorised."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_v = values[order]
    _, first, counts = np.unique(sorted_v, return_index=True, return_counts=True)
    midrank = first + (counts - 1.0) / 2.0
    expanded = np.repeat(midrank, counts) + 1.0
    out = np.empty(len(values), dtype=np.float64)
    out[order] = expanded
    return out


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC (same formula as horizon_control_metrics.binary_auc)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _ranks(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def delong_var(scores: np.ndarray, labels: np.ndarray) -> float:
    """DeLong variance for a single classifier (same formula as get_auc_delong_var)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    ranks_all = _ranks(scores)
    v01 = (ranks_all[pos] - _ranks(scores[pos])) / n_neg
    v10 = 1.0 - (ranks_all[neg] - _ranks(scores[neg])) / n_pos
    return float(np.var(v01, ddof=1) / n_pos + np.var(v10, ddof=1) / n_neg)


def _cluster_bootstrap_auc_ci(task):
    """Patient-level cluster bootstrap AUC CI.  Task = (scores, labels, pids, B, seed)."""
    scores, labels, pids, bootstrap, seed = task
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    pids = np.asarray(pids, dtype=np.int64)
    order = np.argsort(pids, kind="stable")
    sorted_pids = pids[order]
    _, starts, counts = np.unique(sorted_pids, return_index=True, return_counts=True)
    n_patients = int(counts.size)
    rng = np.random.default_rng(int(seed))
    aucs: list[float] = []
    for _ in range(int(bootstrap)):
        sampled = rng.choice(n_patients, size=n_patients, replace=True)
        sizes = counts[sampled]
        total = int(sizes.sum())
        cum = np.cumsum(sizes)
        rep_pos = np.repeat(sampled, sizes)
        offsets = np.arange(total) - np.repeat(cum - sizes, sizes)
        idx = order[starts[rep_pos] + offsets]
        a = auc_mann_whitney(scores[idx], labels[idx])
        if np.isfinite(a):
            aucs.append(float(a))
    if not aucs:
        return (float("nan"), float("nan"), 0)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(aucs))


def cell_metrics(scores: np.ndarray, labels: np.ndarray, bins: int) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    auc = auc_mann_whitney(scores, labels)
    auprc = average_precision(scores, labels)
    cal = calibration_metrics(scores, labels, bins=bins, probability_scores=True)
    ci_low = ci_high = ci_low_d = ci_high_d = float("nan")
    if np.isfinite(auc):
        var = delong_var(scores, labels)
        if np.isfinite(var):
            se = float(np.sqrt(var))
            ci_low_d = float(auc - Z95 * se)
            ci_high_d = float(auc + Z95 * se)
    return {
        "n_rows": int(labels.size),
        "n_positives": int(labels.sum()),
        "n_negatives": int(labels.size) - int(labels.sum()),
        "event_rate": float(labels.mean()),
        "auc": float(auc),
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "auc_ci_low_delong": ci_low_d,
        "auc_ci_high_delong": ci_high_d,
        "auprc": float(auprc),
        "brier": float(cal["brier"]),
        "ece": float(cal["ece"]),
    }


def chapter_of(meta: dict) -> tuple[str, str]:
    codes = meta.get("icd10") or []
    letter = codes[0][0] if codes else ""
    chapter, chapter_en = ICD10_CHAPTER.get(letter, ("?", ""))
    return chapter, chapter_en


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--diseases-yaml", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import yaml

    panel = yaml.safe_load(args.diseases_yaml.read_text(encoding="utf-8"))
    meta_by_id = {d["id"]: d for d in panel["diseases"]}

    dh: dict[tuple[str, float], dict] = defaultdict(lambda: {
        "scores": [], "labels": [], "pids": [], "sex": [], "age": [],
    })

    n_total = 0
    for row in iter_rows(args.rows):
        key = (str(row["disease_id"]), float(row["horizon_years"]))
        g = dh[key]
        g["scores"].append(float(row["score"]))
        g["labels"].append(int(row["label"]))
        g["pids"].append(int(row["patient_index"]))
        g["sex"].append(0 if row["sex"] == "female" else 1)
        g["age"].append(float(row["age_start_years"]))
        n_total += 1

    print(f"[summary] total rows = {n_total}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(dh.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    n_groups = len(items)

    horizon_rows: list[dict] = []
    sex_rows: list[dict] = []
    age_rows: list[dict] = []
    pooled_aucs: list[float] = []
    bootstrap_tasks: list[tuple] = []
    bootstrap_slot: list[int] = []  # index into horizon_rows for each task

    for gi, ((disease_id, horizon), g) in enumerate(items):
        scores = np.asarray(g["scores"], dtype=np.float64)
        labels = np.asarray(g["labels"], dtype=np.int8)
        pids = np.asarray(g["pids"], dtype=np.int64)
        sex = np.asarray(g["sex"], dtype=np.int8)
        age = np.asarray(g["age"], dtype=np.float64)
        meta = meta_by_id.get(disease_id, {})
        chapter, chapter_en = chapter_of(meta)

        m = cell_metrics(scores, labels, args.bins)
        m["disease_id"] = disease_id
        m["disease_name"] = meta.get("name", "")
        m["name_cn"] = meta.get("name_cn", "")
        m["icd10"] = ";".join(meta.get("icd10", []))
        m["category"] = meta.get("category", "")
        m["chapter"] = chapter
        m["chapter_en"] = chapter_en
        m["horizon_years"] = horizon
        m["n_patients"] = int(np.unique(pids).size)
        horizon_rows.append(m)
        if np.isfinite(m["auc"]):
            pooled_aucs.append(float(m["auc"]))
        bootstrap_tasks.append((scores, labels, pids, args.bootstrap, 42 + gi))
        bootstrap_slot.append(len(horizon_rows) - 1)

        for code, sex_label in ((0, "female"), (1, "male")):
            mask = sex == code
            if int(mask.sum()) == 0:
                continue
            sm = cell_metrics(scores[mask], labels[mask], args.bins)
            sm.update({"disease_id": disease_id, "horizon_years": horizon, "sex": sex_label,
                       "n_patients": int(np.unique(pids[mask]).size)})
            sex_rows.append(sm)

        for age_bin in np.unique(age):
            mask = age == age_bin
            am = cell_metrics(scores[mask], labels[mask], args.bins)
            am.update({"disease_id": disease_id, "horizon_years": horizon,
                       "age_start_years": float(age_bin), "n_patients": int(np.unique(pids[mask]).size)})
            age_rows.append(am)

        if (gi + 1) % 10 == 0 or gi + 1 == n_groups:
            print(f"[progress] point metrics {gi + 1}/{n_groups} groups done", flush=True)

    # cluster bootstrap for the main disease x horizon cells (parallel)
    print(f"[bootstrap] {len(bootstrap_tasks)} cells x {args.bootstrap} reps, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_cluster_bootstrap_auc_ci, bootstrap_tasks, chunksize=1))
    for slot, (ci_low, ci_high, used) in zip(bootstrap_slot, results):
        horizon_rows[slot]["auc_ci_low"] = ci_low
        horizon_rows[slot]["auc_ci_high"] = ci_high
    print("[bootstrap] done", flush=True)

    _write_csv(args.out_dir / "disease_horizon_risk.csv", horizon_rows)
    _write_csv(args.out_dir / "disease_horizon_sex_risk.csv", sex_rows)
    _write_csv(args.out_dir / "disease_horizon_age_risk.csv", age_rows)

    macro_by_disease: dict[str, list[dict]] = defaultdict(list)
    for r in horizon_rows:
        macro_by_disease[r["disease_id"]].append(r)
    macro_rows = []
    for disease_id, rs in sorted(macro_by_disease.items()):
        base = {k: rs[0][k] for k in ("disease_id", "disease_name", "name_cn", "icd10", "category", "chapter", "chapter_en")}
        for name in ("auc", "auc_ci_low", "auc_ci_high", "auc_ci_low_delong", "auc_ci_high_delong", "auprc", "brier", "ece"):
            vals = [float(r[name]) for r in rs if np.isfinite(float(r[name]))]
            base["macro_" + name] = float(np.mean(vals)) if vals else float("nan")
        base["n_horizons"] = len(rs)
        base["n_rows_sum"] = int(sum(int(r["n_rows"]) for r in rs))
        base["n_positives_sum"] = int(sum(int(r["n_positives"]) for r in rs))
        macro_rows.append(base)
    _write_csv(args.out_dir / "disease_macro_risk.csv", macro_rows)

    out = {
        "total_rows": n_total,
        "diseases": len({k[0] for k in dh}),
        "disease_horizon_cells": len(dh),
        "pooled_macro_auc": float(np.mean(pooled_aucs)) if pooled_aucs else float("nan"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_csv_cell(r[c]) for c in cols) + "\n")
    print(f"[wrote] {path.name} ({len(rows)} rows)", flush=True)


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return repr(value)
    s = str(value)
    if any(ch in s for ch in (",", '"', "\n")):
        return '"' + s.replace('"', '""') + '"'
    return s


if __name__ == "__main__":
    raise SystemExit(main())
