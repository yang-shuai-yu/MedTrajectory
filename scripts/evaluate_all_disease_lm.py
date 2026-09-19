"""LM-based disease-onset discrimination across ALL diagnosis tokens (1000+ diseases).

Score: LM next-event logit for each diagnosis token at the last prediction position
      (``last_prediction_positions``; the position whose next event is the last
      retained clinical event), under the frozen Track R CARoPE input protocol
      (static prefix + dynamic BOS + left dynamic window).
Label: whether that diagnosis token occurs at any later age in the patient's full
       follow-up (first-occurrence age > prediction age).

DISTINCT from the 38-disease censor-aware horizon risk head:
  * score source  : generative lm_head (next-token logit), NOT horizon_risk_head
  * label         : any future occurrence (not censor-adjusted, not horizon-stratified)
  * prediction pt : last prediction position in the left window (not shared landmarks)

One prediction per patient => rows i.i.d. across patients => DeLong CI is exact.

Usage (server):
  python scripts/evaluate_all_disease_lm.py \
    --checkpoint results/track_r_v2_2/runs/seed42_additiverope1/risk/A2_expanded38/checkpoints/best_val_horizon_auc.pt \
    --protocol configs/track_r_v2_2/TRACK_R_v2_2.json \
    --data-dir data/track_r_v2_1/multitype_static_prefix --split val \
    --vocab-csv data/track_r_v2_1/multitype_static_prefix/vocab/dynamic_token_vocab.csv \
    --out-dir results/all_disease_lm --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import build_official_left_batch  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_assets,
    load_track_r_static_features,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol, prepend_static_context  # noqa: E402
from semantic_delphi_ukb.train_car_rope import load_split  # noqa: E402

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


def _ranks(values: np.ndarray) -> np.ndarray:
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


def age_sex_stratified_auc(scores, labels, sex, age_bin) -> float:
    """Macro AUC over (sex, age_bin) strata -- Delphi-2M 'age-sex-stratified AUC'."""
    aucs = []
    for s in np.unique(sex):
        for b in np.unique(age_bin):
            m = (sex == s) & (age_bin == b)
            if int(m.sum()) < 2:
                continue
            a = auc_mann_whitney(scores[m], labels[m])
            if np.isfinite(a):
                aucs.append(float(a))
    return float(np.mean(aucs)) if aucs else float("nan")


def sex_age_stratified_auc(scores, labels, sex, age_bin, target_sex) -> float:
    """Age-stratified macro AUC restricted to one sex (0=female, 1=male)."""
    aucs = []
    for b in np.unique(age_bin):
        m = (sex == target_sex) & (age_bin == b)
        if int(m.sum()) < 2:
            continue
        a = auc_mann_whitney(scores[m], labels[m])
        if np.isfinite(a):
            aucs.append(float(a))
    return float(np.mean(aucs)) if aucs else float("nan")


def count_train_occurrences(train_bin: Path, diag_token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """Count occurrences (events) of each diagnosis token in the training split."""
    data = np.memmap(train_bin, dtype=np.uint32, mode="r").reshape(-1, 3)
    tokens = data[:, 2].astype(np.int64) + 1
    counts = np.zeros(vocab_size, dtype=np.int64)
    valid = (tokens >= 0) & (tokens < vocab_size)
    np.add.at(counts, tokens[valid], 1)
    return counts[np.asarray(diag_token_ids, dtype=np.int64)]


def load_diag_tokens(vocab_csv: Path) -> list[dict]:
    rows = []
    with vocab_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("event_type") or "").strip() != "diagnosis":
                continue
            code = (r.get("code_norm") or "").strip().upper()
            if not code:
                continue
            rows.append({"token_id": int(r["token_id"]), "token_key": r.get("token_key", ""), "code_norm": code})
    rows.sort(key=lambda r: r["token_id"])
    return rows


def build_first_occurrence(data, p2i, n_patients, diag_token_ids, vocab_size) -> np.ndarray:
    n_diag = len(diag_token_ids)
    token_to_col = np.full(vocab_size, -1, dtype=np.int64)
    token_to_col[np.asarray(diag_token_ids, dtype=np.int64)] = np.arange(n_diag)
    first_age = np.full((n_patients, n_diag), np.inf, dtype=np.float32)
    for pid in range(n_patients):
        start, length = p2i[pid]
        if length == 0:
            continue
        rows = data[int(start):int(start) + int(length)]
        tokens = rows[:, 2].astype(np.int64) + 1
        cols = token_to_col[tokens]
        valid = cols >= 0
        if valid.any():
            np.minimum.at(first_age[pid], cols[valid], rows[valid, 1].astype(np.float32))
    return first_age


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=REPO / "configs/track_r_v2_2/TRACK_R_v2_2.json")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--vocab-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--min-cases", type=int, default=1)
    args = ap.parse_args()

    diag = load_diag_tokens(args.vocab_csv)
    diag_token_ids = np.asarray([d["token_id"] for d in diag], dtype=np.int64)
    print(f"[vocab] {len(diag)} diagnosis tokens", flush=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = CARoPEHorizonMedTrajectory(CARoPEConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(args.device).eval()
    vocab_size = int(model.config.vocab_size)
    print(f"[model] vocab_size={vocab_size}", flush=True)

    protocol = load_track_r_protocol(args.protocol)
    manifest = validate_track_r_data_manifest(args.data_dir, protocol)
    dynamic_context_length = int(protocol["dynamic_context_length"])
    bos_token_id = int(manifest["dynamic_bos_token_id"])

    data, p2i, static = load_split(args.data_dir, args.split, args.max_patients)
    prefix_ids, anchor_ages = load_track_r_assets(args.data_dir, args.split, args.max_patients)
    n_patients = len(p2i)
    print(f"[data] split={args.split} patients={n_patients} ctx_len={dynamic_context_length}", flush=True)

    first_age = build_first_occurrence(data, p2i, n_patients, diag_token_ids, vocab_size)
    print("[data] first-occurrence matrix built", flush=True)

    sex_col = load_track_r_static_features(args.data_dir, args.split, "legacy-sex-residual", args.max_patients).ravel()
    train_bin = args.data_dir / "train.bin"
    train_occ = count_train_occurrences(train_bin, diag_token_ids, vocab_size) if train_bin.exists() else None
    print("[data] sex + training-occurrence counts ready", flush=True)

    patient_ids, dynamic_batch = build_official_left_batch(data, p2i, static, dynamic_context_length, no_event_token_rate=5)
    x_dynamic, age_dynamic, y_dynamic, target_age_dynamic, _ = dynamic_batch
    print(f"[batch] official left batch: {len(patient_ids)} patients", flush=True)

    n_diag = len(diag_token_ids)
    diag_ids_dev = torch.from_numpy(diag_token_ids).to(args.device)
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sex_parts: list[np.ndarray] = []
    age_parts: list[np.ndarray] = []

    for start in range(0, len(patient_ids), args.batch_size):
        stop = min(start + args.batch_size, len(patient_ids))
        batch_pids = patient_ids[start:stop].astype(np.int64)
        prefix = torch.as_tensor(prefix_ids[batch_pids], dtype=torch.long)
        anchors = torch.as_tensor(anchor_ages[batch_pids], dtype=torch.float32)
        x, age, y, target_age, static_mask, bos_mask, _ = prepend_static_context(
            x_dynamic[start:stop], age_dynamic[start:stop], y_dynamic[start:stop],
            target_age_dynamic[start:stop], prefix, anchors, bos_token_id,
        )
        x, age, y, target_age, static_mask, bos_mask = [
            v.to(args.device) for v in (x, age, y, target_age, static_mask, bos_mask)
        ]
        keep, pos = last_prediction_positions(x, y)
        if not bool(keep.any()):
            continue
        batch_idx = torch.arange(x.size(0), device=args.device)
        lm_logits = model(
            x, age, None,
            static_token_mask=static_mask,
            static_feature_mask=None,
            bos_token_mask=bos_mask,
        )[0]
        pred_age = age[batch_idx, pos]
        scores = lm_logits[batch_idx, pos][:, diag_ids_dev]  # [B, n_diag]
        keep_np = keep.cpu().numpy()
        pred_age_np = pred_age.cpu().numpy()[keep_np]
        scores_np = scores.cpu().numpy()[keep_np]
        kept_pids = batch_pids[keep_np]
        fa = first_age[kept_pids, :]
        labels_np = (np.isfinite(fa) & (fa > pred_age_np[:, None])).astype(np.int8)
        score_parts.append(scores_np.astype(np.float32))
        label_parts.append(labels_np)
        sex_parts.append(sex_col[kept_pids].astype(np.int8))
        age_parts.append(pred_age_np.astype(np.float32))
        if len(score_parts) % 50 == 0:
            done = sum(int(s.shape[0]) for s in score_parts)
            print(f"[progress] {done}/{len(patient_ids)} patients scored", flush=True)

    scores = np.concatenate(score_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    sex_all = np.concatenate(sex_parts, axis=0)
    age_all = np.concatenate(age_parts, axis=0)
    age_bin = (age_all / 365.25 // 5 * 5).astype(np.int64)
    print(f"[done] scored {scores.shape[0]} patients x {n_diag} tokens", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, tok in enumerate(diag):
        s = scores[:, d].astype(np.float64)
        lab = labels[:, d].astype(np.int8)
        n_pos = int(lab.sum())
        n_neg = int((lab == 0).sum())
        auc = auc_mann_whitney(s, lab)
        ci_low = ci_high = float("nan")
        if np.isfinite(auc) and n_pos >= 2 and n_neg >= 2:
            var = delong_var(s, lab)
            if np.isfinite(var):
                se = float(np.sqrt(var))
                ci_low = float(auc - Z95 * se)
                ci_high = float(auc + Z95 * se)
        letter = tok["code_norm"][0] if tok["code_norm"] else ""
        chapter, chapter_en = ICD10_CHAPTER.get(letter, ("?", ""))
        rows.append({
            "token_id": tok["token_id"],
            "token_key": tok["token_key"],
            "code_norm": tok["code_norm"],
            "chapter": chapter,
            "chapter_en": chapter_en,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "event_rate": float(lab.mean()),
            "train_occurrences": int(train_occ[d]) if train_occ is not None else -1,
            "auc": float(auc) if np.isfinite(auc) else float("nan"),
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "age_sex_auc": float(age_sex_stratified_auc(s, lab, sex_all, age_bin)),
            "male_auc": float(sex_age_stratified_auc(s, lab, sex_all, age_bin, 1)),
            "female_auc": float(sex_age_stratified_auc(s, lab, sex_all, age_bin, 0)),
        })

    kept = [r for r in rows if r["n_positives"] >= args.min_cases and np.isfinite(r["auc"])]
    _write_csv(args.out_dir / "all_disease_lm_auc.csv", rows)
    (args.out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "protocol": str(args.protocol),
        "split": args.split,
        "n_diagnosis_tokens": len(diag),
        "n_patients_scored": int(scores.shape[0]),
        "n_tokens_with_auc": len(kept),
        "min_cases": args.min_cases,
        "score_scale": "raw_lm_logit",
        "label": "future_occurrence_after_last_position",
        "auc_definitions": "auc=pooled; age_sex_auc=age-sex-stratified macro (Delphi Fig2b); male_auc/female_auc=age-stratified within sex",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"n_diagnosis_tokens": len(diag), "n_patients_scored": int(scores.shape[0]), "n_tokens_with_auc": len(kept)}, indent=2), flush=True)
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_cell(r[c]) for c in cols) + "\n")
    print(f"[wrote] {path.name} ({len(rows)} rows)", flush=True)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return repr(v)
    s = str(v)
    if any(ch in s for ch in (",", '"', "\n")):
        return '"' + s.replace('"', '""') + '"'
    return s


if __name__ == "__main__":
    raise SystemExit(main())
