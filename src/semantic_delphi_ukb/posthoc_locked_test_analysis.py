from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc locked-test significance and calibration analysis.")
    parser.add_argument("--locked-dir", type=Path, default=RESULTS_DIR / "locked_test_horizon_risk")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "locked_test_horizon_risk" / "posthoc_analysis")
    parser.add_argument("--assets-dir", type=Path, default=DOCS_DIR / "report_assets")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    hit_ranks = np.flatnonzero(ranked) + 1
    return float((np.arange(1, positives + 1, dtype=np.float64) / hit_ranks).mean())


def top_decile_stats(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    if labels.size == 0 or labels.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(math.ceil(labels.size * 0.10)))
    top_labels = labels[order[:top_n]]
    top_rate = float(top_labels.mean())
    baseline = float(labels.mean())
    return float(top_labels.sum() / labels.sum()), top_rate, top_rate / baseline if baseline else float("nan")


def load_predictions(path: Path) -> dict[tuple[str, str, str], tuple[float, int]]:
    rows = read_csv(path)
    out = {}
    for row in rows:
        key = (row["patient_index"], f"{float(row['horizon_years']):g}", row["disease_id"])
        out[key] = (float(row["score"]), int(float(row["label"])))
    return out


def model_dirs(locked_dir: Path) -> dict[str, Path]:
    return {
        "MedTrajectory horizon risk": locked_dir / "medtrajectory_test",
        "BERT risk head": locked_dir / "bert_test",
        "Mamba risk head": locked_dir / "mamba_test",
        "MedTrajectory survival-horizon": locked_dir / "survival_test",
    }


def paired_rows(
    main: dict[tuple[str, str, str], tuple[float, int]],
    other: dict[tuple[str, str, str], tuple[float, int]],
    horizon: Optional[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, int]:
    keys = sorted(set(main).intersection(other))
    if horizon is not None:
        keys = [key for key in keys if key[1] == horizon]
    strata_keys = sorted({(key[1], key[2]) for key in keys})
    strata = []
    all_main = []
    all_other = []
    all_labels = []
    mismatched_labels = 0
    for stratum in strata_keys:
        stratum_keys = [key for key in keys if (key[1], key[2]) == stratum]
        labels = []
        main_scores = []
        other_scores = []
        for key in stratum_keys:
            main_score, label = main[key]
            other_score, other_label = other[key]
            if label != other_label:
                mismatched_labels += 1
                continue
            labels.append(label)
            main_scores.append(main_score)
            other_scores.append(other_score)
        if not labels:
            continue
        m = np.asarray(main_scores, dtype=np.float64)
        o = np.asarray(other_scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int8)
        if y.sum() > 0 and y.sum() < y.size:
            strata.append((m, o, y))
        all_main.append(m)
        all_other.append(o)
        all_labels.append(y)
    return strata, np.concatenate(all_main), np.concatenate(all_other), np.concatenate(all_labels), mismatched_labels


def paired_bootstrap(
    strata: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_main: np.ndarray,
    all_other: np.ndarray,
    all_labels: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, float]:
    stratum_main_auc = np.asarray([binary_auc(m, y) for m, _, y in strata], dtype=np.float64)
    stratum_other_auc = np.asarray([binary_auc(o, y) for _, o, y in strata], dtype=np.float64)
    main_auc = float(np.nanmean(stratum_main_auc))
    other_auc = float(np.nanmean(stratum_other_auc))
    diff = float(main_auc - other_auc)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(bootstrap):
        sampled = []
        for main_scores, other_scores, labels in strata:
            n = labels.size
            for _attempt in range(10):
                idx = rng.integers(0, n, n)
                sampled_labels = labels[idx]
                if sampled_labels.sum() > 0 and sampled_labels.sum() < sampled_labels.size:
                    sampled.append(binary_auc(main_scores[idx], sampled_labels) - binary_auc(other_scores[idx], sampled_labels))
                    break
        if sampled:
            diffs.append(float(np.mean(sampled)))
    diff_arr = np.asarray(diffs, dtype=np.float64)
    if diff_arr.size:
        low, high = np.percentile(diff_arr, [2.5, 97.5])
        p = (np.sum(diff_arr <= 0.0) + 1.0) / (diff_arr.size + 1.0)
    else:
        low = high = p = float("nan")
    return {
        "main_auc": main_auc,
        "other_auc": other_auc,
        "auc_diff": diff,
        "diff_ci_low": float(low),
        "diff_ci_high": float(high),
        "one_sided_p_main_le_other": float(p),
        "bootstrap_valid": int(diff_arr.size),
        "paired_rows": int(all_labels.size),
        "paired_positives": int(all_labels.sum()),
    }


def adaptive_calibration(scores: np.ndarray, labels: np.ndarray, bins: int) -> list[dict]:
    order = np.argsort(scores, kind="mergesort")
    chunks = np.array_split(order, bins)
    rows = []
    for idx, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        rows.append(
            {
                "bin": idx,
                "count": int(chunk.size),
                "score_min": float(scores[chunk].min()),
                "score_max": float(scores[chunk].max()),
                "mean_score": float(scores[chunk].mean()),
                "event_rate": float(labels[chunk].mean()),
            }
        )
    return rows


def summarize_predictions(model: str, pred: dict[tuple[str, str, str], tuple[float, int]], bins: int) -> tuple[list[dict], list[dict]]:
    rows = []
    cal_rows = []
    for horizon in ["5", "10"]:
        keys = [key for key in sorted(pred) if key[1] == horizon]
        scores = np.asarray([pred[key][0] for key in keys], dtype=np.float64)
        labels = np.asarray([pred[key][1] for key in keys], dtype=np.int8)
        auc = binary_auc(scores, labels)
        cap, top_rate, lift = top_decile_stats(scores, labels)
        rows.append(
            {
                "model": model,
                "horizon_years": horizon,
                "n": int(labels.size),
                "positives": int(labels.sum()),
                "auc": auc,
                "average_precision": average_precision(scores, labels),
                "brier": float(np.mean((scores - labels) ** 2)),
                "top_decile_capture": cap,
                "top_decile_event_rate": top_rate,
                "top_decile_lift": lift,
                "event_rate": float(labels.mean()),
            }
        )
        for cal in adaptive_calibration(scores, labels, bins):
            cal.update({"model": model, "horizon_years": horizon})
            cal_rows.append(cal)
    return rows, cal_rows


def find_font() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def font(size: int, bold: bool = False):
    path = find_font()
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_calibration_curve(path: Path, cal_rows: list[dict], title: str, horizon: str) -> None:
    selected = [row for row in cal_rows if row["horizon_years"] == horizon]
    models = ["MedTrajectory horizon risk", "BERT risk head", "Mamba risk head", "MedTrajectory survival-horizon"]
    colors = {
        "MedTrajectory horizon risk": (31, 77, 120),
        "BERT risk head": (235, 154, 76),
        "Mamba risk head": (188, 88, 88),
        "MedTrajectory survival-horizon": (61, 153, 142),
    }
    w, h = 1150, 820
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(32, True)
    label_font = font(19)
    tick_font = font(16)
    d.text((45, 32), title, font=title_font, fill=(11, 37, 69))
    d.text((45, 76), f"Adaptive equal-count bins, horizon={horizon}y", font=label_font, fill=(91, 103, 112))
    left, right, top, bottom = 95, w - 280, 135, h - 110
    max_val = 0.0
    for row in selected:
        max_val = max(max_val, float(row["mean_score"]), float(row["event_rate"]))
    max_val = max(0.02, min(1.0, max_val * 1.25))
    for i in range(6):
        val = max_val * i / 5
        x = left + (right - left) * i / 5
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill=(232, 236, 241), width=1)
        d.line((x, top, x, bottom), fill=(232, 236, 241), width=1)
        d.text((left - 70, y - 9), f"{val:.3f}", font=tick_font, fill=(91, 103, 112))
        d.text((x - 20, bottom + 12), f"{val:.3f}", font=tick_font, fill=(91, 103, 112))
    d.line((left, bottom, right, bottom), fill=(70, 83, 95), width=2)
    d.line((left, top, left, bottom), fill=(70, 83, 95), width=2)
    d.line((left, bottom, right, top), fill=(120, 130, 140), width=2)
    d.text((left + 220, h - 45), "Mean predicted risk", font=label_font, fill=(40, 48, 58))
    d.text((20, top + 250), "Observed event rate", font=label_font, fill=(40, 48, 58))

    for model_idx, model in enumerate(models):
        rows = [row for row in selected if row["model"] == model]
        points = []
        for row in rows:
            x = left + (right - left) * (float(row["mean_score"]) / max_val)
            y = bottom - (bottom - top) * (float(row["event_rate"]) / max_val)
            points.append((x, y))
        color = colors[model]
        if len(points) > 1:
            d.line(points, fill=color, width=4)
        for x, y in points:
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        ly = 150 + model_idx * 34
        d.line((right + 35, ly + 8, right + 70, ly + 8), fill=color, width=5)
        d.text((right + 82, ly - 2), model, font=tick_font, fill=(40, 48, 58))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    preds = {
        model: load_predictions(path / "raw_predictions.csv")
        for model, path in model_dirs(args.locked_dir).items()
        if (path / "raw_predictions.csv").exists()
    }
    main_pred = preds["MedTrajectory horizon risk"]

    sig_rows = []
    for other_model in ["BERT risk head", "Mamba risk head", "MedTrajectory survival-horizon"]:
        if other_model not in preds:
            continue
        for horizon in [None, "5", "10"]:
            strata, main_scores, other_scores, labels, mismatched_labels = paired_rows(main_pred, preds[other_model], horizon)
            stats = paired_bootstrap(
                strata,
                main_scores,
                other_scores,
                labels,
                bootstrap=args.bootstrap,
                seed=args.seed + len(sig_rows) * 17,
            )
            sig_rows.append(
                {
                    "comparison": f"MedTrajectory horizon risk vs {other_model}",
                    "horizon_years": "overall" if horizon is None else horizon,
                    "n": int(labels.size),
                    "positives": int(labels.sum()),
                    "skipped_label_mismatch": int(mismatched_labels),
                    **stats,
                }
            )

    summary_rows = []
    cal_rows = []
    for model, pred in preds.items():
        model_summary, model_cal = summarize_predictions(model, pred, args.calibration_bins)
        summary_rows.extend(model_summary)
        cal_rows.extend(model_cal)

    write_csv(args.out_dir / "paired_auc_significance.csv", sig_rows)
    write_csv(args.out_dir / "adaptive_calibration_bins.csv", cal_rows)
    write_csv(args.out_dir / "overall_prediction_summary.csv", summary_rows)
    draw_calibration_curve(
        args.assets_dir / "locked_test_calibration_curve_5y.png",
        cal_rows,
        "Locked-Test Calibration Curve",
        "5",
    )
    draw_calibration_curve(
        args.assets_dir / "locked_test_calibration_curve_10y.png",
        cal_rows,
        "Locked-Test Calibration Curve",
        "10",
    )
    (args.out_dir / "STATUS.md").write_text(
        "# Locked-Test Post-hoc Analysis\n\n"
        "Paired bootstrap compares AUC on exactly matched patient-horizon-disease rows. "
        "Calibration curves use equal-count adaptive bins because selected disease events are rare.\n\n"
        "## Files\n\n"
        "- `paired_auc_significance.csv`\n"
        "- `overall_prediction_summary.csv`\n"
        "- `adaptive_calibration_bins.csv`\n"
        "- `docs/report_assets/locked_test_calibration_curve_5y.png`\n"
        "- `docs/report_assets/locked_test_calibration_curve_10y.png`\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "significance_rows": len(sig_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
