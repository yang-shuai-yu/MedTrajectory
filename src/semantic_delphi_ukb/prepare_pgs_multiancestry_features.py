from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import DATA_ROOT
except ImportError:  # executed from inside the source tree
    from paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT_DIR = Path(str(DATA_ROOT / "pgs_i21_v1"))
DEFAULT_OUTPUT_DIR = Path(str(DATA_ROOT / "pgs_i21_euasian_v1"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Eu+Asian stratified PGS features without reading PGS sources again.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def split_complete(available: np.ndarray, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(available).astype(np.int64)
    rng.shuffle(rows)
    val_count = max(1, int(round(len(rows) * val_fraction)))
    if val_count >= len(rows):
        raise ValueError("val_fraction leaves no training rows")
    train_mask = np.zeros(len(available), dtype=bool)
    val_mask = np.zeros(len(available), dtype=bool)
    train_mask[rows[val_count:]] = True
    val_mask[rows[:val_count]] = True
    return train_mask, val_mask


def fit_stats(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0.0] = 1.0
    return mean, std


def transform(features: np.ndarray, available: np.ndarray, old_mean: np.ndarray, old_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    numeric_count = len(old_mean)
    raw = features[:, :numeric_count].astype(np.float64) * old_std + old_mean
    output = features.astype(np.float32, copy=True)
    output[:, :numeric_count] = ((raw - mean) / std).astype(np.float32)
    output[~available, :numeric_count] = 0.0
    return output


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    pgs_ids = list(manifest["pgs_ids"])
    numeric_count = len(pgs_ids) + 10
    old_mean = np.asarray(manifest["numeric_train_mean"], dtype=np.float64)
    old_std = np.asarray(manifest["numeric_train_std"], dtype=np.float64)
    if len(old_mean) != numeric_count or len(old_std) != numeric_count:
        raise RuntimeError("Input manifest numeric statistics do not match the expected PGS+PC feature count")

    features = {split: np.load(args.input_dir / f"{split}_pgs.npy") for split in ("train", "val", "test")}
    available = {split: np.load(args.input_dir / f"{split}_pgs_available.npy").astype(bool) for split in features}
    rng = np.random.default_rng(args.seed)
    eu_train_mask, eu_val_mask = split_complete(available["train"], args.val_fraction, rng)
    asian_train_mask, asian_val_mask = split_complete(available["val"], args.val_fraction, rng)

    eu_raw = features["train"][eu_train_mask, :numeric_count].astype(np.float64) * old_std + old_mean
    asian_raw = features["val"][asian_train_mask, :numeric_count].astype(np.float64) * old_std + old_mean
    eu_mean, eu_std = fit_stats(eu_raw)
    asian_mean, asian_std = fit_stats(asian_raw)
    pooled_mean, pooled_std = fit_stats(np.concatenate([eu_raw, asian_raw], axis=0))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "train_pgs.npy", transform(features["train"], available["train"], old_mean, old_std, eu_mean, eu_std))
    np.save(output / "val_pgs.npy", transform(features["val"], available["val"], old_mean, old_std, asian_mean, asian_std))
    np.save(output / "test_pgs.npy", transform(features["test"], available["test"], old_mean, old_std, pooled_mean, pooled_std))
    for split in ("train", "val", "test"):
        np.save(output / f"{split}_pgs_available.npy", available[split])
    np.save(output / "train_adapter_mask.npy", eu_train_mask)
    np.save(output / "val_adapter_mask.npy", eu_val_mask)
    np.save(output / "asian_train_adapter_mask.npy", asian_train_mask)
    np.save(output / "asian_val_adapter_mask.npy", asian_val_mask)

    result = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output),
        "pgs_ids": pgs_ids,
        "feature_order": manifest["feature_order"],
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "source_groups": {"train": "European", "val": "Asian", "test": "Black_external"},
        "adapter_train_patients": {"European": int(eu_train_mask.sum()), "Asian": int(asian_train_mask.sum())},
        "adapter_val_patients": {"European": int(eu_val_mask.sum()), "Asian": int(asian_val_mask.sum())},
        "coverage": manifest["coverage"],
        "normalization": "Eu and Asian statistics fit on their own adapter-train subsets; Black external test uses pooled Eu+Asian train statistics.",
        "group_numeric_mean": {"European": eu_mean.tolist(), "Asian": asian_mean.tolist(), "pooled": pooled_mean.tolist()},
        "group_numeric_std": {"European": eu_std.tolist(), "Asian": asian_std.tolist(), "pooled": pooled_std.tolist()},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
