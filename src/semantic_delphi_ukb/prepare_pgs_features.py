from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import DATA_ROOT, EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import DATA_ROOT, EXTERNAL_ROOT

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_PGS_IDS = ["PGS004458", "PGS004197", "PGS004306", "PGS004307"]
DEFAULT_PGS_DIR = Path(str(EXTERNAL_ROOT / "datasets" / "xiangwenjing" / "ukb" / "pgs_final_normalized"))
DEFAULT_PC_CSV = Path(str(EXTERNAL_ROOT / "22009-0.csv"))
DEFAULT_DATA_DIR = Path(str(EXTERNAL_ROOT / "data" / "ukb_semantic_multitype_explicit_split"))
DEFAULT_OUTPUT_DIR = Path(str(DATA_ROOT / "pgs_i21_v1"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align read-only PGS scores and genetic PCs to MedTrajectory patient rows.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pgs-dir", type=Path, default=DEFAULT_PGS_DIR)
    parser.add_argument("--pc-csv", type=Path, default=DEFAULT_PC_CSV)
    parser.add_argument("--pgs-ids", default=",".join(DEFAULT_PGS_IDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def parse_ids(spec: str) -> list[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--pgs-ids must contain unique non-empty PGS IDs")
    return values


def load_patient_rows(data_dir: Path) -> tuple[dict[str, list[int]], dict[int, tuple[str, int]]]:
    split_ids: dict[str, list[int]] = {}
    locations: dict[int, tuple[str, int]] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_patient_index.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: int(row["row_index"]))
        eids = [int(row["eid"]) for row in rows]
        if [int(row["row_index"]) for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"{path} row_index is not contiguous")
        split_ids[split] = eids
        for row_index, eid in enumerate(eids):
            if eid in locations:
                raise RuntimeError(f"eid {eid} appears in multiple splits")
            locations[eid] = (split, row_index)
    return split_ids, locations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path, with_hash: bool) -> dict:
    stat = path.stat()
    payload = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if with_hash:
        payload["sha256"] = sha256(path)
    return payload


def pgs_path(pgs_dir: Path, pgs_id: str) -> Path:
    matches = sorted(pgs_dir.glob(f"{pgs_id}_*_final_Zscore.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one normalized file for {pgs_id}, found {len(matches)}")
    return matches[0]


def finite_float(value: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def fill_pgs_column(
    path: Path,
    column_index: int,
    locations: dict[int, tuple[str, int]],
    raw: dict[str, np.ndarray],
    seen: dict[str, np.ndarray],
) -> int:
    matched = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"IID", "INT_Score"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{path} missing columns {sorted(required)}")
        for row in reader:
            location = locations.get(int(row["IID"]))
            if location is None:
                continue
            value = finite_float(row["INT_Score"])
            if value is None:
                continue
            split, row_index = location
            raw[split][row_index, column_index] = value
            seen[split][row_index, column_index] = True
            matched += 1
    return matched


def fill_pc_columns(
    path: Path,
    start_index: int,
    locations: dict[int, tuple[str, int]],
    raw: dict[str, np.ndarray],
    pc_seen: dict[str, np.ndarray],
) -> int:
    pc_names = [f"22009-0.{index}" for index in range(1, 11)]
    matched = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"eid", *pc_names}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{path} missing PC1-10 columns")
        for row in reader:
            location = locations.get(int(row["eid"]))
            if location is None:
                continue
            values = [finite_float(row[name]) for name in pc_names]
            if any(value is None for value in values):
                continue
            split, row_index = location
            raw[split][row_index, start_index : start_index + 10] = values
            pc_seen[split][row_index] = True
            matched += 1
    return matched


def fit_normalization(
    train_raw: np.ndarray,
    train_pgs_seen: np.ndarray,
    train_pc_seen: np.ndarray,
    pgs_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(train_raw.shape[1], dtype=np.float32)
    std = np.ones(train_raw.shape[1], dtype=np.float32)
    for column in range(train_raw.shape[1]):
        valid = train_pgs_seen[:, column] if column < pgs_count else train_pc_seen
        values = train_raw[valid, column]
        if not len(values):
            raise RuntimeError(f"No training values for feature column {column}")
        mean[column] = values.mean(dtype=np.float64)
        scale = values.std(dtype=np.float64)
        std[column] = 1.0 if scale == 0.0 else scale
    return mean, std


def normalize_split(
    raw: np.ndarray,
    pgs_seen: np.ndarray,
    pc_seen: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pgs_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    pgs_available = pgs_seen.all(axis=1)
    features = np.zeros((len(raw), raw.shape[1] + 2), dtype=np.float32)
    valid_numeric = np.column_stack([pgs_seen, np.repeat(pc_seen[:, None], 10, axis=1)])
    scaled = (raw - mean) / std
    scaled[~valid_numeric] = 0.0
    features[:, : raw.shape[1]] = scaled
    features[:, -2] = pgs_available.astype(np.float32)
    features[:, -1] = pc_seen.astype(np.float32)
    return features, pgs_available


def main() -> int:
    args = build_parser().parse_args()
    pgs_ids = parse_ids(args.pgs_ids)
    split_ids, locations = load_patient_rows(args.data_dir)
    numeric_count = len(pgs_ids) + 10
    raw = {split: np.zeros((len(eids), numeric_count), dtype=np.float32) for split, eids in split_ids.items()}
    pgs_seen = {split: np.zeros((len(eids), len(pgs_ids)), dtype=bool) for split, eids in split_ids.items()}
    pc_seen = {split: np.zeros(len(eids), dtype=bool) for split, eids in split_ids.items()}

    sources = []
    matches = {}
    for column_index, pgs_id in enumerate(pgs_ids):
        path = pgs_path(args.pgs_dir, pgs_id)
        sources.append(source_record(path, with_hash=True))
        matches[pgs_id] = fill_pgs_column(path, column_index, locations, raw, pgs_seen)
    sources.append(source_record(args.pc_csv, with_hash=False))
    pc_matches = fill_pc_columns(args.pc_csv, len(pgs_ids), locations, raw, pc_seen)

    mean, std = fit_normalization(raw["train"], pgs_seen["train"], pc_seen["train"], len(pgs_ids))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    coverage = {}
    for split in ("train", "val", "test"):
        features, available = normalize_split(raw[split], pgs_seen[split], pc_seen[split], mean, std, len(pgs_ids))
        np.save(args.output_dir / f"{split}_pgs.npy", features)
        np.save(args.output_dir / f"{split}_pgs_available.npy", available)
        coverage[split] = {
            "patients": len(split_ids[split]),
            "pgs_complete": int(available.sum()),
            "pgs_coverage": float(available.mean()),
            "pc_complete": int(pc_seen[split].sum()),
            "pc_coverage": float(pc_seen[split].mean()),
        }

    manifest = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "pgs_ids": pgs_ids,
        "feature_order": pgs_ids + [f"PC{index}" for index in range(1, 11)] + ["pgs_available", "pc_available"],
        "numeric_train_mean": mean.tolist(),
        "numeric_train_std": std.tolist(),
        "coverage": coverage,
        "source_matches": {**matches, "PC1_10": pc_matches},
        "read_only_sources": sources,
        "normalization": "Numeric columns standardized using available train-split rows only; missing values are zero with masks.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
