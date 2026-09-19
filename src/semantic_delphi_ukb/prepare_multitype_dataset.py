from __future__ import annotations

import argparse
import csv
import json
import random
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from semantic_delphi_ukb.layout import resolve_multitype_paths, resolve_project_layout


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_REFERENCE_MAPPING = (
    REPO_DIR / "outputs" / "icd_reference_prior_1024" / "icd2019_en" / "reference_token_mapping.csv"
)
DEFAULT_REFERENCE_EMBEDDINGS = (
    REPO_DIR / "outputs" / "icd_reference_prior_1024" / "icd2019_en" / "reference_token_embeddings_64d.npy"
)
ICD10_SEMANTIC_EVENT_TYPES = {"diagnosis", "cancer", "death"}
LEGACY_CODE_ALIASES: Dict[str, str] = {
    "A90": "A97",
    "A91": "A97.2",
    "B59": "B48.5",
    "I84": "K64",
    "C42": "C96",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_eid_allowlist(path: Optional[Path]) -> Optional[set[int]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if "eid" in first_line.lower():
            reader = csv.DictReader(handle)
            return {int(row["eid"]) for row in reader if row.get("eid", "").strip()}
        reader = csv.reader(handle)
        return {int(row[0]) for row in reader if row and row[0].strip()}


def load_explicit_split_lookup(
    train_path: Optional[Path],
    val_path: Optional[Path],
    test_path: Optional[Path],
) -> Dict[int, str]:
    split_lookup: Dict[int, str] = {}
    for split_name, split_path in (("train", train_path), ("val", val_path), ("test", test_path)):
        eid_set = load_eid_allowlist(split_path)
        if eid_set is None:
            continue
        overlap = [eid for eid in eid_set if eid in split_lookup]
        if overlap:
            raise RuntimeError(f"Found overlapping eids across explicit split files, first overlap eid={overlap[0]}")
        for eid in eid_set:
            split_lookup[eid] = split_name
    return split_lookup


def write_uint32_bin(path: Path, payload: array) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        payload.tofile(handle)
    return len(payload) // 3


def build_parser() -> argparse.ArgumentParser:
    project_layout = resolve_project_layout()
    multitype_paths = resolve_multitype_paths(project_layout, data_version="v1", experiment="exp2")

    parser = argparse.ArgumentParser(
        description="Convert multitype exp-token JSONL plus static CSV into Delphi-compatible .bin and static matrices."
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--data-version", type=str, default="v1")
    parser.add_argument("--experiment", type=str, default="exp2")
    parser.add_argument("--input-jsonl", type=Path, default=multitype_paths.tokens_exp2_diag_proc_cancer_death_jsonl)
    parser.add_argument("--static-csv", type=Path, default=multitype_paths.static_features_v1_csv)
    parser.add_argument("--vocab-csv", type=Path, default=multitype_paths.dynamic_token_vocab_csv)
    parser.add_argument("--output-dir", type=Path, default=multitype_paths.experiment_dir)
    parser.add_argument("--labels-output", type=Path, default=None)
    parser.add_argument("--semantic-output", type=Path, default=None)
    parser.add_argument("--semantic-alignment-csv", type=Path, default=None)
    parser.add_argument("--reference-mapping-csv", type=Path, default=DEFAULT_REFERENCE_MAPPING)
    parser.add_argument("--reference-embedding-npy", type=Path, default=DEFAULT_REFERENCE_EMBEDDINGS)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-events", type=int, default=1)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--eid-allowlist-csv", type=Path, default=None)
    parser.add_argument("--train-eid-csv", type=Path, default=None)
    parser.add_argument("--val-eid-csv", type=Path, default=None)
    parser.add_argument("--test-eid-csv", type=Path, default=None)
    return parser


def load_static_rows(path: Path) -> Dict[int, dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        out: Dict[int, dict] = {}
        for row in reader:
            out[int(row["eid"])] = row
    return out


def load_labels_from_vocab(path: Path) -> List[str]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["token_id"]), row["token_key"]))
    max_id = max(token_id for token_id, _ in rows)
    labels = [""] * (max_id + 1)
    labels[0] = "Padding"
    labels[1] = "No event"
    for token_id, token_key in rows:
        labels[token_id] = token_key
    return labels


def load_vocab_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_reference_mapping(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: Dict[str, int] = {}
    for row in rows:
        code = (row.get("code_token") or "").strip()
        if code and code not in mapping:
            mapping[code] = int(row["embedding_row"])
    return mapping


def resolve_reference_code(code_norm: str, reference_lookup: Dict[str, int]) -> Tuple[str, str]:
    code_norm = (code_norm or "").strip().upper()
    if not code_norm:
        return "", "missing_code_norm"
    if code_norm in reference_lookup:
        return code_norm, "exact"
    alias = LEGACY_CODE_ALIASES.get(code_norm, "")
    if alias and alias in reference_lookup:
        return alias, f"alias:{code_norm}->{alias}"
    return "", "missing_reference"


def build_semantic_matrix(
    vocab_rows: Sequence[dict],
    reference_lookup: Dict[str, int],
    reference_matrix: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    if reference_matrix.ndim != 2:
        raise ValueError(f"reference embedding matrix must be 2D, got shape={reference_matrix.shape}")

    vocab_size = max(int(row["token_id"]) for row in vocab_rows) + 1
    semantic_matrix = np.zeros((vocab_size, int(reference_matrix.shape[1])), dtype=np.float32)
    alignment_rows: List[dict] = []

    for row in vocab_rows:
        token_id = int(row["token_id"])
        token_key = (row.get("token_key") or "").strip()
        event_type = (row.get("event_type") or "").strip()
        code_norm = (row.get("code_norm") or "").strip().upper()

        if token_id in (0, 1):
            alignment_rows.append(
                {
                    "token_id": token_id,
                    "token_key": token_key,
                    "event_type": event_type,
                    "code_norm": code_norm,
                    "mapped_reference_code": "",
                    "has_embedding": 0,
                    "note": "special_token",
                }
            )
            continue

        if event_type not in ICD10_SEMANTIC_EVENT_TYPES:
            alignment_rows.append(
                {
                    "token_id": token_id,
                    "token_key": token_key,
                    "event_type": event_type,
                    "code_norm": code_norm,
                    "mapped_reference_code": "",
                    "has_embedding": 0,
                    "note": "non_icd_event_zero_init",
                }
            )
            continue

        mapped_code, note = resolve_reference_code(code_norm, reference_lookup)
        has_embedding = int(bool(mapped_code))
        if has_embedding:
            semantic_matrix[token_id] = reference_matrix[reference_lookup[mapped_code]]
        alignment_rows.append(
            {
                "token_id": token_id,
                "token_key": token_key,
                "event_type": event_type,
                "code_norm": code_norm,
                "mapped_reference_code": mapped_code,
                "has_embedding": has_embedding,
                "note": note,
            }
        )

    return semantic_matrix, alignment_rows


def write_alignment_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "token_id",
                "token_key",
                "event_type",
                "code_norm",
                "mapped_reference_code",
                "has_embedding",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def static_fieldnames() -> List[str]:
    return [
        "sex_id",
        "ethnicity_id",
        "height_cm",
        "weight_kg",
        "bmi",
        "age_recruit",
        "height_missing",
        "weight_missing",
        "bmi_missing",
        "age_recruit_missing",
    ]


def static_row_to_vector(row: dict) -> np.ndarray:
    def f(name: str) -> float:
        value = (row.get(name) or "").strip()
        return 0.0 if value == "" else float(value)

    vector = np.asarray(
        [
            f("sex_id"),
            f("ethnicity_id"),
            f("height_cm"),
            f("weight_kg"),
            f("bmi"),
            f("age_recruit"),
            f("height_missing"),
            f("weight_missing"),
            f("bmi_missing"),
            f("age_recruit_missing"),
        ],
        dtype=np.float32,
    )
    return vector


def compute_static_normalization(rows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.vstack(rows)
    numeric_idx = np.asarray([2, 3, 4, 5], dtype=np.int64)
    mean = np.zeros(mat.shape[1], dtype=np.float32)
    std = np.ones(mat.shape[1], dtype=np.float32)
    for idx in numeric_idx:
        values = mat[:, idx]
        mask_idx = idx + 4
        valid = mat[:, mask_idx] < 0.5
        if valid.any():
            mean[idx] = float(values[valid].mean())
            std_val = float(values[valid].std())
            std[idx] = 1.0 if std_val == 0.0 else std_val
    return mean, std


def normalize_static_vector(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = vector.copy()
    for idx in [2, 3, 4, 5]:
        if out[idx + 4] < 0.5:
            out[idx] = (out[idx] - mean[idx]) / std[idx]
        else:
            out[idx] = 0.0
    return out.astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    project_layout = resolve_project_layout(args.dataset_root)
    explicit_split_lookup = load_explicit_split_lookup(args.train_eid_csv, args.val_eid_csv, args.test_eid_csv)
    explicit_split_mode = bool(explicit_split_lookup)
    eid_allowlist = load_eid_allowlist(args.eid_allowlist_csv)

    static_lookup = load_static_rows(args.static_csv)
    vocab_rows = load_vocab_rows(args.vocab_csv)
    labels = load_labels_from_vocab(args.vocab_csv)
    labels_output = args.labels_output or (args.output_dir / "labels.csv")
    semantic_output = args.semantic_output or (args.output_dir / "semantic_input_embeddings_64d.npy")
    semantic_alignment_csv = args.semantic_alignment_csv or (args.output_dir / "semantic_token_alignment.csv")
    reference_lookup = load_reference_mapping(args.reference_mapping_csv)
    reference_matrix = np.load(args.reference_embedding_npy).astype(np.float32)
    semantic_matrix, alignment_rows = build_semantic_matrix(vocab_rows, reference_lookup, reference_matrix)

    payloads = {"train": array("I"), "val": array("I"), "test": array("I")}
    patient_rows = {"train": [], "val": [], "test": []}
    static_raw = {"train": [], "val": [], "test": []}

    rng = random.Random(args.seed)
    total_patients = 0
    kept_patients = 0
    skipped_allowlist = 0
    skipped_split = 0
    dropped_min_events = 0

    for record in iter_jsonl(args.input_jsonl):
        total_patients += 1
        if args.max_patients and total_patients > args.max_patients:
            break

        eid = int(record["eid"])
        if eid_allowlist is not None and eid not in eid_allowlist:
            skipped_allowlist += 1
            continue
        if explicit_split_mode and eid not in explicit_split_lookup:
            skipped_split += 1
            continue

        tokens = [int(x) for x in record.get("tokens", [])]
        ages = [int(x) for x in record.get("ages", [])]
        if len(tokens) != len(ages):
            raise RuntimeError(f"Token/age length mismatch for eid={eid}")
        if len(tokens) < args.min_events:
            dropped_min_events += 1
            continue

        static_row = static_lookup.get(eid)
        if static_row is None:
            raise RuntimeError(f"Missing static row for eid={eid}")

        if explicit_split_mode:
            split_name = explicit_split_lookup[eid]
        else:
            split_name = "val" if rng.random() < args.val_ratio else "train"

        kept_patients += 1
        static_raw[split_name].append(static_row_to_vector(static_row))
        patient_rows[split_name].append({"eid": eid, "num_events": len(tokens)})
        for age_days, token_id in sorted(zip(ages, tokens), key=lambda item: (item[0], item[1])):
            raw_token_id = int(token_id) - 1
            payloads[split_name].extend((eid, int(age_days), raw_token_id))

    if not static_raw["train"]:
        raise RuntimeError("No training patients collected; cannot compute static normalization.")

    train_mean, train_std = compute_static_normalization(static_raw["train"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_output.write_text("\n".join(labels) + "\n", encoding="utf-8")
    np.save(semantic_output, semantic_matrix)
    write_alignment_csv(semantic_alignment_csv, alignment_rows)

    norm_stats = {
        "feature_order": static_fieldnames(),
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
    }
    (args.output_dir / "static_norm.json").write_text(json.dumps(norm_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    for split_name in ["train", "val", "test"]:
        rows = static_raw[split_name]
        if rows:
            mat = np.vstack([normalize_static_vector(row, train_mean, train_std) for row in rows]).astype(np.float32)
        else:
            mat = np.zeros((0, len(static_fieldnames())), dtype=np.float32)
        np.save(args.output_dir / f"{split_name}_static.npy", mat)

        with (args.output_dir / f"{split_name}_patient_index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["row_index", "eid", "num_events"])
            writer.writeheader()
            for row_index, row in enumerate(patient_rows[split_name]):
                writer.writerow({"row_index": row_index, **row})

    train_rows = write_uint32_bin(args.output_dir / "train.bin", payloads["train"])
    val_rows = write_uint32_bin(args.output_dir / "val.bin", payloads["val"])
    test_rows = write_uint32_bin(args.output_dir / "test.bin", payloads["test"]) if (explicit_split_mode or patient_rows["test"]) else None

    manifest = {
        "dataset_root": str(project_layout.dataset_root),
        "input_jsonl": str(args.input_jsonl),
        "static_csv": str(args.static_csv),
        "vocab_csv": str(args.vocab_csv),
        "output_dir": str(args.output_dir),
        "labels_output": str(labels_output),
        "semantic_output": str(semantic_output),
        "semantic_alignment_csv": str(semantic_alignment_csv),
        "reference_mapping_csv": str(args.reference_mapping_csv),
        "reference_embedding_npy": str(args.reference_embedding_npy),
        "semantic_embedding_dim": int(semantic_matrix.shape[1]),
        "split_mode": "explicit" if explicit_split_mode else "allowlist_random_val",
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "min_events": args.min_events,
        "total_patients_seen": total_patients if not args.max_patients else min(total_patients, args.max_patients),
        "kept_patients": kept_patients,
        "skipped_allowlist": skipped_allowlist,
        "skipped_split": skipped_split,
        "dropped_min_events": dropped_min_events,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_patients": len(patient_rows["train"]),
        "val_patients": len(patient_rows["val"]),
        "test_patients": len(patient_rows["test"]),
        "semantic_tokens_total": int(semantic_matrix.shape[0]),
        "semantic_tokens_with_reference": int(sum(int(row["has_embedding"]) for row in alignment_rows)),
        "semantic_tokens_without_reference": int(sum(1 for row in alignment_rows if row["note"] == "missing_reference")),
        "procedure_or_other_zero_init_tokens": int(sum(1 for row in alignment_rows if row["note"] == "non_icd_event_zero_init")),
    }
    (args.output_dir / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
