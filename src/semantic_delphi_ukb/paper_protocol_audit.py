"""Audits and freeze-manifest helpers for the paper protocol.

The functions in this module are intentionally side-effect free.  They are used
before a locked-test run to prove that the split artifacts, patient identities,
and frozen inputs are aligned.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


REQUIRED_SPLIT_FILES = (
    "{split}.bin",
    "{split}_static.npy",
    "{split}_patient_index.csv",
    "{split}_followup_end_age_days.npy",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eid_sequence_sha256(eids: Sequence[int]) -> str:
    payload = "".join(f"{int(eid)}\n" for eid in eids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def eid_set_sha256(eids: Sequence[int]) -> str:
    return eid_sequence_sha256(sorted(int(eid) for eid in eids))


def split_files(data_dir: Path, split: str) -> dict[str, Path]:
    return {name.format(split=split): data_dir / name.format(split=split) for name in REQUIRED_SPLIT_FILES}


def read_split_eids(index_path: Path) -> list[int]:
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "eid" not in rows[0]:
        raise ValueError(f"{index_path} must contain a non-empty eid column")
    eids = [int(row["eid"]) for row in rows]
    if len(set(eids)) != len(eids):
        raise ValueError(f"{index_path} contains duplicate eids")
    return eids


def read_eid_csv(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "eid" in rows[0]:
        eids = [int(row["eid"]) for row in rows]
    else:
        eids = [int(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not eids or len(set(eids)) != len(eids):
        raise ValueError(f"{path} must contain non-empty unique EIDs")
    return eids


def audit_split(
    data_dir: Path,
    split: str,
    require_followup: bool = True,
    require_nonempty: bool = False,
) -> dict[str, object]:
    files = split_files(data_dir, split)
    missing = [str(path) for name, path in files.items() if not path.exists() and (require_followup or "followup" not in name)]
    if missing:
        return {"split": split, "data_dir": str(data_dir), "ok": False, "missing": missing}
    index_path = files[f"{split}_patient_index.csv"]
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    eids = read_split_eids(index_path)
    import numpy as np

    static = np.load(files[f"{split}_static.npy"], mmap_mode="r")
    followup = np.load(files[f"{split}_followup_end_age_days.npy"], mmap_mode="r")
    bin_size = files[f"{split}.bin"].stat().st_size
    if bin_size % (4 * 3) != 0:
        raise ValueError(f"{files[f'{split}.bin']} is not a uint32 [N,3] stream")
    errors = []
    event_counts = None
    if "num_events" not in index_rows[0]:
        errors.append("patient_index_missing_num_events")
    else:
        event_counts = [int(row["num_events"]) for row in index_rows]
        if any(count <= 0 for count in event_counts):
            errors.append("patient_index_has_nonpositive_num_events")
    expected_row_indices = list(range(len(index_rows)))
    if "row_index" in index_rows[0] and [int(row["row_index"]) for row in index_rows] != expected_row_indices:
        errors.append("patient_index_row_order_mismatch")
    trajectory_aligned = False
    if event_counts is not None and sum(event_counts) == bin_size // (4 * 3):
        trajectory = np.memmap(files[f"{split}.bin"], dtype=np.uint32, mode="r").reshape(-1, 3)
        offset = 0
        trajectory_aligned = True
        for eid, count in zip(eids, event_counts):
            if not np.all(trajectory[offset : offset + count, 0] == eid):
                trajectory_aligned = False
                break
            offset += count
    else:
        errors.append("trajectory_row_count_mismatch")
    if not trajectory_aligned:
        errors.append("trajectory_eids_do_not_match_patient_index")
    checks = {
        "patient_count": len(eids),
        "static_rows": int(static.shape[0]),
        "followup_rows": int(followup.shape[0]),
        "static_columns": int(static.shape[1]) if static.ndim == 2 else None,
        "bin_rows": int(bin_size // (4 * 3)),
        "indexed_event_rows": int(sum(event_counts)) if event_counts is not None else None,
        "eid_sequence_sha256": eid_sequence_sha256(eids),
        "eid_set_sha256": eid_set_sha256(eids),
        "sha256": {name: sha256_file(path) for name, path in files.items() if path.exists()},
        "errors": sorted(set(errors)),
    }
    checks["aligned"] = (
        checks["patient_count"] == checks["static_rows"] == checks["followup_rows"]
        and trajectory_aligned
        and not errors
    )
    if require_nonempty and checks["patient_count"] == 0:
        checks["aligned"] = False
    return {"split": split, "data_dir": str(data_dir), "ok": bool(checks["aligned"]), **checks, "eids": eids}


def audit_disjoint_splits(audits: Mapping[str, Mapping[str, object]], splits: Sequence[str] = ("train", "val", "test")) -> dict[str, object]:
    overlap: dict[str, list[int]] = {}
    for left_idx, left in enumerate(splits):
        left_eids = set(int(value) for value in audits.get(left, {}).get("eids", []))
        for right in splits[left_idx + 1 :]:
            shared = sorted(left_eids & set(int(value) for value in audits.get(right, {}).get("eids", [])))
            if shared:
                overlap[f"{left}:{right}"] = shared[:20]
    return {"ok": not overlap and all(bool(audits.get(split, {}).get("ok")) for split in splits), "overlap_examples": overlap}


def build_freeze_manifest(
    data_dir: Path,
    split_audits: Mapping[str, Mapping[str, object]],
    checkpoints: Mapping[str, Path],
    diseases_yaml: Path,
    protocol: Mapping[str, object],
    frozen_files: Optional[Mapping[str, Path]] = None,
    model_data_dirs: Optional[Mapping[str, Path]] = None,
    model_split_audits: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> dict[str, object]:
    if not diseases_yaml.exists():
        raise FileNotFoundError(diseases_yaml)
    frozen_splits = {
        name: {key: value for key, value in dict(audit).items() if key != "eids"}
        for name, audit in split_audits.items()
    }
    manifest = {
        "protocol": dict(protocol),
        "data_dir": str(data_dir.resolve()),
        "splits": frozen_splits,
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in checkpoints.items()
        },
        "diseases_yaml": {"path": str(diseases_yaml.resolve()), "sha256": sha256_file(diseases_yaml)},
    }
    if frozen_files:
        manifest["frozen_files"] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in frozen_files.items()
        }
    if model_data_dirs:
        manifest["model_data_dirs"] = {
            name: {"data_dir": str(path.resolve())}
            for name, path in model_data_dirs.items()
        }
    if model_split_audits:
        manifest["model_split_audits"] = {
            name: dict(audit)
            for name, audit in model_split_audits.items()
        }
    return manifest


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
