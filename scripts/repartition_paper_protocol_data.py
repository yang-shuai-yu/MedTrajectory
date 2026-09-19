"""Repartition an existing paper_protocol_v1 train/val artifact.

This is used when the canonical JSONL is not available on the training host.
It combines the already prepared paper train and validation patients, removes a
frozen test EID set, and deterministically rebuilds train/val/test while
preserving the paper vocabulary and token encoding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.paper_protocol import exact_random_split  # noqa: E402
from semantic_delphi_ukb.paper_protocol_audit import eid_set_sha256  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [int(row["eid"]) for row in rows] if rows and "eid" in rows[0] else [
        int(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    result = set(values)
    if not result or len(result) != len(values):
        raise ValueError("test EID list must be non-empty and unique")
    return result


def read_index(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "eid" not in rows[0] or "num_events" not in rows[0]:
        raise ValueError(f"invalid patient index: {path}")
    return rows


def split_source_patients(profile_dir: Path):
    patients = []
    for split in ("train", "val"):
        data = np.fromfile(profile_dir / f"{split}.bin", dtype=np.uint32).reshape(-1, 3)
        static = np.load(profile_dir / f"{split}_static.npy").astype(np.float32)
        followup = np.load(profile_dir / f"{split}_followup_end_age_days.npy").astype(np.float32)
        index = read_index(profile_dir / f"{split}_patient_index.csv")
        if len(index) != len(static) or len(index) != len(followup):
            raise ValueError(f"{split} arrays are not aligned in {profile_dir}")
        offset = 0
        for row_idx, row in enumerate(index):
            count = int(row["num_events"])
            segment = data[offset : offset + count].copy()
            offset += count
            if len(segment) != count or (len(segment) and not np.all(segment[:, 0] == int(row["eid"]))):
                raise ValueError(f"trajectory/index mismatch for eid={row['eid']} in {profile_dir}/{split}")
            patients.append((int(row["eid"]), segment, static[row_idx].copy(), float(followup[row_idx]), dict(row)))
        if offset != len(data):
            raise ValueError(f"unused trajectory rows in {profile_dir}/{split}.bin")
    eids = [item[0] for item in patients]
    if len(set(eids)) != len(eids):
        raise ValueError(f"duplicate EIDs across source train/val in {profile_dir}")
    return patients


def write_partition(profile_dir: Path, split: str, patients: list[tuple[int, np.ndarray, np.ndarray, float, dict]]) -> dict:
    payload = np.concatenate([item[1] for item in patients], axis=0) if patients else np.empty((0, 3), dtype=np.uint32)
    payload.astype(np.uint32, copy=False).tofile(profile_dir / f"{split}.bin")
    static = np.stack([item[2] for item in patients], axis=0).astype(np.float32) if patients else np.empty((0, 1), dtype=np.float32)
    np.save(profile_dir / f"{split}_static.npy", static)
    np.save(profile_dir / f"{split}_followup_end_age_days.npy", np.asarray([item[3] for item in patients], dtype=np.float32))
    fields = ["row_index", "eid", "num_events", "model_eligible", "birth_anchor_available", "anchor_only", "sex_raw", "ethnicity_raw"]
    with (profile_dir / f"{split}_patient_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, (eid, segment, _static, _followup, source_row) in enumerate(patients):
            output = {field: source_row.get(field, "") for field in fields}
            output.update({"row_index": row_index, "eid": eid, "num_events": len(segment)})
            writer.writerow(output)
    return {
        "patients": len(patients),
        "rows": int(len(payload)),
        "real_events": int(len(payload) - sum(int(item[4].get("anchor_only", 0)) for item in patients)),
        "model_eligible": int(sum(int(item[4].get("model_eligible", 0)) for item in patients)),
        "birth_anchor_available": int(sum(int(item[4].get("birth_anchor_available", 0)) for item in patients)),
    }


def copy_static_profile_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"train.bin", "val.bin", "test.bin", "train_static.npy", "val_static.npy", "test_static.npy", "train_followup_end_age_days.npy", "val_followup_end_age_days.npy", "test_followup_end_age_days.npy", "train_patient_index.csv", "val_patient_index.csv", "test_patient_index.csv", "prepare_manifest.json"}:
            continue
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-eids-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--profiles", default="diagnosis_death,multitype,reduced_multitype")
    parser.add_argument("--protocol-id", default="paper_protocol_v2_locked_test")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    test_eids = load_eids(args.test_eids_csv)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if not profiles:
        raise ValueError("at least one profile is required")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    staging_dir = args.output_dir.with_name(f".{args.output_dir.name}.building")
    if staging_dir.exists():
        raise FileExistsError(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        root_source_manifest = json.loads((args.source_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
        root_manifest = dict(root_source_manifest)
        root_manifest["protocol_id"] = args.protocol_id
        root_manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        root_manifest["source_data"] = {
            "path": str(args.source_dir.resolve()),
            "prepare_manifest_sha256": sha256_file(args.source_dir / "prepare_manifest.json"),
        }
        root_manifest["split"] = {
            **dict(root_source_manifest.get("split", {})),
            "method": "repartition_existing_paper_protocol_train_val_plus_locked_test",
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "test_participants": len(test_eids),
        }
        root_manifest["locked_test_source"] = {
            "path": str(args.test_eids_csv.resolve()),
            "sha256": sha256_file(args.test_eids_csv),
            "eid_set_sha256": eid_set_sha256(list(test_eids)),
            "patients": len(test_eids),
        }
        for profile in profiles:
            source_profile = args.source_dir / profile
            if not source_profile.exists():
                raise FileNotFoundError(source_profile)
            patients = split_source_patients(source_profile)
            all_eids = {item[0] for item in patients}
            unknown = test_eids - all_eids
            if unknown:
                raise ValueError(f"{profile}: {len(unknown)} test EIDs are absent from paper train/val")
            train_val = [item for item in patients if item[0] not in test_eids]
            split = exact_random_split([item[0] for item in train_val], args.seed, args.validation_fraction)
            train_ids, val_ids = split.train, split.validation
            partitions = {
                "train": [item for item in train_val if item[0] in train_ids],
                "val": [item for item in train_val if item[0] in val_ids],
                "test": [item for item in patients if item[0] in test_eids],
            }
            target_profile = staging_dir / profile
            copy_static_profile_files(source_profile, target_profile)
            summaries = {name: write_partition(target_profile, name, items) for name, items in partitions.items()}
            for name in ("longitudinal.bin", "longitudinal_static.npy", "longitudinal_followup_end_age_days.npy", "longitudinal_patient_index.csv", "longitudinal_outcomes.csv"):
                source = source_profile / name
                if source.exists():
                    shutil.copy2(source, target_profile / name)
            profile_manifest = json.loads((source_profile / "prepare_manifest.json").read_text(encoding="utf-8"))
            profile_manifest["protocol_id"] = args.protocol_id
            profile_manifest["source_prepare_manifest_sha256"] = sha256_file(source_profile / "prepare_manifest.json")
            profile_manifest["split"] = {
                **dict(profile_manifest.get("split", {})),
                "method": "repartition_existing_paper_protocol_train_val_plus_locked_test",
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "train_participants": summaries["train"]["patients"],
                "validation_participants": summaries["val"]["patients"],
                "test_participants": summaries["test"]["patients"],
            }
            profile_manifest["split_summaries"] = {**dict(profile_manifest.get("split_summaries", {})), **summaries}
            profile_manifest["locked_test_source"] = root_manifest["locked_test_source"]
            (target_profile / "prepare_manifest.json").write_text(json.dumps(profile_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            root_manifest.setdefault("profiles", {})[profile] = profile_manifest
        (staging_dir / "prepare_manifest.json").write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging_dir.replace(args.output_dir)
    except BaseException:
        shutil.rmtree(staging_dir)
        raise
    print(json.dumps({"status": "READY", "output_dir": str(args.output_dir), "test_patients": len(test_eids), "profiles": profiles}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
