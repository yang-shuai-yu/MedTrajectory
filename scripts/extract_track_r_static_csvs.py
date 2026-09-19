"""Extract Track R instance-0 static fields from a UKB phenotype CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


FIELDS = {
    "31": "31-0.0",
    "21001": "21001-0.0",
    "20116": "20116-0.0",
    "1558": "1558-0.0",
    "21022": "21022-0.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    manifest = args.manifest.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {field_id: output_dir / f"{field_id}-0.csv" for field_id in FIELDS}
    existing = [path for path in paths.values() if path.exists()]
    if manifest.exists() or existing:
        raise FileExistsError("refusing to overwrite existing Track R extraction outputs")

    counts = {field_id: Counter() for field_id in FIELDS}
    row_count = 0
    last_eid = None
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        required = ["eid", *FIELDS.values()]
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"source is missing columns: {missing}")
        indices = {name: header.index(name) for name in required}
        handles = {field_id: path.open("w", encoding="utf-8", newline="") for field_id, path in paths.items()}
        writers = {
            # Match the existing UKB_extract CSV convention byte-for-byte.
            field_id: csv.writer(file_handle, lineterminator="\r\n")
            for field_id, file_handle in handles.items()
        }
        try:
            for field_id, column in FIELDS.items():
                writers[field_id].writerow(["eid", column])
            for row in reader:
                raw_eid = row[indices["eid"]].strip()
                if not raw_eid:
                    raise ValueError(f"missing eid at source row {row_count + 2}")
                eid = str(int(float(raw_eid)))
                if last_eid is not None and int(eid) <= last_eid:
                    raise ValueError(f"source EIDs are not strictly increasing at {eid}")
                last_eid = int(eid)
                row_count += 1
                for field_id, column in FIELDS.items():
                    value = row[indices[column]].strip()
                    writers[field_id].writerow([eid, value])
                    if value:
                        counts[field_id][value] += 1
                if row_count % 10000 == 0:
                    for file_handle in handles.values():
                        file_handle.flush()
                    print(json.dumps({"rows_processed": row_count}), flush=True)
        finally:
            for file_handle in handles.values():
                file_handle.close()

    payload = {
        "source": str(source),
        "source_sha256": sha256(source),
        "source_columns": list(FIELDS.values()),
        "row_count": row_count,
        "eid_order": "strictly_increasing",
        "fields": {
            field_id: {
                "path": str(path),
                "sha256": sha256(path),
                "nonmissing": int(sum(counts[field_id].values())),
                "observed_value_counts": dict(sorted(counts[field_id].items())),
            }
            for field_id, path in paths.items()
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "row_count": row_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
