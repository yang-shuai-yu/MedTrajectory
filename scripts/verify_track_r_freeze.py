"""Verify every file bound into a Track R validation freeze manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    args = p.parse_args(argv)
    payload = json.loads(args.freeze.read_text(encoding="utf-8-sig"))
    if payload.get("manifest_version") != "track_r_validation_freeze_v2":
        raise ValueError("unsupported Track R freeze manifest")
    checked = 0
    for name in ("protocol", "data_manifest", "field_audit", "capacity_report", "validation_landmarks"):
        item = payload[name]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"freeze hash mismatch: {name}")
        checked += 1
    for name, item in payload["models"].items():
        for path_key, hash_key in (("checkpoint", "checkpoint_sha256"), ("validation_rows", "validation_rows_sha256")):
            if sha256(Path(item[path_key])) != item[hash_key]:
                raise ValueError(f"freeze hash mismatch: {name}/{path_key}")
            checked += 1
    print(json.dumps({"verified": True, "files": checked, "test_result_class": payload["test_result_class"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
