"""Extend an existing locked-test manifest with shared-landmark assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--landmark-manifest", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    for path in (args.landmark_manifest, args.coverage_audit):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = dict(payload.get("frozen_files", {}))
    for item in frozen.values():
        path = Path(item.get("path", ""))
        if path.is_file():
            item["sha256"] = sha256_file(path)
    for name, path in (("shared_test_landmarks", args.landmark_manifest), ("shared_test_landmark_coverage_audit", args.coverage_audit)):
        frozen[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    payload["frozen_files"] = frozen
    protocol = dict(payload.get("protocol", {}))
    protocol["landmark_selection"] = "shared_test_manifest_intersection_v1"
    protocol["shared_landmark_manifest"] = str(args.landmark_manifest.resolve())
    protocol["shared_landmark_manifest_sha256"] = sha256_file(args.landmark_manifest)
    protocol["shared_landmark_count"] = int(json.loads(args.landmark_manifest.read_text(encoding="utf-8"))["landmark_count"])
    payload["protocol"] = protocol
    payload["parent_freeze_manifest"] = {"path": str(args.input.resolve()), "sha256": sha256_file(args.input)}
    payload["ok"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "shared_landmark_count": protocol["shared_landmark_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
