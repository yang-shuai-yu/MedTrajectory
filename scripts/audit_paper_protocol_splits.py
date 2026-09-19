"""Audit paper-protocol split artifacts and optionally create a freeze manifest.

Examples:
  python scripts/audit_paper_protocol_splits.py --data-dir data/paper_protocol_v1/multitype --splits train,val
  python scripts/audit_paper_protocol_splits.py --data-dir /locked/multitype --splits train,val,test --require-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.paper_protocol_audit import (  # noqa: E402
    audit_disjoint_splits,
    audit_split,
    build_freeze_manifest,
    eid_set_sha256,
    read_eid_csv,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--splits", default="train,val")
    p.add_argument("--require-test", action="store_true")
    p.add_argument("--test-eids-csv", type=Path, default=None, help="frozen EID list that test must match exactly")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diseases-yaml", type=Path, default=None)
    p.add_argument("--checkpoint", action="append", default=[], help="name=/path/to/checkpoint; repeatable")
    p.add_argument("--protocol-json", type=Path, default=None)
    p.add_argument("--asset", action="append", default=[], help="name=/path/to/frozen-asset; repeatable")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if args.require_test and "test" not in splits:
        splits.append("test")
    audits = {
        split: audit_split(args.data_dir, split, require_followup=True, require_nonempty=args.require_test and split == "test")
        for split in splits
    }
    disjoint = audit_disjoint_splits(audits, splits)
    compact_audits = {
        split: {key: value for key, value in audit.items() if key != "eids"}
        for split, audit in audits.items()
    }
    expected_test = None
    if args.test_eids_csv is not None:
        expected = read_eid_csv(args.test_eids_csv)
        observed = [int(eid) for eid in audits.get("test", {}).get("eids", [])]
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        expected_test = {
            "ok": not missing and not unexpected and len(expected) == len(observed),
            "path": str(args.test_eids_csv.resolve()),
            "patient_count": len(expected),
            "eid_set_sha256": eid_set_sha256(expected),
            "observed_eid_set_sha256": eid_set_sha256(observed),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_examples": missing[:20],
            "unexpected_examples": unexpected[:20],
        }
        if not expected_test["ok"]:
            disjoint["ok"] = False
    payload = {"data_dir": str(args.data_dir.resolve()), "splits": compact_audits, "disjoint": disjoint}
    if expected_test is not None:
        payload["expected_test_eids"] = expected_test
    if args.diseases_yaml is not None and args.checkpoint:
        checkpoints = {}
        for item in args.checkpoint:
            name, path = item.split("=", 1)
            checkpoints[name] = Path(path)
        protocol = json.loads(args.protocol_json.read_text(encoding="utf-8")) if args.protocol_json else {}
        assets = {}
        for item in args.asset:
            name, raw_path = item.split("=", 1)
            assets[name] = Path(raw_path)
        payload["freeze_manifest"] = build_freeze_manifest(
            args.data_dir, audits, checkpoints, args.diseases_yaml, protocol, assets
        )
    write_json(args.out, payload)
    print(json.dumps({"ok": bool(disjoint["ok"]), "out": str(args.out), "splits": splits}, indent=2))
    return 0 if disjoint["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
