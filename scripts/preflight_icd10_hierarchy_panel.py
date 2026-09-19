from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "ukb_semantic_multitype_explicit_split"


def count_vocab(vocab_csv: Path) -> tuple[int, int]:
    total = 0
    diagnosis = 0
    with vocab_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"token_id", "event_type", "code_norm"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"vocab CSV missing columns: {sorted(missing)}")
        for row in reader:
            total += 1
            if row.get("event_type", "").strip() == "diagnosis" and row.get("code_norm", "").strip():
                diagnosis += 1
    return total, diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight check for ICD-10 hierarchy panel generation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vocab-csv", type=Path, default=None)
    args = parser.parse_args()

    print(f"Project root: {ROOT}")
    print(f"Data dir: {args.data_dir}")
    if not args.data_dir.exists() and args.vocab_csv is None:
        print("STATUS: BLOCKED")
        print("Reason: data-dir does not exist and --vocab-csv was not provided.")
        print("Next on server:")
        print("  python scripts/preflight_icd10_hierarchy_panel.py --data-dir data/ukb_semantic_multitype_explicit_split")
        print("  python scripts/build_icd10_hierarchy_panel.py --data-dir data/ukb_semantic_multitype_explicit_split")
        return 2

    vocab_csv = args.vocab_csv
    if vocab_csv is None:
        manifest = args.data_dir / "prepare_manifest.json"
        print(f"Manifest: {manifest}")
        if not manifest.exists():
            print("STATUS: BLOCKED")
            print("Reason: prepare_manifest.json is missing.")
            print("Pass --vocab-csv explicitly if the vocabulary CSV exists elsewhere.")
            return 2
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        vocab_csv = Path(str(payload.get("vocab_csv", "")))
        if not vocab_csv.is_absolute():
            vocab_csv = args.data_dir / vocab_csv

    print(f"Vocab CSV: {vocab_csv}")
    if not vocab_csv.exists():
        print("STATUS: BLOCKED")
        print("Reason: vocab CSV does not exist.")
        return 2

    total, diagnosis = count_vocab(vocab_csv)
    print("STATUS: READY")
    print(f"Vocab rows: {total}")
    print(f"Diagnosis rows with code_norm: {diagnosis}")
    print("Next:")
    print("  python scripts/build_icd10_hierarchy_panel.py --data-dir data/ukb_semantic_multitype_explicit_split")
    print("  python src/semantic_delphi_ukb/summarize_icd10_hierarchy_panel.py --data-dir data/ukb_semantic_multitype_explicit_split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
