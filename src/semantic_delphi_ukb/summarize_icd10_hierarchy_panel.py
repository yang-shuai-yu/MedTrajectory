from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_delphi_ukb.tte_targets import parse_selected_diseases, token_ids_for_disease  # noqa: E402

DEFAULT_PANEL = ROOT / "docs" / "icd10_hierarchy_panel.yaml"
DEFAULT_OUT = ROOT / "results" / "icd10_hierarchy_panel" / "icd10_hierarchy_panel_token_summary.csv"


def load_token_codes_from_vocab(vocab_csv: Path) -> dict[int, str]:
    token_codes: dict[int, str] = {}
    with vocab_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type", "").strip() == "diagnosis" and row.get("code_norm", "").strip():
                token_codes[int(row["token_id"])] = row["code_norm"].strip().upper()
    return token_codes


def resolve_vocab_csv(data_dir: Path | None, vocab_csv: Path | None) -> Path:
    if vocab_csv is not None:
        return vocab_csv
    if data_dir is None:
        raise ValueError("Pass either --data-dir or --vocab-csv.")
    manifest = data_dir / "prepare_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    return Path(str(payload["vocab_csv"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize and validate an ICD-10 hierarchy panel against a diagnosis vocabulary.")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--vocab-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    vocab_csv = resolve_vocab_csv(args.data_dir, args.vocab_csv)
    diseases = parse_selected_diseases(args.panel)
    token_codes = load_token_codes_from_vocab(vocab_csv)
    ids = {d.disease_id for d in diseases}

    rows = []
    missing_parents = []
    by_level: Counter[str] = Counter()
    empty_targets = []
    for disease in diseases:
        tokens = token_ids_for_disease(disease, token_codes)
        level = disease.level or "unspecified"
        by_level[level] += 1
        if disease.parent_id and disease.parent_id not in ids:
            missing_parents.append((disease.disease_id, disease.parent_id))
        if not tokens:
            empty_targets.append(disease.disease_id)
        rows.append(
            {
                "id": disease.disease_id,
                "name": disease.name,
                "level": level,
                "parent_id": disease.parent_id,
                "category": disease.category,
                "icd10": ";".join(disease.ranges),
                "token_count": len(tokens),
                "token_ids": " ".join(str(t) for t in tokens[:100]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "name", "level", "parent_id", "category", "icd10", "token_count", "token_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")
    print("Targets by level:", dict(sorted(by_level.items())))
    print(f"Empty targets: {len(empty_targets)}")
    print(f"Missing parent links: {len(missing_parents)}")
    if empty_targets[:10]:
        print("First empty targets:", ", ".join(empty_targets[:10]))
    if missing_parents[:10]:
        print("First missing parents:", ", ".join(f"{child}->{parent}" for child, parent in missing_parents[:10]))
    return 1 if missing_parents else 0


if __name__ == "__main__":
    raise SystemExit(main())
