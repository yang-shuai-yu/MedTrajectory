from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "ukb_semantic_multitype_explicit_split"
DEFAULT_OUT = ROOT / "docs" / "icd10_hierarchy_panel.yaml"
DEFAULT_SUMMARY = ROOT / "results" / "icd10_hierarchy_panel" / "icd10_hierarchy_panel_summary.csv"


ICD10_CHAPTERS = [
    ("ICD10_CHAPTER_A_B", "Certain infectious and parasitic diseases", "A00-B99", "A00", "B99"),
    ("ICD10_CHAPTER_C_D_NEOPLASMS", "Neoplasms", "C00-D48", "C00", "D48"),
    ("ICD10_CHAPTER_D_BLOOD", "Diseases of the blood and immune mechanism", "D50-D89", "D50", "D89"),
    ("ICD10_CHAPTER_E", "Endocrine, nutritional and metabolic diseases", "E00-E90", "E00", "E90"),
    ("ICD10_CHAPTER_F", "Mental and behavioural disorders", "F00-F99", "F00", "F99"),
    ("ICD10_CHAPTER_G", "Diseases of the nervous system", "G00-G99", "G00", "G99"),
    ("ICD10_CHAPTER_H_EYE", "Diseases of the eye and adnexa", "H00-H59", "H00", "H59"),
    ("ICD10_CHAPTER_H_EAR", "Diseases of the ear and mastoid process", "H60-H95", "H60", "H95"),
    ("ICD10_CHAPTER_I", "Diseases of the circulatory system", "I00-I99", "I00", "I99"),
    ("ICD10_CHAPTER_J", "Diseases of the respiratory system", "J00-J99", "J00", "J99"),
    ("ICD10_CHAPTER_K", "Diseases of the digestive system", "K00-K93", "K00", "K93"),
    ("ICD10_CHAPTER_L", "Diseases of the skin and subcutaneous tissue", "L00-L99", "L00", "L99"),
    ("ICD10_CHAPTER_M", "Diseases of the musculoskeletal system and connective tissue", "M00-M99", "M00", "M99"),
    ("ICD10_CHAPTER_N", "Diseases of the genitourinary system", "N00-N99", "N00", "N99"),
    ("ICD10_CHAPTER_O", "Pregnancy, childbirth and the puerperium", "O00-O99", "O00", "O99"),
    ("ICD10_CHAPTER_P", "Certain conditions originating in the perinatal period", "P00-P96", "P00", "P96"),
    ("ICD10_CHAPTER_Q", "Congenital malformations and chromosomal abnormalities", "Q00-Q99", "Q00", "Q99"),
    ("ICD10_CHAPTER_R", "Symptoms, signs and abnormal clinical findings", "R00-R99", "R00", "R99"),
    ("ICD10_CHAPTER_S_T", "Injury, poisoning and external causes", "S00-T98", "S00", "T98"),
    ("ICD10_CHAPTER_V_Y", "External causes of morbidity and mortality", "V01-Y98", "V01", "Y98"),
    ("ICD10_CHAPTER_Z", "Factors influencing health status and contact with health services", "Z00-Z99", "Z00", "Z99"),
]


def code_key(code: str) -> tuple[str, int] | None:
    match = re.match(r"^([A-Z])([0-9]{2})", code.strip().upper())
    if not match:
        return None
    return match.group(1), int(match.group(2))


def code_in_range(code: str, start: str, stop: str) -> bool:
    parsed = code_key(code)
    parsed_start = code_key(start)
    parsed_stop = code_key(stop)
    if parsed is None or parsed_start is None or parsed_stop is None:
        return False
    letter, number = parsed
    start_letter, start_number = parsed_start
    stop_letter, stop_number = parsed_stop
    return (start_letter, start_number) <= (letter, number) <= (stop_letter, stop_number)


def normalize_icd10(code: str) -> str:
    code = code.strip().upper()
    code = code.replace(" ", "")
    return code


def three_char(code: str) -> str | None:
    match = re.match(r"^([A-Z][0-9]{2})", code)
    return match.group(1) if match else None


def four_char(code: str) -> str | None:
    match = re.match(r"^([A-Z][0-9]{2})(?:\.?([0-9A-Z]))", code)
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def chapter_for(code: str) -> tuple[str, str, str, str, str] | None:
    for chapter in ICD10_CHAPTERS:
        if code_in_range(code, chapter[3], chapter[4]):
            return chapter
    return None


def auto_block_for(code: str) -> tuple[str, str, str, str] | None:
    parsed = code_key(code)
    chapter = chapter_for(code)
    if parsed is None or chapter is None:
        return None
    letter, number = parsed
    chapter_start = code_key(chapter[3])
    chapter_stop = code_key(chapter[4])
    if chapter_start is None or chapter_stop is None:
        return None
    lower_bound = chapter_start[1] if letter == chapter_start[0] else 0
    upper_bound = chapter_stop[1] if letter == chapter_stop[0] else 99
    start_num = max((number // 10) * 10, lower_bound)
    stop_num = min(start_num + 9, upper_bound)
    start = f"{letter}{start_num:02d}"
    stop = f"{letter}{stop_num:02d}"
    if not code_in_range(start, chapter[3], chapter[4]):
        start = chapter[3] if chapter[3][0] == letter else start
    block_range = start if start == stop else f"{start}-{stop}"
    block_id = f"ICD10_BLOCK_{start}_{stop}".replace("-", "_")
    return block_id, f"ICD-10 block {block_range}", block_range, chapter[0]


def resolve_vocab_csv(data_dir: Path, vocab_csv: Path | None) -> Path:
    if vocab_csv is not None:
        return vocab_csv
    manifest = data_dir / "prepare_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing prepare_manifest.json under {data_dir}; pass --vocab-csv explicitly.")
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    return Path(str(payload["vocab_csv"]))


def load_diagnosis_codes(vocab_csv: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with vocab_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type", "").strip() != "diagnosis":
                continue
            code = normalize_icd10(row.get("code_norm", ""))
            if three_char(code):
                counts[code] += 1
    return counts


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_panel(path: Path, items: list[dict]) -> None:
    lines = [
        "# Auto-generated ICD-10 hierarchy disease panel.",
        "# Levels: chapter, auto_block, 3char, optional 4char.",
        "# auto_block uses observed ICD-10 codes grouped by letter + decade ranges,",
        "# so it is data-driven and should be treated as a practical block layer rather than a complete WHO block catalogue.",
        "",
        "diseases:",
    ]
    for item in items:
        lines.append(f"  - id: {item['id']}")
        lines.append(f"    name: {yaml_quote(item['name'])}")
        lines.append(f"    name_cn: {yaml_quote(item.get('name_cn', item['name']))}")
        lines.append(f"    category: {item['category']}")
        lines.append(f"    level: {item['level']}")
        if item.get("parent_id"):
            lines.append(f"    parent_id: {item['parent_id']}")
        lines.append("    icd10:")
        for spec in item["icd10"]:
            lines.append(f"      - {spec}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "level", "parent_id", "icd10", "token_count", "name"])
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item["id"],
                    "level": item["level"],
                    "parent_id": item.get("parent_id", ""),
                    "icd10": ";".join(item["icd10"]),
                    "token_count": item["token_count"],
                    "name": item["name"],
                }
            )


def build_items(code_counts: Counter[str], include_4char: bool, min_tokens: int) -> list[dict]:
    items: list[dict] = []
    chapter_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    three_counts: Counter[str] = Counter()
    four_counts: Counter[str] = Counter()
    block_meta: dict[str, tuple[str, str, str, str]] = {}

    for code, count in code_counts.items():
        chapter = chapter_for(code)
        block = auto_block_for(code)
        code3 = three_char(code)
        code4 = four_char(code)
        if chapter:
            chapter_counts[chapter[0]] += count
        if block:
            block_meta[block[0]] = block
            block_counts[block[0]] += count
        if code3:
            three_counts[code3] += count
        if include_4char and code4:
            four_counts[code4] += count

    for chapter_id, name, range_spec, _start, _stop in ICD10_CHAPTERS:
        count = chapter_counts[chapter_id]
        if count >= min_tokens:
            items.append(
                {
                    "id": chapter_id,
                    "name": name,
                    "name_cn": name,
                    "category": "icd10_chapter",
                    "level": "chapter",
                    "parent_id": "",
                    "icd10": [range_spec],
                    "token_count": count,
                }
            )

    for block_id in sorted(block_counts):
        count = block_counts[block_id]
        if count < min_tokens:
            continue
        _, name, range_spec, parent_id = block_meta[block_id]
        items.append(
            {
                "id": block_id,
                "name": name,
                "name_cn": name,
                "category": "icd10_auto_block",
                "level": "auto_block",
                "parent_id": parent_id,
                "icd10": [range_spec],
                "token_count": count,
            }
        )

    for code3 in sorted(three_counts):
        count = three_counts[code3]
        if count < min_tokens:
            continue
        block = auto_block_for(code3)
        items.append(
            {
                "id": f"ICD10_3CHAR_{code3}",
                "name": f"ICD-10 {code3}",
                "name_cn": f"ICD-10 {code3}",
                "category": "icd10_3char",
                "level": "3char",
                "parent_id": block[0] if block else "",
                "icd10": [code3],
                "token_count": count,
            }
        )

    for code4 in sorted(four_counts):
        count = four_counts[code4]
        if count < min_tokens:
            continue
        parent = code4.split(".", 1)[0]
        items.append(
            {
                "id": f"ICD10_4CHAR_{code4.replace('.', '_')}",
                "name": f"ICD-10 {code4}",
                "name_cn": f"ICD-10 {code4}",
                "category": "icd10_4char",
                "level": "4char",
                "parent_id": f"ICD10_3CHAR_{parent}",
                "icd10": [code4],
                "token_count": count,
            }
        )

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a data-driven ICD-10 hierarchy panel from the diagnosis vocabulary.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vocab-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--include-4char", action="store_true", help="Include observed 4-character ICD-10 codes.")
    parser.add_argument("--min-tokens", type=int, default=1, help="Minimum diagnosis vocabulary tokens represented by a target.")
    args = parser.parse_args()

    vocab_csv = resolve_vocab_csv(args.data_dir, args.vocab_csv)
    code_counts = load_diagnosis_codes(vocab_csv)
    items = build_items(code_counts, include_4char=args.include_4char, min_tokens=max(1, args.min_tokens))
    write_panel(args.out, items)
    write_summary(args.summary_csv, items)

    by_level = Counter(item["level"] for item in items)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_csv}")
    print("Targets by level:", dict(sorted(by_level.items())))
    print(f"Diagnosis vocab codes observed: {len(code_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
