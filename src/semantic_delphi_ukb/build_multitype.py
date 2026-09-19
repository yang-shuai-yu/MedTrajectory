from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from semantic_delphi_ukb.layout import MultitypePaths, ProjectLayout, resolve_multitype_paths, resolve_project_layout


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_LABELS_PATH = REPO_DIR / "data" / "ukb_simulated_data" / "labels.csv"

SPECIAL_TOKEN_LIMIT = 12
DEATH_MODEL_TOKEN_ID = 1269


@dataclass
class CanonicalEvent:
    event_type: str
    source_field: int
    date_field: Optional[int]
    code_raw: str
    code_norm: str
    token_key: Optional[str]
    event_date_raw: Optional[str]
    age_years: Optional[float]
    age_days: Optional[int]
    time_status: str
    note: str

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "source_field": self.source_field,
            "date_field": self.date_field,
            "code_raw": self.code_raw,
            "code_norm": self.code_norm,
            "token_key": self.token_key,
            "event_date_raw": self.event_date_raw,
            "age_years": self.age_years,
            "age_days": self.age_days,
            "time_status": self.time_status,
            "note": self.note,
        }


class SortedCsvLookup:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("r", encoding="utf-8", newline="")
        self.reader = csv.DictReader(self.handle)
        self.current: Optional[dict] = next(self.reader, None)

    def get(self, eid: int) -> Optional[dict]:
        target = int(eid)
        while self.current is not None and int(self.current["eid"]) < target:
            self.current = next(self.reader, None)
        if self.current is not None and int(self.current["eid"]) == target:
            return self.current
        return None

    def close(self) -> None:
        self.handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical UKB multitype records and exp0/exp1/exp2 token JSONL files."
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--ukb-extract-dir", type=Path, default=None)
    parser.add_argument("--tokens-jsonl", type=Path, default=None)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--data-version", type=str, default="v1")
    parser.add_argument("--experiment", type=str, default="exp2")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--create-dirs", action="store_true")
    return parser


def resolve_tokens_jsonl(project_layout: ProjectLayout, ukb_extract_dir: Path, tokens_jsonl: Optional[Path]) -> Path:
    if tokens_jsonl is not None:
        return tokens_jsonl
    extract_tokens = ukb_extract_dir / "tokens.jsonl"
    if extract_tokens.exists():
        return extract_tokens
    return project_layout.tokens_jsonl


def load_labels(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def normalize_icd3(code: str) -> str:
    code = (code or "").strip().upper()
    match = re.match(r"^([A-Z][0-9A-Z]{2})", code)
    return match.group(1) if match else ""


def extract_code3_from_label(label: str) -> str:
    if not label:
        return ""
    match = re.match(r"^([A-Z][0-9A-Z]{2})", label)
    return match.group(1) if match else ""


def parse_date_ordinal(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    return date.fromisoformat(value).toordinal()


def iter_tokens(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row_value(row: Optional[dict], key: str) -> str:
    if row is None:
        return ""
    return (row.get(key) or "").strip()


def wide_pairs(code_row: Optional[dict], date_row: Optional[dict], code_prefix: str, date_prefix: str) -> List[Tuple[str, str]]:
    if code_row is None:
        return []
    pairs: List[Tuple[str, str]] = []
    for key, value in code_row.items():
        if key == "eid":
            continue
        code = (value or "").strip()
        if not code:
            continue
        date_key = key.replace(code_prefix, date_prefix, 1)
        event_date = row_value(date_row, date_key)
        pairs.append((code, event_date))
    return pairs


def sort_dated_events(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    def sort_key(item: Tuple[str, str]) -> Tuple[str, str]:
        event_date = item[1] or "9999-12-31"
        return (event_date, item[0])

    return sorted(pairs, key=sort_key)


def disease_token_sequence(record: dict, labels: Sequence[str]) -> List[Tuple[str, float, int]]:
    out: List[Tuple[str, float, int]] = []
    for model_token_id, age_years in zip(record.get("tokens", []), record.get("ages", [])):
        model_token_id = int(model_token_id)
        if model_token_id <= SPECIAL_TOKEN_LIMIT or model_token_id == DEATH_MODEL_TOKEN_ID:
            continue
        label = labels[model_token_id] if 0 <= model_token_id < len(labels) else ""
        code3 = extract_code3_from_label(label)
        if not code3:
            continue
        out.append((code3, float(age_years), model_token_id))
    return out


def greedy_align_diagnoses(
    sorted_pairs: List[Tuple[str, str]],
    token_seq: List[Tuple[str, float, int]],
) -> List[Tuple[int, float]]:
    aligned: List[Tuple[int, float]] = []
    token_idx = 0
    for pair_idx, (raw_code, _) in enumerate(sorted_pairs):
        if token_idx >= len(token_seq):
            break
        code3 = normalize_icd3(raw_code)
        next_code3, age_years, _ = token_seq[token_idx]
        if code3 == next_code3:
            aligned.append((pair_idx, age_years))
            token_idx += 1
    return aligned


def infer_birth_ordinal(matched_events: List[Tuple[int, float]], sorted_pairs: List[Tuple[str, str]]) -> Optional[float]:
    anchors: List[float] = []
    for pair_idx, age_years in matched_events:
        _, event_date = sorted_pairs[pair_idx]
        event_ordinal = parse_date_ordinal(event_date)
        if event_ordinal is None:
            continue
        anchors.append(float(event_ordinal) - float(age_years) * 365.25)
    if not anchors:
        return None
    anchors.sort()
    return anchors[len(anchors) // 2]


def age_from_birth_anchor(event_date: str, birth_ordinal: Optional[float]) -> Tuple[Optional[float], Optional[int], str]:
    event_ordinal = parse_date_ordinal(event_date)
    if event_ordinal is None:
        return None, None, "missing_date"
    if birth_ordinal is None:
        return None, None, "missing_birth_anchor"
    age_days = int(round(float(event_ordinal) - float(birth_ordinal)))
    age_years = round(float(age_days) / 365.25, 4)
    return age_years, age_days, "resolved_from_birth_anchor"


def build_static_payload(
    sex_row: Optional[dict],
    ethnicity_row: Optional[dict],
    height_row: Optional[dict],
    weight_row: Optional[dict],
    bmi_row: Optional[dict],
    age_recruit_row: Optional[dict],
) -> dict:
    def numeric_payload(field_id: int, raw: str) -> dict:
        raw = (raw or "").strip()
        if not raw:
            return {"field_id": field_id, "value": None, "missing": True}
        return {"field_id": field_id, "value": float(raw), "missing": False}

    sex_raw = row_value(sex_row, "31-0.0")
    ethnicity_raw = row_value(ethnicity_row, "21000-0.0")
    return {
        "sex": {
            "field_id": 31,
            "value_raw": sex_raw,
            "value_id": None if not sex_raw else int(sex_raw),
        },
        "height_cm": numeric_payload(50, row_value(height_row, "50-0.0")),
        "weight_kg": numeric_payload(21002, row_value(weight_row, "21002-0.0")),
        "bmi": numeric_payload(21001, row_value(bmi_row, "21001-0.0")),
        "age_recruit": numeric_payload(21022, row_value(age_recruit_row, "21022-0.0")),
        "ethnicity": {
            "field_id": 21000,
            "value_raw": ethnicity_raw,
            "value_id": None if not ethnicity_raw else ethnicity_raw,
        },
    }


def make_diagnosis_events(
    sorted_pairs: List[Tuple[str, str]],
    matched_events: List[Tuple[int, float]],
) -> Tuple[List[CanonicalEvent], int]:
    matched_by_idx = {pair_idx: age_years for pair_idx, age_years in matched_events}
    out: List[CanonicalEvent] = []
    matched_count = 0
    for pair_idx, (code_raw, event_date) in enumerate(sorted_pairs):
        code_norm = normalize_icd3(code_raw)
        age_years = matched_by_idx.get(pair_idx)
        age_days = None if age_years is None else int(round(float(age_years) * 365.25))
        time_status = "matched_from_existing_tokens" if age_years is not None else "unresolved_diagnosis_age"
        if age_years is not None:
            matched_count += 1
        out.append(
            CanonicalEvent(
                event_type="diagnosis",
                source_field=41202,
                date_field=41262,
                code_raw=code_raw,
                code_norm=code_norm,
                token_key=f"diag:{code_norm}" if code_norm else None,
                event_date_raw=event_date or None,
                age_years=None if age_years is None else round(float(age_years), 4),
                age_days=age_days,
                time_status=time_status,
                note="aligned_to_existing_tokens" if age_years is not None else "not_matched_in_existing_tokens",
            )
        )
    return out, matched_count


def make_dated_events(
    pairs: List[Tuple[str, str]],
    event_type: str,
    source_field: int,
    date_field: int,
    token_prefix: str,
    code_mode: str,
    birth_ordinal: Optional[float],
) -> List[CanonicalEvent]:
    out: List[CanonicalEvent] = []
    for code_raw, event_date in sort_dated_events(pairs):
        code_norm = normalize_icd3(code_raw) if code_mode == "icd3" else (code_raw or "").strip().upper()
        age_years, age_days, time_status = age_from_birth_anchor(event_date, birth_ordinal)
        out.append(
            CanonicalEvent(
                event_type=event_type,
                source_field=source_field,
                date_field=date_field,
                code_raw=code_raw,
                code_norm=code_norm,
                token_key=f"{token_prefix}:{code_norm}" if code_norm else None,
                event_date_raw=event_date or None,
                age_years=age_years,
                age_days=age_days,
                time_status=time_status,
                note="",
            )
        )
    return out


def make_undated_events(values: List[str], event_type: str, source_field: int, token_prefix: str, code_mode: str) -> List[CanonicalEvent]:
    out: List[CanonicalEvent] = []
    for code_raw in values:
        code_raw = (code_raw or "").strip()
        if not code_raw:
            continue
        code_norm = normalize_icd3(code_raw) if code_mode == "icd3" else code_raw.upper()
        out.append(
            CanonicalEvent(
                event_type=event_type,
                source_field=source_field,
                date_field=None,
                code_raw=code_raw,
                code_norm=code_norm,
                token_key=f"{token_prefix}:{code_norm}" if code_norm else None,
                event_date_raw=None,
                age_years=None,
                age_days=None,
                time_status="unresolved",
                note="no_time_anchor",
            )
        )
    return out


def scalar_list_from_row(row: Optional[dict]) -> List[str]:
    if row is None:
        return []
    out: List[str] = []
    for key, value in row.items():
        if key == "eid":
            continue
        value = (value or "").strip()
        if value:
            out.append(value)
    return out


def collect_token_sets(events: Iterable[CanonicalEvent], token_sets: Dict[str, set]) -> None:
    for event in events:
        if event.token_key is None or event.age_days is None:
            continue
        if event.event_type == "diagnosis":
            token_sets["exp0"].add(event.token_key)
            token_sets["exp1"].add(event.token_key)
            token_sets["exp2"].add(event.token_key)
        elif event.event_type == "procedure":
            token_sets["exp1"].add(event.token_key)
            token_sets["exp2"].add(event.token_key)
        elif event.event_type in {"cancer", "death"}:
            token_sets["exp2"].add(event.token_key)


def ensure_dirs(paths: MultitypePaths) -> None:
    for path in [
        paths.multitype_root,
        paths.canonical_dir,
        paths.vocab_dir,
        paths.model_input_dir,
        paths.experiment_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_token_vocabulary(token_sets: Dict[str, set]) -> Dict[str, int]:
    ordered = ["Padding", "No event"]
    all_keys = sorted(token_sets["exp2"])
    ordered.extend(all_keys)
    return {token_key: idx for idx, token_key in enumerate(ordered)}


def event_embedding_strategy(event: CanonicalEvent) -> str:
    if event.event_type in {"diagnosis", "cancer", "death"}:
        return "icd10_64d_pca"
    if event.event_type == "procedure":
        return "learned_vocab"
    return "unknown"


def write_vocab_files(paths: MultitypePaths, vocab: Dict[str, int], event_catalog: Dict[str, CanonicalEvent], static_schema: dict) -> None:
    with paths.dynamic_token_vocab_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token_id", "token_key", "event_type", "code_norm", "embedding_strategy"])
        writer.writerow([0, "Padding", "padding", "", "none"])
        writer.writerow([1, "No event", "no_event", "", "none"])
        for token_key, token_id in sorted(vocab.items(), key=lambda item: item[1]):
            if token_key in {"Padding", "No event"}:
                continue
            event = event_catalog[token_key]
            writer.writerow([token_id, token_key, event.event_type, event.code_norm, event_embedding_strategy(event)])

    with paths.dynamic_token_types_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_type", "token_prefix", "embedding_strategy"])
        writer.writerow(["diagnosis", "diag", "icd10_64d_pca"])
        writer.writerow(["procedure", "proc", "learned_vocab"])
        writer.writerow(["cancer", "cancer", "icd10_64d_pca"])
        writer.writerow(["death", "death", "icd10_64d_pca"])

    paths.static_schema_json.write_text(json.dumps(static_schema, ensure_ascii=False, indent=2), encoding="utf-8")


def build_static_row(eid: int, static_payload: dict, sex_categories: Dict[str, int], ethnicity_categories: Dict[str, int]) -> dict:
    sex_raw = static_payload["sex"]["value_raw"] or ""
    ethnicity_raw = static_payload["ethnicity"]["value_raw"] or ""
    return {
        "eid": eid,
        "sex_raw": sex_raw,
        "sex_id": "" if not sex_raw else sex_categories[sex_raw],
        "ethnicity_raw": ethnicity_raw,
        "ethnicity_id": "" if not ethnicity_raw else ethnicity_categories[ethnicity_raw],
        "height_cm": "" if static_payload["height_cm"]["missing"] else static_payload["height_cm"]["value"],
        "weight_kg": "" if static_payload["weight_kg"]["missing"] else static_payload["weight_kg"]["value"],
        "bmi": "" if static_payload["bmi"]["missing"] else static_payload["bmi"]["value"],
        "age_recruit": "" if static_payload["age_recruit"]["missing"] else static_payload["age_recruit"]["value"],
        "height_missing": int(static_payload["height_cm"]["missing"]),
        "weight_missing": int(static_payload["weight_kg"]["missing"]),
        "bmi_missing": int(static_payload["bmi"]["missing"]),
        "age_recruit_missing": int(static_payload["age_recruit"]["missing"]),
    }


def events_to_token_record(
    eid: int,
    events: List[CanonicalEvent],
    allowed_types: set[str],
    vocab: Dict[str, int],
) -> dict:
    rows = [
        (event.age_days, event.token_key, event)
        for event in events
        if event.event_type in allowed_types and event.age_days is not None and event.token_key in vocab
    ]
    rows.sort(key=lambda item: (int(item[0]), item[1]))
    return {
        "eid": eid,
        "tokens": [int(vocab[token_key]) for _, token_key, _ in rows],
        "ages": [int(age_days) for age_days, _, _ in rows],
        "token_keys": [token_key for _, token_key, _ in rows],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    project_layout = resolve_project_layout(args.dataset_root)
    ukb_extract_dir = args.ukb_extract_dir or project_layout.ukb_extract_root
    tokens_jsonl = resolve_tokens_jsonl(project_layout, ukb_extract_dir, args.tokens_jsonl)
    paths = resolve_multitype_paths(project_layout, data_version=args.data_version, experiment=args.experiment)

    if args.create_dirs:
        ensure_dirs(paths)
    else:
        ensure_dirs(paths)

    labels = load_labels(args.labels_path)

    lookups = {
        "diag_code": SortedCsvLookup(ukb_extract_dir / "41202-0.csv"),
        "diag_date": SortedCsvLookup(ukb_extract_dir / "41262-0.csv"),
        "proc_code": SortedCsvLookup(ukb_extract_dir / "41200-0.csv"),
        "proc_date": SortedCsvLookup(ukb_extract_dir / "41260-0.csv"),
        "cancer_code": SortedCsvLookup(ukb_extract_dir / "40006-0.csv"),
        "cancer_date": SortedCsvLookup(ukb_extract_dir / "40005-0.csv"),
        "death_code": SortedCsvLookup(ukb_extract_dir / "40001-0.csv"),
        "death_date": SortedCsvLookup(ukb_extract_dir / "40000-0.csv"),
        "sex": SortedCsvLookup(ukb_extract_dir / "31-0.csv"),
        "ethnicity": SortedCsvLookup(ukb_extract_dir / "21000-0.csv"),
        "height": SortedCsvLookup(ukb_extract_dir / "50-0.csv"),
        "weight": SortedCsvLookup(ukb_extract_dir / "21002-0.csv"),
        "bmi": SortedCsvLookup(ukb_extract_dir / "21001-0.csv"),
        "age_recruit": SortedCsvLookup(ukb_extract_dir / "21022-0.csv"),
        "medication": SortedCsvLookup(ukb_extract_dir / "ukb_20003.csv"),
        "secdiag": SortedCsvLookup(ukb_extract_dir / "ukb_41270.csv"),
    }

    token_sets = {"exp0": set(), "exp1": set(), "exp2": set()}
    event_catalog: Dict[str, CanonicalEvent] = {}
    sex_values: set[str] = set()
    ethnicity_values: set[str] = set()
    summary = {
        "patients_seen": 0,
        "diagnosis_age_matches": 0,
        "resolved_procedure_events": 0,
        "resolved_cancer_events": 0,
        "resolved_death_events": 0,
    }

    with (
        paths.patient_records_core_jsonl.open("w", encoding="utf-8") as core_handle,
        paths.patient_records_extended_jsonl.open("w", encoding="utf-8") as ext_handle,
    ):
        for record in iter_tokens(tokens_jsonl):
            if args.max_patients and summary["patients_seen"] >= args.max_patients:
                break
            summary["patients_seen"] += 1

            eid = int(record["eid"])
            static_payload = build_static_payload(
                sex_row=lookups["sex"].get(eid),
                ethnicity_row=lookups["ethnicity"].get(eid),
                height_row=lookups["height"].get(eid),
                weight_row=lookups["weight"].get(eid),
                bmi_row=lookups["bmi"].get(eid),
                age_recruit_row=lookups["age_recruit"].get(eid),
            )

            if static_payload["sex"]["value_raw"]:
                sex_values.add(static_payload["sex"]["value_raw"])
            if static_payload["ethnicity"]["value_raw"]:
                ethnicity_values.add(static_payload["ethnicity"]["value_raw"])

            diag_pairs = sort_dated_events(
                wide_pairs(lookups["diag_code"].get(eid), lookups["diag_date"].get(eid), "41202", "41262")
            )
            token_seq = disease_token_sequence(record, labels)
            matched_diag = greedy_align_diagnoses(diag_pairs, token_seq)
            birth_ordinal = infer_birth_ordinal(matched_diag, diag_pairs)
            diagnosis_events, matched_count = make_diagnosis_events(diag_pairs, matched_diag)
            summary["diagnosis_age_matches"] += matched_count

            procedure_events = make_dated_events(
                pairs=wide_pairs(lookups["proc_code"].get(eid), lookups["proc_date"].get(eid), "41200", "41260"),
                event_type="procedure",
                source_field=41200,
                date_field=41260,
                token_prefix="proc",
                code_mode="raw",
                birth_ordinal=birth_ordinal,
            )
            cancer_events = make_dated_events(
                pairs=wide_pairs(lookups["cancer_code"].get(eid), lookups["cancer_date"].get(eid), "40006", "40005"),
                event_type="cancer",
                source_field=40006,
                date_field=40005,
                token_prefix="cancer",
                code_mode="icd3",
                birth_ordinal=birth_ordinal,
            )
            death_events = make_dated_events(
                pairs=wide_pairs(lookups["death_code"].get(eid), lookups["death_date"].get(eid), "40001", "40000"),
                event_type="death",
                source_field=40001,
                date_field=40000,
                token_prefix="death",
                code_mode="icd3",
                birth_ordinal=birth_ordinal,
            )

            summary["resolved_procedure_events"] += sum(1 for event in procedure_events if event.age_days is not None)
            summary["resolved_cancer_events"] += sum(1 for event in cancer_events if event.age_days is not None)
            summary["resolved_death_events"] += sum(1 for event in death_events if event.age_days is not None)

            core_events = diagnosis_events + procedure_events + cancer_events + death_events
            core_events.sort(key=lambda event: (event.age_days if event.age_days is not None else 10**12, event.token_key or ""))
            collect_token_sets(core_events, token_sets)
            for event in core_events:
                if event.token_key and event.token_key not in event_catalog:
                    event_catalog[event.token_key] = event

            extended_events = core_events + make_undated_events(
                scalar_list_from_row(lookups["medication"].get(eid)),
                event_type="medication",
                source_field=20003,
                token_prefix="med",
                code_mode="raw",
            ) + make_undated_events(
                scalar_list_from_row(lookups["secdiag"].get(eid)),
                event_type="secondary_diagnosis",
                source_field=41270,
                token_prefix="secdiag",
                code_mode="icd3",
            )

            core_record = {
                "eid": eid,
                "age_anchor": {
                    "mode": "diagnosis_subsequence_anchor",
                    "matched_diagnosis_events": matched_count,
                    "birth_ordinal_inferred": None if birth_ordinal is None else round(float(birth_ordinal), 2),
                },
                "static": static_payload,
                "dynamic": [event.to_dict() for event in core_events],
            }
            extended_record = {
                "eid": eid,
                "age_anchor": core_record["age_anchor"],
                "static": static_payload,
                "dynamic": [event.to_dict() for event in extended_events],
            }
            core_handle.write(json.dumps(core_record, ensure_ascii=False) + "\n")
            ext_handle.write(json.dumps(extended_record, ensure_ascii=False) + "\n")

    sex_categories = {value: idx for idx, value in enumerate(sorted(sex_values))}
    ethnicity_categories = {value: idx for idx, value in enumerate(sorted(ethnicity_values))}
    static_schema = {
        "sex": {"type": "categorical", "field_id": 31, "categories": sex_categories},
        "ethnicity": {"type": "categorical", "field_id": 21000, "categories": ethnicity_categories},
        "height_cm": {"type": "numeric", "field_id": 50},
        "weight_kg": {"type": "numeric", "field_id": 21002},
        "bmi": {"type": "numeric", "field_id": 21001},
        "age_recruit": {"type": "numeric", "field_id": 21022},
    }

    vocab = build_token_vocabulary(token_sets)
    write_vocab_files(paths, vocab, event_catalog, static_schema)

    with (
        paths.patient_records_core_jsonl.open("r", encoding="utf-8") as core_handle,
        paths.tokens_exp0_diag_jsonl.open("w", encoding="utf-8") as exp0_handle,
        paths.tokens_exp1_diag_proc_jsonl.open("w", encoding="utf-8") as exp1_handle,
        paths.tokens_exp2_diag_proc_cancer_death_jsonl.open("w", encoding="utf-8") as exp2_handle,
        paths.static_features_v1_csv.open("w", encoding="utf-8", newline="") as static_handle,
    ):
        static_writer = csv.DictWriter(
            static_handle,
            fieldnames=[
                "eid",
                "sex_raw",
                "sex_id",
                "ethnicity_raw",
                "ethnicity_id",
                "height_cm",
                "weight_kg",
                "bmi",
                "age_recruit",
                "height_missing",
                "weight_missing",
                "bmi_missing",
                "age_recruit_missing",
            ],
        )
        static_writer.writeheader()

        for line in core_handle:
            record = json.loads(line)
            eid = int(record["eid"])
            events = [CanonicalEvent(**event) for event in record["dynamic"]]

            exp0_record = events_to_token_record(eid, events, {"diagnosis"}, vocab)
            exp1_record = events_to_token_record(eid, events, {"diagnosis", "procedure"}, vocab)
            exp2_record = events_to_token_record(eid, events, {"diagnosis", "procedure", "cancer", "death"}, vocab)
            exp0_handle.write(json.dumps(exp0_record, ensure_ascii=False) + "\n")
            exp1_handle.write(json.dumps(exp1_record, ensure_ascii=False) + "\n")
            exp2_handle.write(json.dumps(exp2_record, ensure_ascii=False) + "\n")
            static_writer.writerow(build_static_row(eid, record["static"], sex_categories, ethnicity_categories))

    build_summary = {
        "environment": project_layout.environment,
        "dataset_root": str(project_layout.dataset_root),
        "ukb_extract_dir": str(ukb_extract_dir),
        "tokens_jsonl": str(tokens_jsonl),
        "data_version": paths.data_version,
        "experiment": paths.experiment,
        "patients_seen": summary["patients_seen"],
        "diagnosis_age_matches": summary["diagnosis_age_matches"],
        "resolved_procedure_events": summary["resolved_procedure_events"],
        "resolved_cancer_events": summary["resolved_cancer_events"],
        "resolved_death_events": summary["resolved_death_events"],
        "token_counts": {
            "exp0": len(token_sets["exp0"]),
            "exp1": len(token_sets["exp1"]),
            "exp2": len(token_sets["exp2"]),
        },
        "paths": {
            "patient_records_core_jsonl": str(paths.patient_records_core_jsonl),
            "patient_records_extended_jsonl": str(paths.patient_records_extended_jsonl),
            "tokens_exp0_diag_jsonl": str(paths.tokens_exp0_diag_jsonl),
            "tokens_exp1_diag_proc_jsonl": str(paths.tokens_exp1_diag_proc_jsonl),
            "tokens_exp2_diag_proc_cancer_death_jsonl": str(paths.tokens_exp2_diag_proc_cancer_death_jsonl),
            "static_features_v1_csv": str(paths.static_features_v1_csv),
            "dynamic_token_vocab_csv": str(paths.dynamic_token_vocab_csv),
            "dynamic_token_types_csv": str(paths.dynamic_token_types_csv),
            "static_schema_json": str(paths.static_schema_json),
        },
    }
    (paths.multitype_root / "build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for lookup in lookups.values():
        lookup.close()

    print(json.dumps(build_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
