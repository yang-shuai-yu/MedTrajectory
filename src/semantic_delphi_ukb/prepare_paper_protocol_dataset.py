from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_delphi_ukb.paper_protocol import (
    LONGITUDINAL_LANDMARK,
    LONGITUDINAL_MIN_CASES,
    LONGITUDINAL_OUTCOME_END,
    LONGITUDINAL_OUTCOME_START,
    PAPER_SPLIT_SEED,
    PAPER_VALIDATION_FRACTION,
    TRAINING_INPUT_END,
    age_days_at,
    exact_random_split,
    is_alive_at_landmark,
    is_longitudinal_outcome,
    is_training_input_event,
    parse_iso_date,
)


DEFAULT_SOURCE_ROOT = Path(r"the original Delphi research workspace")
DEFAULT_CURRENT_DATA = REPO_DIR / "data" / "ukb_semantic_multitype_explicit_split"
DEFAULT_OUTPUT = REPO_DIR / "data" / "paper_protocol_v1"
STATIC_FEATURE_ORDER = ("sex_id",)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    allowed_event_types: frozenset[str]
    reduced_cohort: bool = False


@dataclass
class SplitBuffer:
    payload: array = field(default_factory=lambda: array("I"))
    patient_rows: list[dict] = field(default_factory=list)
    static_rows: list[list[float]] = field(default_factory=list)
    followup_end_ages: list[float] = field(default_factory=list)
    longitudinal_outcomes: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ProfileState:
    spec: ProfileSpec
    output_dir: Path
    source_to_target_token: dict[int, int]
    token_key_to_target: dict[str, int]
    buffers: dict[str, SplitBuffer]
    counts: Counter = field(default_factory=Counter)
    longitudinal_case_counts: Counter = field(default_factory=Counter)


PROFILES = (
    ProfileSpec("diagnosis_death", frozenset({"diagnosis", "death"})),
    ProfileSpec("multitype", frozenset({"diagnosis", "procedure", "cancer", "death"})),
    ProfileSpec("reduced_multitype", frozenset({"diagnosis", "procedure", "cancer", "death"}), True),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build isolated Delphi-2M paper-protocol datasets without training.")
    parser.add_argument("--canonical-jsonl", type=Path, default=DEFAULT_SOURCE_ROOT / "canonical" / "patient_records_core.jsonl")
    parser.add_argument("--static-csv", type=Path, default=DEFAULT_SOURCE_ROOT / "model_input" / "static_features_v1.csv")
    parser.add_argument("--source-vocab-csv", type=Path, default=DEFAULT_SOURCE_ROOT / "vocab" / "dynamic_token_vocab.csv")
    parser.add_argument(
        "--source-semantic-npy",
        type=Path,
        default=DEFAULT_CURRENT_DATA / "semantic_input_embeddings_64d.npy",
    )
    parser.add_argument("--current-data-dir", type=Path, default=DEFAULT_CURRENT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=PAPER_SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=PAPER_VALIDATION_FRACTION)
    parser.add_argument(
        "--test-eids-csv",
        type=Path,
        default=None,
        help="Explicit locked-test EID list. When set, train/val are rebuilt from the remaining patients.",
    )
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--profiles", default=",".join(profile.name for profile in PROFILES))
    return parser


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_participant_ids(static_csv: Path, max_patients: int) -> list[int]:
    output: list[int] = []
    with static_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            output.append(int(row["eid"]))
            if max_patients and len(output) >= max_patients:
                break
    return output


def load_test_eids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "eid" in rows[0]:
        values = [int(row["eid"]) for row in rows]
    else:
        values = [int(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    output = set(values)
    if len(output) != len(values) or not output:
        raise ValueError("test EID list must be non-empty and unique")
    return output


def load_reduced_cohort(current_data_dir: Path) -> set[int]:
    cohort: set[int] = set()
    for split in ("train", "val", "test"):
        path = current_data_dir / f"{split}_patient_index.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            cohort.update(int(row["eid"]) for row in csv.DictReader(handle))
    return cohort


def load_vocab(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_profile_vocab(source_rows: Sequence[Mapping[str, str]], allowed_types: frozenset[str]) -> tuple[list[dict], dict[int, int]]:
    kept = [
        row
        for row in source_rows
        if int(row["token_id"]) in (0, 1) or (row.get("event_type") or "").strip() in allowed_types
    ]
    kept.sort(key=lambda row: int(row["token_id"]))
    output: list[dict] = []
    source_to_target: dict[int, int] = {}
    for target_id, row in enumerate(kept):
        source_id = int(row["token_id"])
        source_to_target[source_id] = target_id
        output.append({**dict(row), "source_token_id": source_id, "token_id": target_id})
    if [int(row["token_id"]) for row in output[:2]] != [0, 1]:
        raise RuntimeError("source vocabulary must begin with Padding and No event")
    return output, source_to_target


def write_profile_vocab(output_dir: Path, rows: Sequence[Mapping[str, object]], source_semantic: np.ndarray) -> None:
    vocab_dir = output_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["token_id", "token_key", "event_type", "code_norm", "embedding_strategy", "source_token_id"]
    with (vocab_dir / "dynamic_token_vocab.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)
    (output_dir / "labels.csv").write_text(
        "\n".join(str(row["token_key"]) for row in rows) + "\n",
        encoding="utf-8",
    )
    source_indices = np.asarray([int(row["source_token_id"]) for row in rows], dtype=np.int64)
    np.save(output_dir / "semantic_input_embeddings_64d.npy", source_semantic[source_indices].astype(np.float32))


def source_signature(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "first_mib_sha256": digest.hexdigest(),
    }


def event_date(event: Mapping[str, object]) -> date | None:
    return parse_iso_date(str(event.get("event_date_raw") or ""))


def sex_static(record: Mapping[str, object]) -> list[float]:
    static = record["static"]
    sex = static["sex"]
    value = sex.get("value_id")
    return [float(value) if value is not None else -1.0]


def patient_metadata(record: Mapping[str, object]) -> tuple[str, str]:
    static = record["static"]
    return str(static["sex"].get("value_raw") or ""), str(static["ethnicity"].get("value_raw") or "")


def age_recruit_anchor(record: Mapping[str, object]) -> float:
    payload = record["static"]["age_recruit"]
    value = payload.get("value")
    return max(1.0, float(value) * 365.25) if value is not None else 1.0


def write_uint32_bin(path: Path, payload: array) -> int:
    with path.open("wb") as handle:
        payload.tofile(handle)
    return len(payload) // 3


def append_patient(
    buffer: SplitBuffer,
    eid: int,
    rows: list[tuple[int, int]],
    static_vector: list[float],
    followup_end_age: float,
    model_eligible: bool,
    birth_anchor_available: bool,
    sex_raw: str,
    ethnicity_raw: str,
    anchor_only: bool,
    outcomes: set[int] | None = None,
) -> None:
    row_index = len(buffer.patient_rows)
    for age_days, token_id in rows:
        buffer.payload.extend((eid, max(0, int(age_days)), int(token_id) - 1))
    buffer.patient_rows.append(
        {
            "row_index": row_index,
            "eid": eid,
            "num_events": len(rows),
            "model_eligible": int(model_eligible),
            "birth_anchor_available": int(birth_anchor_available),
            "anchor_only": int(anchor_only),
            "sex_raw": sex_raw,
            "ethnicity_raw": ethnicity_raw,
        }
    )
    buffer.static_rows.append(static_vector)
    buffer.followup_end_ages.append(float(followup_end_age))
    if outcomes:
        buffer.longitudinal_outcomes.extend((row_index, token_id) for token_id in sorted(outcomes))


def write_split(output_dir: Path, split: str, buffer: SplitBuffer) -> dict:
    rows_written = write_uint32_bin(output_dir / f"{split}.bin", buffer.payload)
    static = np.asarray(buffer.static_rows, dtype=np.float32).reshape((-1, len(STATIC_FEATURE_ORDER)))
    np.save(output_dir / f"{split}_static.npy", static)
    np.save(output_dir / f"{split}_followup_end_age_days.npy", np.asarray(buffer.followup_end_ages, dtype=np.float32))
    index_path = output_dir / f"{split}_patient_index.csv"
    fields = [
        "row_index",
        "eid",
        "num_events",
        "model_eligible",
        "birth_anchor_available",
        "anchor_only",
        "sex_raw",
        "ethnicity_raw",
    ]
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(buffer.patient_rows)
    if split == "longitudinal":
        with (output_dir / "longitudinal_outcomes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "token_id"])
            writer.writerows(buffer.longitudinal_outcomes)
    sentinel_rows = len(buffer.patient_rows) if split == "longitudinal" else sum(
        int(row["anchor_only"]) for row in buffer.patient_rows
    )
    return {
        "patients": len(buffer.patient_rows),
        "rows": rows_written,
        "real_events": int(rows_written - sentinel_rows),
        "model_eligible": int(sum(int(row["model_eligible"]) for row in buffer.patient_rows)),
        "birth_anchor_available": int(sum(int(row["birth_anchor_available"]) for row in buffer.patient_rows)),
    }


def selected_profiles(value: str) -> list[ProfileSpec]:
    names = {item.strip() for item in value.split(",") if item.strip()}
    known = {profile.name: profile for profile in PROFILES}
    unknown = names - set(known)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    return [profile for profile in PROFILES if profile.name in names]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = selected_profiles(args.profiles)
    participant_ids = load_participant_ids(args.static_csv, args.max_patients)
    locked_test_eids = load_test_eids(args.test_eids_csv) if args.test_eids_csv else set()
    unknown_test_eids = locked_test_eids - set(participant_ids)
    if unknown_test_eids:
        raise ValueError(f"test EID list contains {len(unknown_test_eids)} patients absent from static.csv")
    train_val_participant_ids = [eid for eid in participant_ids if eid not in locked_test_eids]
    paper_split = exact_random_split(train_val_participant_ids, args.seed, args.validation_fraction)
    reduced_cohort = load_reduced_cohort(args.current_data_dir)
    reduced_participant_ids = [eid for eid in participant_ids if eid in reduced_cohort]
    if locked_test_eids and "reduced_multitype" in {profile.name for profile in profiles} and not (locked_test_eids & reduced_cohort):
        raise ValueError("locked test EIDs do not intersect reduced_multitype cohort; cannot create a P4 test split")
    reduced_train_val_ids = [eid for eid in reduced_participant_ids if eid not in locked_test_eids]
    reduced_paper_split = exact_random_split(reduced_train_val_ids, args.seed, args.validation_fraction)
    source_vocab = load_vocab(args.source_vocab_csv)
    source_semantic = np.load(args.source_semantic_npy).astype(np.float32)
    if source_semantic.shape[0] <= max(int(row["token_id"]) for row in source_vocab):
        raise ValueError("source semantic matrix is shorter than source vocabulary")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states: list[ProfileState] = []
    for profile in profiles:
        output_dir = args.output_dir / profile.name
        output_dir.mkdir(parents=True, exist_ok=True)
        vocab_rows, source_to_target = build_profile_vocab(source_vocab, profile.allowed_event_types)
        write_profile_vocab(output_dir, vocab_rows, source_semantic)
        states.append(
            ProfileState(
                spec=profile,
                output_dir=output_dir,
                source_to_target_token=source_to_target,
                token_key_to_target={str(row["token_key"]): int(row["token_id"]) for row in vocab_rows},
                buffers={name: SplitBuffer() for name in (("train", "val", "test", "longitudinal") if locked_test_eids else ("train", "val", "longitudinal"))},
            )
        )

    allowed_ids = set(participant_ids)
    processed = 0
    for record in iter_jsonl(args.canonical_jsonl):
        eid = int(record["eid"])
        if eid not in allowed_ids:
            continue
        processed += 1
        birth_anchor = record["age_anchor"].get("birth_ordinal_inferred")
        birth_anchor_available = birth_anchor is not None
        cutoff_age = age_days_at(TRAINING_INPUT_END, birth_anchor)
        fallback_age = age_recruit_anchor(record)
        followup_end_age = cutoff_age if cutoff_age is not None else fallback_age
        static_vector = sex_static(record)
        sex_raw, ethnicity_raw = patient_metadata(record)
        events = record.get("dynamic", [])
        death_dates = [event_date(event) for event in events if event.get("event_type") == "death"]
        alive = is_alive_at_landmark(value for value in death_dates if value is not None)
        for state in states:
            if state.spec.reduced_cohort and eid not in reduced_cohort:
                continue
            active_split = reduced_paper_split if state.spec.reduced_cohort else paper_split
            if eid in (locked_test_eids if not state.spec.reduced_cohort else locked_test_eids & reduced_cohort):
                split = "test"
            else:
                split = "val" if eid in active_split.validation else "train"
            input_rows: list[tuple[int, int]] = []
            outcomes: set[int] = set()
            for event in events:
                event_type = str(event.get("event_type") or "")
                if event_type not in state.spec.allowed_event_types:
                    continue
                target_token = state.token_key_to_target.get(str(event.get("token_key") or ""))
                if target_token is None:
                    continue
                current_date = event_date(event)
                age_days = event.get("age_days")
                if is_training_input_event(current_date) and age_days is not None:
                    input_rows.append((int(age_days), target_token))
                elif current_date is None:
                    state.counts["events_dropped_missing_date"] += 1
                if event_type == "diagnosis" and is_longitudinal_outcome(current_date):
                    outcomes.add(target_token)

            input_rows.sort(key=lambda item: (item[0], item[1]))
            anchor_only = not input_rows
            model_eligible = bool(birth_anchor_available and not anchor_only)
            if anchor_only:
                input_rows = [(int(round(followup_end_age)), 1)]
                state.counts["anchor_only_patients"] += 1
            append_patient(
                state.buffers[split],
                eid,
                input_rows,
                static_vector,
                followup_end_age,
                model_eligible,
                birth_anchor_available,
                sex_raw,
                ethnicity_raw,
                anchor_only,
            )
            state.counts["cohort_patients"] += 1
            state.counts[f"{split}_patients"] += 1
            if alive:
                longitudinal_rows = list(input_rows)
                if not anchor_only:
                    longitudinal_rows.append((int(round(followup_end_age)), 1))
                append_patient(
                    state.buffers["longitudinal"],
                    eid,
                    longitudinal_rows,
                    static_vector,
                    followup_end_age,
                    model_eligible,
                    birth_anchor_available,
                    sex_raw,
                    ethnicity_raw,
                    anchor_only,
                    outcomes,
                )
                state.counts["longitudinal_alive_patients"] += 1
                for token_id in outcomes:
                    state.longitudinal_case_counts[token_id] += 1
        if args.max_patients and processed >= args.max_patients:
            break

    root_manifest = {
        "protocol": "paper_protocol_v1",
        "training_started": False,
            "split": {
            "method": "exact_random_patient_train_val_plus_locked_test" if locked_test_eids else "exact_random_patient_80_20",
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "all_participants": len(participant_ids),
            "train_participants": len(paper_split.train),
            "validation_participants": len(paper_split.validation),
            "test_participants": len(locked_test_eids),
        },
        "calendar_windows": {
            "training_input_end_inclusive": TRAINING_INPUT_END.isoformat(),
            "longitudinal_alive_landmark": LONGITUDINAL_LANDMARK.isoformat(),
            "gap": f"{LONGITUDINAL_LANDMARK.isoformat()}..{date(2021, 6, 30).isoformat()}",
            "longitudinal_outcome_inclusive": f"{LONGITUDINAL_OUTCOME_START.isoformat()}..{LONGITUDINAL_OUTCOME_END.isoformat()}",
        },
        "static_model_features": list(STATIC_FEATURE_ORDER),
        "subgroup_only_fields": ["ethnicity_raw"],
        "source": {
            "canonical": source_signature(args.canonical_jsonl),
            "static": source_signature(args.static_csv),
            "vocab": source_signature(args.source_vocab_csv),
            "semantic": source_signature(args.source_semantic_npy),
        },
        "limitations": [
            "The available extract lacks UKB birth year/month and assessment-date fields.",
            "Participants without a recoverable diagnosis-derived birth anchor are retained but marked longitudinal model-ineligible.",
            "Anchor-only participants use a no-event indexing row; this is not presented as an observed clinical event.",
        ],
        "profiles": {},
    }
    if args.test_eids_csv:
        test_digest = hashlib.sha256(args.test_eids_csv.read_bytes()).hexdigest()
        root_manifest["locked_test_source"] = {
            "path": str(args.test_eids_csv.resolve()),
            "sha256": test_digest,
            "patients": len(locked_test_eids),
        }

    for state in states:
        split_summaries = {name: write_split(state.output_dir, name, buffer) for name, buffer in state.buffers.items()}
        eligible_tokens = sorted(
            int(token_id)
            for token_id, count in state.longitudinal_case_counts.items()
            if count >= LONGITUDINAL_MIN_CASES
        )
        manifest = {
            "protocol": "paper_protocol_v1",
            "profile": state.spec.name,
            "allowed_event_types": sorted(state.spec.allowed_event_types),
            "reduced_cohort": state.spec.reduced_cohort,
            "split": {
                "method": "exact_random_patient_train_val_plus_locked_test" if locked_test_eids else "exact_random_patient_80_20",
                "seed": args.seed,
                "train_participants": int(state.counts["train_patients"]),
                "validation_participants": int(state.counts["val_patients"]),
                "test_participants": int(state.counts["test_patients"]),
            },
            "vocab_size": len(state.source_to_target_token),
            "vocab_csv": "vocab/dynamic_token_vocab.csv",
            "semantic_output": "semantic_input_embeddings_64d.npy",
            "static_feature_order": list(STATIC_FEATURE_ORDER),
            "ethnicity_model_input": False,
            "split_summaries": split_summaries,
            "counts": dict(state.counts),
            "longitudinal_min_cases": LONGITUDINAL_MIN_CASES,
            "longitudinal_eligible_disease_tokens": eligible_tokens,
            "longitudinal_eligible_disease_count": len(eligible_tokens),
            "training_started": False,
        }
        (state.output_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        root_manifest["profiles"][state.spec.name] = manifest

    (args.output_dir / "prepare_manifest.json").write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")
    print(json.dumps(root_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
