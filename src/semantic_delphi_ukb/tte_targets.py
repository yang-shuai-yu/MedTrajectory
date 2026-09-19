from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import EXTERNAL_ROOT

import bisect
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Allows YAML/token mapping utilities to run in light CPU packaging environments.
    torch = None  # type: ignore[assignment]


MASK_TIME = -10000.0
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]


class DiseaseSpec:
    def __init__(
        self,
        disease_id: str,
        name: str,
        name_cn: str,
        category: str,
        ranges: list[str],
        level: str = "",
        parent_id: str = "",
    ):
        self.disease_id = disease_id
        self.name = name
        self.name_cn = name_cn
        self.category = category
        self.ranges = ranges
        self.level = level
        self.parent_id = parent_id


def parse_selected_diseases(path: Path) -> list[DiseaseSpec]:
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_items = payload.get("diseases", payload)
    except Exception:
        raw_items = parse_simple_yaml(path)
    return [
        DiseaseSpec(
            disease_id=str(item["id"]),
            name=str(item["name"]),
            name_cn=str(item.get("name_cn", item["name"])),
            category=str(item.get("category", "")),
            ranges=[str(x).strip().upper() for x in item["icd10"]],
            level=str(item.get("level", "")),
            parent_id=str(item.get("parent_id", item.get("parent", ""))),
        )
        for item in raw_items
    ]


def parse_simple_yaml(path: Path) -> list[dict]:
    items: list[dict] = []
    current: dict | None = None
    reading_codes = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "diseases:":
            continue
        if line.startswith("- id:"):
            if current is not None:
                items.append(current)
            current = {"id": clean_scalar(line.split(":", 1)[1])}
            reading_codes = False
            continue
        if current is None:
            continue
        if reading_codes and line.startswith("- "):
            current.setdefault("icd10", []).append(clean_scalar(line[2:]))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == "icd10":
                current["icd10"] = []
                reading_codes = True
            else:
                current[key.strip()] = clean_scalar(value)
                reading_codes = False
    if current is not None:
        items.append(current)
    return items


def clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def resolve_dataset_dir() -> Path:
    local = REPO_DIR / "data" / "ukb_semantic_multitype_explicit_split"
    if local.exists():
        return local
    remote_reference = Path(str(EXTERNAL_ROOT / "data" / "ukb_semantic_multitype_explicit_split"))
    return remote_reference


def load_token_codes(data_dir: Path | None = None) -> dict[int, str]:
    data_dir = data_dir or resolve_dataset_dir()
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    vocab_csv = Path(str(manifest["vocab_csv"]))
    if not vocab_csv.is_absolute():
        vocab_csv = data_dir / vocab_csv
    token_codes: dict[int, str] = {}
    with vocab_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type", "").strip() == "diagnosis" and row.get("code_norm", "").strip():
                token_codes[int(row["token_id"])] = row["code_norm"].strip().upper()
    return token_codes


def token_ids_for_disease(disease: DiseaseSpec, token_codes: dict[int, str]) -> list[int]:
    return sorted(token_id for token_id, code in token_codes.items() if any(code_matches_range(code, spec) for spec in disease.ranges))


def code_matches_range(code: str, spec: str) -> bool:
    code = code.strip().upper()
    spec = spec.strip().upper()
    if "-" not in spec:
        return code == spec or code.startswith(spec + ".")
    start, stop = [part.strip() for part in spec.split("-", 1)]
    parsed_code = parse_three_char_code(code)
    parsed_start = parse_three_char_code(start)
    parsed_stop = parse_three_char_code(stop)
    if parsed_code is None or parsed_start is None or parsed_stop is None:
        return False
    letter, number = parsed_code
    start_letter, start_number = parsed_start
    stop_letter, stop_number = parsed_stop
    return (start_letter, start_number) <= (letter, number) <= (stop_letter, stop_number)


def parse_three_char_code(code: str):
    match = re.match(r"^([A-Z])([0-9]{2})", code)
    return (match.group(1), int(match.group(2))) if match else None


def load_selected_disease_token_groups(path: Path, data_dir: Path | None = None) -> tuple[list[DiseaseSpec], list[list[int]]]:
    diseases = parse_selected_diseases(path)
    token_codes = load_token_codes(data_dir)
    token_groups = [token_ids_for_disease(disease, token_codes) for disease in diseases]
    return diseases, token_groups


def disease_specs_payload(diseases: Sequence[DiseaseSpec]) -> list[dict]:
    return [disease.__dict__ for disease in diseases]


def build_patient_disease_ages(
    data: np.ndarray,
    p2i: np.ndarray,
    token_groups: Sequence[Sequence[int]],
    vocab_size: int,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    token_to_diseases: list[list[int]] = [[] for _ in range(vocab_size)]
    for disease_idx, tokens in enumerate(token_groups):
        for token in tokens:
            if 0 <= int(token) < vocab_size:
                token_to_diseases[int(token)].append(disease_idx)

    patient_disease_ages: list[list[np.ndarray]] = []
    patient_last_ages = np.zeros(len(p2i), dtype=np.float32)
    for start, length in p2i:
        rows = data[int(start) : int(start) + int(length)]
        per_disease = [[] for _ in token_groups]
        if len(rows):
            patient_last_ages[len(patient_disease_ages)] = float(rows[-1, 1])
        for _, age_days, raw_token in rows:
            token_id = int(raw_token) + 1
            if 0 <= token_id < vocab_size:
                for disease_idx in token_to_diseases[token_id]:
                    per_disease[disease_idx].append(float(age_days))
        patient_disease_ages.append([np.asarray(values, dtype=np.float32) for values in per_disease])
    return patient_disease_ages, patient_last_ages


def build_tte_batch(
    ix: Sequence[int],
    ages: torch.Tensor,
    valid_tokens: torch.Tensor,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_last_ages: np.ndarray,
    horizon_years: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ix_list = [int(i) for i in ix.tolist()] if torch.is_tensor(ix) else [int(i) for i in ix]
    ages_np = ages.detach().cpu().numpy()
    valid_np = valid_tokens.detach().cpu().numpy().astype(bool)
    batch_size, seq_len = ages_np.shape
    num_diseases = len(patient_disease_ages[0])
    events = np.zeros((batch_size, seq_len, num_diseases), dtype=np.float32)
    durations = np.ones((batch_size, seq_len, num_diseases), dtype=np.float32)
    mask = np.zeros((batch_size, seq_len, num_diseases), dtype=np.float32)
    horizon_days = float(horizon_years) * 365.25

    for batch_idx, patient_idx in enumerate(ix_list):
        last_age = float(patient_last_ages[patient_idx])
        for pos in range(seq_len):
            current_age = float(ages_np[batch_idx, pos])
            if not valid_np[batch_idx, pos] or current_age <= MASK_TIME / 2 or current_age >= last_age:
                continue
            censor_days = max(1.0, min(last_age - current_age, horizon_days))
            for disease_idx, disease_ages in enumerate(patient_disease_ages[patient_idx]):
                next_idx = bisect.bisect_right(disease_ages, current_age)
                duration = censor_days
                event = 0.0
                if next_idx < len(disease_ages):
                    delta = float(disease_ages[next_idx]) - current_age
                    if 0.0 < delta <= horizon_days:
                        duration = max(1.0, delta)
                        event = 1.0
                events[batch_idx, pos, disease_idx] = event
                durations[batch_idx, pos, disease_idx] = duration / 365.25
                mask[batch_idx, pos, disease_idx] = 1.0

    return (
        torch.tensor(events, dtype=torch.float32, device=device),
        torch.tensor(durations, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def build_horizon_risk_batch(
    ix: Sequence[int],
    ages: torch.Tensor,
    valid_tokens: torch.Tensor,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_last_ages: np.ndarray,
    horizons_years: Sequence[float],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix_list = [int(i) for i in ix.tolist()] if torch.is_tensor(ix) else [int(i) for i in ix]
    ages_np = ages.detach().cpu().numpy()
    valid_np = valid_tokens.detach().cpu().numpy().astype(bool)
    batch_size, seq_len = ages_np.shape
    num_diseases = len(patient_disease_ages[0])
    events = np.zeros((batch_size, seq_len, len(horizons_years), num_diseases), dtype=np.float32)
    mask = np.zeros_like(events)

    for batch_idx, patient_idx in enumerate(ix_list):
        last_age = float(patient_last_ages[patient_idx])
        for pos in range(seq_len):
            current_age = float(ages_np[batch_idx, pos])
            if not valid_np[batch_idx, pos] or current_age <= MASK_TIME / 2 or current_age >= last_age:
                continue
            for horizon_idx, horizon_years in enumerate(horizons_years):
                horizon_days = float(horizon_years) * 365.25
                if last_age - current_age < 1.0:
                    continue
                for disease_idx, disease_ages in enumerate(patient_disease_ages[patient_idx]):
                    next_idx = bisect.bisect_right(disease_ages, current_age)
                    event = 0.0
                    if next_idx < len(disease_ages):
                        delta = float(disease_ages[next_idx]) - current_age
                        if 0.0 < delta <= horizon_days:
                            event = 1.0
                    events[batch_idx, pos, horizon_idx, disease_idx] = event
                    mask[batch_idx, pos, horizon_idx, disease_idx] = 1.0

    return (
        torch.tensor(events, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def build_survival_horizon_batch(
    ix: Sequence[int],
    ages: torch.Tensor,
    valid_tokens: torch.Tensor,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_last_ages: np.ndarray,
    num_bins: int,
    bin_years: float,
    horizons_years: Sequence[float],
    device: str,
    censor_aware: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ix_list = [int(i) for i in ix.tolist()] if torch.is_tensor(ix) else [int(i) for i in ix]
    ages_np = ages.detach().cpu().numpy()
    valid_np = valid_tokens.detach().cpu().numpy().astype(bool)
    batch_size, seq_len = ages_np.shape
    num_diseases = len(patient_disease_ages[0])
    survival_event = np.zeros((batch_size, seq_len, num_diseases, num_bins), dtype=np.float32)
    survival_mask = np.zeros_like(survival_event)
    horizon_event = np.zeros((batch_size, seq_len, len(horizons_years), num_diseases), dtype=np.float32)
    horizon_mask = np.zeros_like(horizon_event)
    bin_days = float(bin_years) * 365.25
    max_days = float(num_bins) * bin_days

    for batch_idx, patient_idx in enumerate(ix_list):
        last_age = float(patient_last_ages[patient_idx])
        for pos in range(seq_len):
            current_age = float(ages_np[batch_idx, pos])
            if not valid_np[batch_idx, pos] or current_age <= MASK_TIME / 2 or current_age >= last_age:
                continue
            follow_days = max(1.0, last_age - current_age)
            censor_days = min(follow_days, max_days)
            censor_bins = max(1, min(num_bins, int(np.ceil(censor_days / bin_days))))
            for disease_idx, disease_ages in enumerate(patient_disease_ages[patient_idx]):
                next_idx = bisect.bisect_right(disease_ages, current_age)
                event_delta = None
                if next_idx < len(disease_ages):
                    delta = float(disease_ages[next_idx]) - current_age
                    if delta > 0.0:
                        event_delta = delta

                survival_mask[batch_idx, pos, disease_idx, :censor_bins] = 1.0
                if event_delta is not None and event_delta <= max_days:
                    event_bin = max(0, min(num_bins - 1, int(np.floor((event_delta - 1e-6) / bin_days))))
                    survival_mask[batch_idx, pos, disease_idx, : event_bin + 1] = 1.0
                    survival_event[batch_idx, pos, disease_idx, event_bin] = 1.0

                for horizon_idx, horizon_years in enumerate(horizons_years):
                    horizon_days = float(horizon_years) * 365.25
                    if follow_days < 1.0:
                        continue
                    event_in_horizon = event_delta is not None and event_delta <= horizon_days
                    if not censor_aware or event_in_horizon or follow_days >= horizon_days:
                        horizon_mask[batch_idx, pos, horizon_idx, disease_idx] = 1.0
                    if event_in_horizon:
                        horizon_event[batch_idx, pos, horizon_idx, disease_idx] = 1.0

    return (
        torch.tensor(survival_event, dtype=torch.float32, device=device),
        torch.tensor(survival_mask, dtype=torch.float32, device=device),
        torch.tensor(horizon_event, dtype=torch.float32, device=device),
        torch.tensor(horizon_mask, dtype=torch.float32, device=device),
    )
