from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.calibration_auc import (  # noqa: E402
    age_stratified_delong_rows,
    aggregate_age_brackets_delong,
    build_horizon_case_control,
    extract_official_lm_records,
    precompute_official_prediction_indices,
    select_age_landmarks,
    validate_age_groups,
)
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    parse_selected_diseases,
    token_ids_for_disease,
)
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    load_followup_end_ages,
    parse_horizons,
)
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official Delphi-2M age-stratified AUC plus censoring-aware horizon-risk AUC.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--filter-min-total", type=int, default=100)
    parser.add_argument("--disease-chunk-size", type=int, default=200)
    parser.add_argument("--age-groups", default="40,45,50,55,60,65,70,75")
    parser.add_argument("--offset", type=float, default=0.1, help="Prediction-to-target gap in days; 0.1 matches the official pipeline.")
    parser.add_argument("--horizons", default=None, help="Comma-separated years; defaults to checkpoint metadata.")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def resolve_diseases_yaml(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    root_path = REPO_DIR / "selected_diseases.yaml"
    if root_path.exists():
        return root_path
    return REPO_DIR / "docs" / "selected_diseases.yaml"


def resolve_vocab_csv(data_dir: Path) -> Path:
    manifest_path = data_dir / "prepare_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        configured = Path(str(manifest.get("vocab_csv", "")))
        if configured.exists():
            return configured
    for candidate in (data_dir / "vocab" / "dynamic_token_vocab.csv", data_dir / "dynamic_token_vocab.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve dynamic_token_vocab.csv under {data_dir}")


def load_vocab_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_disease_token_groups(diseases_yaml: Path, vocab_rows: Sequence[dict]):
    diseases = parse_selected_diseases(diseases_yaml)
    token_codes = {
        int(row["token_id"]): (row.get("code_norm") or "").strip().upper()
        for row in vocab_rows
        if (row.get("event_type") or "").strip() == "diagnosis" and (row.get("code_norm") or "").strip()
    }
    return diseases, [token_ids_for_disease(disease, token_codes) for disease in diseases]


def load_sex_mapping(vocab_csv: Path) -> dict[str, int]:
    schema_path = vocab_csv.parent / "static_schema.json"
    if not schema_path.exists():
        manifest_path = vocab_csv.parent.parent / "prepare_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if manifest.get("static_feature_order") == ["sex_id"]:
                return {"female": 0, "male": 1}
        raise FileNotFoundError(f"Missing static sex schema: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    categories = schema["sex"]["categories"]
    mapping: dict[str, int] = {}
    if "0" in categories:
        mapping["female"] = int(categories["0"])
    if "1" in categories:
        mapping["male"] = int(categories["1"])
    if not mapping:
        raise RuntimeError(f"No UKB sex categories found in {schema_path}")
    return mapping


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def load_model(checkpoint: Mapping[str, object], device: str) -> torch.nn.Module:
    model_args = dict(checkpoint["model_args"])
    is_car_rope = checkpoint.get("model_family") == "CARoPE_v1" or "rope_scales" in model_args
    if is_car_rope:
        model = CARoPEHorizonMedTrajectory(CARoPEConfig(**model_args))
    else:
        model = HorizonRiskMedTrajectory(HorizonRiskConfig(**model_args))
    state_dict = dict(checkpoint["model"])
    for key in list(state_dict):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod.") :]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def resolve_horizons(args: argparse.Namespace, checkpoint: Mapping[str, object], model: torch.nn.Module) -> list[float]:
    if args.horizons:
        horizons = parse_horizons(args.horizons)
    else:
        checkpoint_config = checkpoint.get("config", {})
        configured = checkpoint_config.get("horizons") if isinstance(checkpoint_config, Mapping) else None
        horizons = [float(value) for value in configured] if configured else [5.0, 10.0]
    if len(horizons) != int(model.config.num_horizons):
        raise ValueError(f"Resolved {len(horizons)} horizons but checkpoint head has {model.config.num_horizons}")
    return horizons


def build_official_left_batch(data, p2i, static, block_size: int, no_event_token_rate: int, fixed_padding_ages=None):
    patient_ids = np.arange(len(p2i), dtype=np.int64)
    batch = get_batch(
        patient_ids,
        data,
        p2i,
        static,
        select="left",
        padding="none" if fixed_padding_ages is not None else "random",
        block_size=block_size,
        device="cpu",
        no_event_token_rate=no_event_token_rate,
        fixed_padding_ages=fixed_padding_ages,
        cut_batch=False,
    )
    return patient_ids, batch


def split_chunks(values: Sequence[int], chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("disease-chunk-size must be positive")
    values_array = np.asarray(values, dtype=np.int64)
    return [values_array[start : start + chunk_size] for start in range(0, len(values_array), chunk_size)]


@torch.no_grad()
def infer_score_chunk(model, batch, token_chunk: np.ndarray, batch_size: int, device: str, collect_risk: bool):
    lm_parts: list[np.ndarray] = []
    risk_parts: list[np.ndarray] = []
    patient_count = batch[0].shape[0]
    for start in range(0, patient_count, batch_size):
        stop = min(start + batch_size, patient_count)
        x, age, y, target_age, static = [tensor[start:stop].to(device) for tensor in batch]
        logits, _, _, _, risk_logits = model(x, age, static)
        lm_parts.append(logits[:, :, token_chunk].detach().cpu().numpy().astype(np.float16))
        if collect_risk:
            risk_parts.append(risk_logits.detach().cpu().numpy().astype(np.float32))
    lm_scores = np.concatenate(lm_parts, axis=0)
    risk_scores = np.concatenate(risk_parts, axis=0) if collect_risk else None
    return lm_scores, risk_scores


def diagnosis_metadata(vocab_rows: Sequence[dict], targets: np.ndarray, minimum_count: int) -> list[dict]:
    valid_targets = targets[targets > 1]
    token_values, counts = np.unique(valid_targets.astype(np.int64), return_counts=True)
    count_by_token = dict(zip(token_values.tolist(), counts.tolist()))
    rows = []
    for row in vocab_rows:
        if (row.get("event_type") or "").strip() != "diagnosis":
            continue
        token_id = int(row["token_id"])
        count = int(count_by_token.get(token_id, 0))
        if count >= minimum_count:
            rows.append(
                {
                    "token_id": token_id,
                    "token_name": row.get("token_key", ""),
                    "code_norm": (row.get("code_norm") or "").strip().upper(),
                    "target_count": count,
                }
            )
    return rows


def append_lm_rows(
    output: list[dict],
    batch_arrays: tuple[np.ndarray, ...],
    token_scores: np.ndarray,
    token_meta: Mapping[str, object],
    patient_ids: np.ndarray,
    prediction_indices: np.ndarray,
    sex_values: np.ndarray,
    sex_mapping: Mapping[str, int],
    age_groups: np.ndarray,
    offset: float,
    seed: int,
    disease_ids: Sequence[str],
) -> None:
    _, input_ages, targets, target_ages, _ = batch_arrays
    for sex_name, sex_id in sex_mapping.items():
        sex_mask = sex_values == float(sex_id)
        records = extract_official_lm_records(
            targets=targets[sex_mask],
            input_ages=input_ages[sex_mask],
            target_ages=target_ages[sex_mask],
            token_scores=token_scores[sex_mask],
            token_id=int(token_meta["token_id"]),
            patient_ids=patient_ids[sex_mask],
            offset_days=offset,
            prediction_indices=prediction_indices[sex_mask],
        )
        if len(records["scores"]) == 0:
            continue
        output.extend(
            age_stratified_delong_rows(
                scores=records["scores"],
                labels=records["labels"],
                prediction_age_days=records["prediction_age_days"],
                patient_ids=records["patient_ids"],
                age_groups=age_groups,
                rng=np.random.default_rng(np.random.SeedSequence([seed, int(token_meta["token_id"]), int(sex_id)])),
                metadata={
                    "score_source": "official_lm_token",
                    "sex": sex_name,
                    **dict(token_meta),
                    "disease_ids": list(disease_ids),
                    "horizon_years": None,
                },
            )
        )


def build_risk_rows(
    risk_scores: np.ndarray,
    batch_arrays: tuple[np.ndarray, ...],
    patient_ids: np.ndarray,
    patient_disease_ages,
    patient_last_ages: np.ndarray,
    diseases,
    horizons: Sequence[float],
    sex_values: np.ndarray,
    sex_mapping: Mapping[str, int],
    age_groups: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    input_tokens, input_ages, _, _, _ = batch_arrays
    landmarks = select_age_landmarks(input_tokens, input_ages, patient_ids, age_groups, rng)
    landmark_scores = risk_scores[landmarks["row_indices"], landmarks["positions"]]
    output: list[dict] = []
    for horizon_idx, horizon in enumerate(horizons):
        for disease_idx, disease in enumerate(diseases):
            labels, eligible = build_horizon_case_control(
                landmarks["patient_ids"],
                landmarks["prediction_age_days"],
                patient_disease_ages,
                patient_last_ages,
                horizon,
                disease_idx,
            )
            for sex_name, sex_id in sex_mapping.items():
                sex_mask = sex_values[landmarks["patient_ids"].astype(np.int64)] == float(sex_id)
                selected = eligible & sex_mask
                output.extend(
                    age_stratified_delong_rows(
                        scores=landmark_scores[:, horizon_idx, disease_idx][selected],
                        labels=labels[selected],
                        prediction_age_days=landmarks["prediction_age_days"][selected],
                        patient_ids=landmarks["patient_ids"][selected],
                        age_groups=age_groups,
                        rng=rng,
                        metadata={
                            "score_source": "horizon_risk",
                            "sex": sex_name,
                            "token_id": None,
                            "token_name": None,
                            "code_norm": None,
                            "target_count": None,
                            "disease_ids": [disease.disease_id],
                            "disease_id": disease.disease_id,
                            "disease_name": disease.name,
                            "disease_name_cn": disease.name_cn,
                            "horizon_years": float(horizon),
                        },
                    )
                )
    return output


def lm_disease_macro_rows(token_rows: Sequence[Mapping[str, object]], diseases, token_groups) -> list[dict]:
    by_token = {int(row["token_id"]): row for row in token_rows}
    output = []
    for disease, token_ids in zip(diseases, token_groups):
        members = [by_token[int(token)] for token in token_ids if int(token) in by_token]
        if not members:
            continue
        aucs = np.asarray([float(row["auc"]) for row in members], dtype=np.float64)
        variances = np.asarray([float(row["auc_variance_delong"]) for row in members], dtype=np.float64)
        output.append(
            {
                "score_source": "lm_member_token_macro",
                "disease_id": disease.disease_id,
                "disease_name": disease.name,
                "disease_name_cn": disease.name_cn,
                "horizon_years": None,
                "auc": float(aucs.mean()),
                "auc_variance_delong": float(variances.sum() / len(members) ** 2) if np.isfinite(variances).all() else float("nan"),
                "n_tokens": int(len(members)),
                "n_cases": int(sum(int(row["n_cases"]) for row in members)),
                "n_controls": int(sum(int(row["n_controls"]) for row in members)),
                "note": "Macro mean of official token AUCs; not a disease-union AUC.",
            }
        )
    return output


def sanitize_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    age_groups = np.asarray([float(value) for value in args.age_groups.split(",") if value.strip()], dtype=np.float64)
    validate_age_groups(age_groups)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = load_model(checkpoint, args.device)
    horizons = resolve_horizons(args, checkpoint, model)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    vocab_csv = resolve_vocab_csv(data_dir)
    vocab_rows = load_vocab_rows(vocab_csv)
    diseases, token_groups = load_disease_token_groups(diseases_yaml, vocab_rows)
    if len(diseases) != int(model.config.num_tte_tasks):
        raise ValueError(f"Disease YAML has {len(diseases)} groups but checkpoint head has {model.config.num_tte_tasks}")

    checkpoint_config = checkpoint.get("config", {})
    checkpoint_diseases = checkpoint_config.get("diseases") if isinstance(checkpoint_config, Mapping) else None
    if checkpoint_diseases:
        checkpoint_ids = [str(item["disease_id"]) for item in checkpoint_diseases]
        current_ids = [disease.disease_id for disease in diseases]
        if checkpoint_ids != current_ids:
            raise ValueError("Disease YAML order does not match checkpoint disease metadata")

    data, p2i, static = load_split(data_dir, args.split, args.max_patients)
    no_event_rate = int(checkpoint_config.get("no_event_token_rate", 5)) if isinstance(checkpoint_config, Mapping) else 5
    patient_ids, batch = build_official_left_batch(data, p2i, static, int(model.config.block_size), no_event_rate)
    batch_arrays = tuple(tensor.detach().cpu().numpy() for tensor in batch)
    input_tokens, input_ages, targets, target_ages, static_features = batch_arrays

    token_metadata = diagnosis_metadata(vocab_rows, targets, args.filter_min_total)
    if not token_metadata:
        raise RuntimeError("No diagnosis tokens passed filter-min-total")
    sex_mapping = load_sex_mapping(vocab_csv)
    sex_values = static_features[:, 0]
    prediction_indices = precompute_official_prediction_indices(input_ages, target_ages, args.offset)
    token_to_diseases: dict[int, list[str]] = {}
    for disease, tokens in zip(diseases, token_groups):
        for token in tokens:
            token_to_diseases.setdefault(int(token), []).append(disease.disease_id)

    detailed_rows: list[dict] = []
    risk_scores = None
    metadata_by_token = {int(row["token_id"]): row for row in token_metadata}
    for chunk_idx, token_chunk in enumerate(split_chunks(list(metadata_by_token), args.disease_chunk_size)):
        lm_scores, chunk_risk_scores = infer_score_chunk(
            model,
            batch,
            token_chunk,
            args.batch_size,
            args.device,
            collect_risk=chunk_idx == 0,
        )
        if chunk_risk_scores is not None:
            risk_scores = chunk_risk_scores
        for local_idx, token_id in enumerate(token_chunk.tolist()):
            append_lm_rows(
                detailed_rows,
                batch_arrays,
                lm_scores[:, :, local_idx],
                metadata_by_token[token_id],
                patient_ids,
                prediction_indices,
                sex_values,
                sex_mapping,
                age_groups,
                args.offset,
                args.seed,
                token_to_diseases.get(token_id, []),
            )

    if risk_scores is None:
        raise RuntimeError("Risk logits were not collected")
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(
        data, p2i, token_groups, int(model.config.vocab_size)
    )
    patient_followup_end_ages = load_followup_end_ages(
        data_dir, args.split, patient_last_ages, args.max_patients
    )
    risk_rows = build_risk_rows(
        risk_scores,
        batch_arrays,
        patient_ids,
        patient_disease_ages,
        patient_followup_end_ages,
        diseases,
        horizons,
        sex_values,
        sex_mapping,
        age_groups,
        np.random.default_rng(np.random.SeedSequence([args.seed, 9173])),
    )
    detailed_rows.extend(risk_rows)

    lm_rows = [row for row in detailed_rows if row["score_source"] == "official_lm_token"]
    lm_token = aggregate_age_brackets_delong(
        lm_rows, ["score_source", "token_id", "token_name", "code_norm", "target_count"]
    )
    lm_token_by_sex = aggregate_age_brackets_delong(
        lm_rows, ["score_source", "sex", "token_id", "token_name", "code_norm", "target_count"]
    )
    risk_overall = aggregate_age_brackets_delong(
        risk_rows,
        ["score_source", "disease_id", "disease_name", "disease_name_cn", "horizon_years"],
    )
    risk_by_sex = aggregate_age_brackets_delong(
        risk_rows,
        ["score_source", "sex", "disease_id", "disease_name", "disease_name_cn", "horizon_years"],
    )
    lm_disease_macro = lm_disease_macro_rows(lm_token, diseases, token_groups)
    aggregates = {
        "lm_token_official_age_sex_macro": lm_token,
        "lm_token_by_sex": lm_token_by_sex,
        "lm_disease_member_token_macro": lm_disease_macro,
        "horizon_risk_age_sex_macro": risk_overall,
        "horizon_risk_by_sex": risk_by_sex,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "calibration_auc_rows.json"
    aggregates_path = args.out_dir / "calibration_auc_aggregates.json"
    summary_path = args.out_dir / "summary.json"
    write_json(rows_path, detailed_rows)
    write_json(aggregates_path, aggregates)
    summary = {
        "metric_note": "official_lm_token is the Delphi age-stratified ROC AUC, not a probability-calibration metric",
        "checkpoint": str(args.checkpoint),
        "data_dir": str(data_dir),
        "split": args.split,
        "patients": int(len(patient_ids)),
        "left_window_block_size": int(model.config.block_size),
        "offset_days": float(args.offset),
        "age_groups": age_groups.tolist(),
        "horizons_years": horizons,
        "diagnosis_tokens_evaluated": int(len(token_metadata)),
        "selected_diseases": int(len(diseases)),
        "detailed_rows": int(len(detailed_rows)),
        "lm_mean_auc": float(np.mean([row["auc"] for row in lm_token])) if lm_token else float("nan"),
        "horizon_risk_mean_auc": float(np.mean([row["auc"] for row in risk_overall])) if risk_overall else float("nan"),
        "outputs": {"rows": str(rows_path), "aggregates": str(aggregates_path)},
    }
    write_json(summary_path, summary)
    print(json.dumps(sanitize_json(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
