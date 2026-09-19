from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from semantic_delphi_ukb.calibration_auc import (
    age_stratified_delong_rows,
    aggregate_age_brackets_delong,
    extract_official_lm_records,
    get_auc_delong_var,
)
from semantic_delphi_ukb.paper_protocol import PAPER_MEDICAL_AUC_AGE_GROUPS


def official_medical_auc_rows(
    targets: np.ndarray,
    input_ages: np.ndarray,
    target_ages: np.ndarray,
    token_scores: np.ndarray,
    token_id: int,
    patient_ids: np.ndarray,
    sex_values: np.ndarray,
    sex_mapping: Mapping[str, int],
    age_groups: Sequence[float] = PAPER_MEDICAL_AUC_AGE_GROUPS,
    offset_days: float = 0.1,
    seed: int = 1337,
    metadata: Mapping[str, object] | None = None,
) -> list[dict]:
    output: list[dict] = []
    for sex_name, sex_id in sex_mapping.items():
        selected = np.asarray(sex_values) == int(sex_id)
        records = extract_official_lm_records(
            targets=np.asarray(targets)[selected],
            input_ages=np.asarray(input_ages)[selected],
            target_ages=np.asarray(target_ages)[selected],
            token_scores=np.asarray(token_scores)[selected],
            token_id=token_id,
            patient_ids=np.asarray(patient_ids)[selected],
            offset_days=offset_days,
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
                rng=np.random.default_rng(np.random.SeedSequence([seed, token_id, int(sex_id)])),
                metadata={
                    "metric": "delphi2m_medical_auc",
                    "sex": sex_name,
                    "token_id": int(token_id),
                    **dict(metadata or {}),
                },
            )
        )
    return output


def aggregate_official_medical_auc(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    return aggregate_age_brackets_delong(rows, ["metric", "token_id"])


def longitudinal_delong_rows(
    disease_scores: np.ndarray,
    disease_outcomes: np.ndarray,
    disease_ids: Sequence[object],
    minimum_cases: int = 25,
) -> list[dict]:
    scores = np.asarray(disease_scores, dtype=np.float64)
    outcomes = np.asarray(disease_outcomes, dtype=np.int8)
    if scores.shape != outcomes.shape or scores.ndim != 2:
        raise ValueError("scores and outcomes must share [patient, disease] shape")
    if scores.shape[1] != len(disease_ids):
        raise ValueError("disease_ids do not match the disease dimension")

    rows: list[dict] = []
    for disease_idx, disease_id in enumerate(disease_ids):
        labels = outcomes[:, disease_idx]
        cases = scores[labels == 1, disease_idx]
        controls = scores[labels == 0, disease_idx]
        if len(cases) < int(minimum_cases) or len(controls) == 0:
            continue
        auc, variance = get_auc_delong_var(controls, cases)
        rows.append(
            {
                "metric": "delphi2m_longitudinal_auc",
                "disease_id": disease_id,
                "auc": auc,
                "auc_delong": auc,
                "auc_variance_delong": variance,
                "n_cases": int(len(cases)),
                "n_controls": int(len(controls)),
            }
        )
    return rows

