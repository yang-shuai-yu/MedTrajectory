from __future__ import annotations

import bisect
from typing import Sequence

import numpy as np


DAYS_PER_YEAR = 365.25


def censor_aware_horizon_target(
    current_age_days: float,
    disease_ages_days: Sequence[float] | np.ndarray,
    followup_end_age_days: float,
    horizon_years: float,
) -> tuple[float, float, float]:
    disease_ages = np.asarray(disease_ages_days, dtype=np.float64)
    horizon_days = float(horizon_years) * DAYS_PER_YEAR
    next_idx = bisect.bisect_right(disease_ages, float(current_age_days))
    if next_idx < len(disease_ages):
        delta = float(disease_ages[next_idx]) - float(current_age_days)
        if 0.0 < delta <= horizon_days:
            return 1.0, 1.0, max(1.0, delta) / DAYS_PER_YEAR

    followup_days = float(followup_end_age_days) - float(current_age_days)
    if followup_days >= horizon_days:
        return 0.0, 1.0, float(horizon_years)
    return 0.0, 0.0, max(1.0, followup_days) / DAYS_PER_YEAR


def build_censor_aware_horizon_arrays(
    current_ages_days: Sequence[float] | np.ndarray,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_followup_end_ages: Sequence[float] | np.ndarray,
    horizons_years: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_ages = np.asarray(current_ages_days, dtype=np.float64)
    followup_ends = np.asarray(patient_followup_end_ages, dtype=np.float64)
    if len(current_ages) != len(patient_disease_ages) or len(current_ages) != len(followup_ends):
        raise ValueError("patient arrays are not aligned")
    if not patient_disease_ages:
        raise ValueError("patient_disease_ages must not be empty")

    disease_count = len(patient_disease_ages[0])
    shape = (len(current_ages), len(horizons_years), disease_count)
    labels = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    durations = np.zeros(shape, dtype=np.float32)
    for patient_idx, current_age in enumerate(current_ages):
        if len(patient_disease_ages[patient_idx]) != disease_count:
            raise ValueError("all patients must have the same disease count")
        for horizon_idx, horizon in enumerate(horizons_years):
            for disease_idx, disease_ages in enumerate(patient_disease_ages[patient_idx]):
                label, eligible, duration = censor_aware_horizon_target(
                    current_age,
                    disease_ages,
                    followup_ends[patient_idx],
                    horizon,
                )
                labels[patient_idx, horizon_idx, disease_idx] = label
                mask[patient_idx, horizon_idx, disease_idx] = eligible
                durations[patient_idx, horizon_idx, disease_idx] = duration
    return labels, mask, durations

