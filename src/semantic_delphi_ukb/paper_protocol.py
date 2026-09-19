from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import numpy as np


DAYS_PER_YEAR = 365.25
PAPER_SPLIT_SEED = 1337
PAPER_VALIDATION_FRACTION = 0.20
PAPER_MEDICAL_AUC_AGE_GROUPS = tuple(range(50, 80, 5))
TRAINING_INPUT_END = date(2020, 6, 30)
LONGITUDINAL_LANDMARK = date(2020, 7, 1)
LONGITUDINAL_OUTCOME_START = date(2021, 7, 1)
LONGITUDINAL_OUTCOME_END = date(2022, 7, 1)
LONGITUDINAL_MIN_CASES = 25


@dataclass(frozen=True)
class PaperSplit:
    train: frozenset[int]
    validation: frozenset[int]


def exact_random_split(
    participant_ids: Sequence[int] | np.ndarray,
    seed: int = PAPER_SPLIT_SEED,
    validation_fraction: float = PAPER_VALIDATION_FRACTION,
) -> PaperSplit:
    ids = np.asarray(participant_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("participant_ids must be a non-empty one-dimensional sequence")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("participant_ids must be unique")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    shuffled = ids[np.random.default_rng(seed).permutation(len(ids))]
    validation_count = int(math.ceil(len(ids) * validation_fraction))
    validation = frozenset(int(value) for value in shuffled[:validation_count])
    train = frozenset(int(value) for value in shuffled[validation_count:])
    return PaperSplit(train=train, validation=validation)


def parse_iso_date(value: str | None) -> date | None:
    value = (value or "").strip()
    return date.fromisoformat(value) if value else None


def is_training_input_event(event_date: date | None) -> bool:
    return event_date is not None and event_date <= TRAINING_INPUT_END


def is_longitudinal_outcome(event_date: date | None) -> bool:
    return (
        event_date is not None
        and LONGITUDINAL_OUTCOME_START <= event_date <= LONGITUDINAL_OUTCOME_END
    )


def is_alive_at_landmark(death_dates: Iterable[date]) -> bool:
    return all(death_date >= LONGITUDINAL_LANDMARK for death_date in death_dates)


def age_days_at(reference_date: date, birth_ordinal: float | None) -> float | None:
    if birth_ordinal is None:
        return None
    return float(reference_date.toordinal()) - float(birth_ordinal)

