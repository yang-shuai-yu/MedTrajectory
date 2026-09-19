from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class GeneratedEvent:
    token_id: int
    age_days: float
    event_type: str
    wait_was_clamped: bool = False


def clinical_event_type(label: str) -> str:
    prefix = label.split(":", 1)[0]
    return {
        "diag": "diagnosis",
        "proc": "procedure",
        "cancer": "cancer",
        "death": "death",
    }.get(prefix, "other")


def clinical_candidate_mask(labels: Sequence[str], device: torch.device | str) -> torch.Tensor:
    allowed = torch.tensor(
        [clinical_event_type(label) != "other" for label in labels],
        dtype=torch.bool,
        device=device,
    )
    return allowed


def filtered_rate_logits(logits: torch.Tensor, ignore_tokens: Iterable[int]) -> torch.Tensor:
    filtered = logits.clone()
    ignored = {1, *(int(value) for value in ignore_tokens)}
    for token_id in ignored:
        if 0 <= token_id < filtered.numel():
            filtered[token_id] = -torch.inf
    return filtered


def frozen_log_rate(logits: torch.Tensor, ignore_tokens: Iterable[int], t_min: float) -> torch.Tensor:
    filtered = filtered_rate_logits(logits, ignore_tokens)
    raw_lse = torch.logsumexp(filtered, dim=-1)
    return -torch.log(torch.exp(-raw_lse) + float(t_min))


def waiting_time_nll(
    logits: torch.Tensor,
    delta_days: float,
    ignore_tokens: Iterable[int],
    t_min: float,
) -> torch.Tensor:
    log_rate = frozen_log_rate(logits, ignore_tokens, t_min)
    dt_eff = max(float(delta_days), 1.0)
    return -log_rate + torch.exp(log_rate) * (dt_eff + float(t_min))


def _top_p_probabilities(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1]")
    scores = logits / float(temperature)
    scores = scores.masked_fill(~candidate_mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    if top_p >= 1.0:
        return probabilities
    sorted_prob, sorted_index = torch.sort(probabilities, descending=True)
    cumulative = sorted_prob.cumsum(dim=-1)
    keep = cumulative - sorted_prob < float(top_p)
    sorted_prob = sorted_prob * keep
    sorted_prob = sorted_prob / sorted_prob.sum().clamp_min(torch.finfo(sorted_prob.dtype).tiny)
    output = torch.zeros_like(probabilities)
    output.scatter_(0, sorted_index, sorted_prob)
    return output


def inverse_cdf_sample(probabilities: torch.Tensor, uniform: torch.Tensor) -> int:
    cumulative = probabilities.cumsum(dim=-1)
    index = torch.searchsorted(cumulative, uniform.clamp(max=1.0 - 1e-7)).clamp_max(
        probabilities.numel() - 1
    )
    return int(index.item())


def sample_event_and_wait(
    logits: torch.Tensor,
    *,
    candidate_mask: torch.Tensor,
    ignore_tokens: Iterable[int],
    t_min: float,
    temperature: float,
    top_p: float,
    death_token_mask: torch.Tensor | None,
    death_logit_bias: float,
    minimum_wait_days: float,
    event_uniform: torch.Tensor,
    wait_uniform: torch.Tensor,
    rate_candidate_mask: torch.Tensor | None = None,
) -> tuple[int, float]:
    token_id, wait_days, _ = sample_event_and_wait_with_diagnostics(
        logits,
        candidate_mask=candidate_mask,
        ignore_tokens=ignore_tokens,
        t_min=t_min,
        temperature=temperature,
        top_p=top_p,
        death_token_mask=death_token_mask,
        death_logit_bias=death_logit_bias,
        minimum_wait_days=minimum_wait_days,
        event_uniform=event_uniform,
        wait_uniform=wait_uniform,
        rate_candidate_mask=rate_candidate_mask,
    )
    return token_id, wait_days


def sample_event_and_wait_with_diagnostics(
    logits: torch.Tensor,
    *,
    candidate_mask: torch.Tensor,
    ignore_tokens: Iterable[int],
    t_min: float,
    temperature: float,
    top_p: float,
    death_token_mask: torch.Tensor | None,
    death_logit_bias: float,
    minimum_wait_days: float,
    event_uniform: torch.Tensor,
    wait_uniform: torch.Tensor,
    rate_candidate_mask: torch.Tensor | None = None,
) -> tuple[int, float, bool]:
    event_logits = logits.clone()
    if death_token_mask is not None and death_logit_bias:
        event_logits[death_token_mask] += float(death_logit_bias)
    probabilities = _top_p_probabilities(event_logits, candidate_mask, temperature, top_p)
    token_id = inverse_cdf_sample(probabilities, event_uniform)

    # The frozen Track R rate is deliberately computed from untempered logits.
    # A restricted candidate space may also remove terminal-event hazards from
    # the ordinary event clock in validation-only ablations.
    rate_logits = logits
    if rate_candidate_mask is not None:
        rate_logits = logits.masked_fill(~rate_candidate_mask, -torch.inf)
    log_rate = frozen_log_rate(rate_logits, ignore_tokens, t_min)
    wait = -torch.log(wait_uniform.clamp_min(1e-12)) / torch.exp(log_rate) - float(t_min)
    raw_wait_days = float(wait.item())
    wait_was_clamped = raw_wait_days < float(minimum_wait_days)
    wait_days = max(raw_wait_days, float(minimum_wait_days))
    return token_id, wait_days, wait_was_clamped


def trajectory_validity_metrics(events: Sequence[GeneratedEvent]) -> dict[str, float]:
    nonmonotonic = sum(
        events[index].age_days <= events[index - 1].age_days
        for index in range(1, len(events))
    )
    seen_death = False
    post_death = 0
    for event in events:
        if seen_death:
            post_death += 1
        if event.event_type == "death":
            seen_death = True
    clamped = sum(event.wait_was_clamped for event in events)
    return {
        "nonmonotonic_time_rate": nonmonotonic / max(len(events) - 1, 1),
        "post_death_event_rate": post_death / max(len(events), 1),
        "minimum_wait_clamp_rate": clamped / max(len(events), 1),
    }


def levenshtein(left: Sequence[int], right: Sequence[int]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + int(a != b)))
        previous = current
    return previous[-1]


def normalized_edit_distance(left: Sequence[int], right: Sequence[int]) -> float:
    return levenshtein(left, right) / max(len(left), len(right), 1)


def set_metrics(actual: set[int], predicted: set[int]) -> tuple[float, float, float]:
    if not actual and not predicted:
        return 1.0, 1.0, 1.0
    intersection = len(actual & predicted)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(actual) if actual else 1.0
    union = len(actual | predicted)
    return precision, recall, intersection / union if union else 1.0


def js_divergence(left: Sequence[str], right: Sequence[str]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    left_counts = torch.tensor([left.count(key) for key in keys], dtype=torch.float64)
    right_counts = torch.tensor([right.count(key) for key in keys], dtype=torch.float64)
    if left_counts.sum() == 0 or right_counts.sum() == 0:
        return 1.0
    p = left_counts / left_counts.sum()
    q = right_counts / right_counts.sum()
    mean = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    return float(
        0.5 * (p[p_mask] * torch.log2(p[p_mask] / mean[p_mask])).sum()
        + 0.5 * (q[q_mask] * torch.log2(q[q_mask] / mean[q_mask])).sum()
    )


def expected_calibration_error(probability: Sequence[float], outcome: Sequence[int], bins: int = 10) -> float:
    if not probability:
        return float("nan")
    prob = torch.tensor(probability, dtype=torch.float64)
    target = torch.tensor(outcome, dtype=torch.float64)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    total = float(len(probability))
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (prob >= boundaries[index]) & (prob <= boundaries[index + 1])
        else:
            mask = (prob >= boundaries[index]) & (prob < boundaries[index + 1])
        if bool(mask.any()):
            ece += float(mask.sum()) / total * abs(float(prob[mask].mean() - target[mask].mean()))
    return ece
