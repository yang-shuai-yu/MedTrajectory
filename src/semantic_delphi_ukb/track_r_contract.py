from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


MASK_TIME = -10000.0
REQUIRED_STATIC_ORDER = ("sex", "bmi", "smoking", "alcohol")


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def protocol_contract_sha256(protocol: Mapping) -> str:
    payload = copy.deepcopy(dict(protocol))
    payload.pop("protocol_manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def load_track_r_protocol(path: Path) -> dict:
    path = Path(path)
    overlay = json.loads(path.read_text(encoding="utf-8-sig"))
    if "extends" in overlay:
        base = load_track_r_protocol((path.parent / overlay["extends"]).resolve())
        protocol = _deep_merge(base, overlay)
    else:
        protocol = overlay
    if protocol.get("protocol_id") not in ("track_r_v2_1", "track_r_v2_2"):
        raise ValueError("expected protocol_id=track_r_v2_1 or track_r_v2_2")
    prefix = protocol["static_prefix"]
    if tuple(prefix["logical_order"]) != REQUIRED_STATIC_ORDER:
        raise ValueError(f"static prefix order must be {REQUIRED_STATIC_ORDER}")
    if int(prefix["fixed_length"]) != len(REQUIRED_STATIC_ORDER):
        raise ValueError("static prefix must contain exactly one token per registered field")
    anchor = prefix["age_anchor"]
    if not anchor.get("allow_anchor_after_first_dynamic_event"):
        raise ValueError("protocol must record that recruitment age can follow early dynamic events")
    if anchor.get("temporal_visibility") != (
        "a_dynamic_query_may_attend_static_prefix_only_when_query_age_days_gte_static_anchor_age_days"
    ):
        raise ValueError("static temporal visibility rule is not frozen")
    loss = protocol["loss_contract"]
    required_false = (
        "static_to_static_transition",
        "static_to_first_dynamic_transition",
        "dynamic_to_static_transition",
        "static_tokens_are_targets",
        "optional_additional_bos_or_static_end_token",
    )
    if any(loss.get(name) is not False for name in required_false):
        raise ValueError("Track R loss exclusions must be explicit and false")
    bos = protocol["dynamic_bos"]
    if not bos.get("required") or int(bos.get("fixed_length", 0)) != 1:
        raise ValueError("Track R requires exactly one dynamic BOS token")
    if bos.get("age_anchor") != "first_dynamic_clinical_event_age_in_retained_window":
        raise ValueError("Track R dynamic BOS age contract is not frozen")
    if loss.get("dynamic_bos_to_first_dynamic_ce") is not True:
        raise ValueError("dynamic BOS must predict the first retained clinical event")
    if loss.get("dynamic_bos_to_first_dynamic_time_loss") is not False:
        raise ValueError("dynamic BOS transition must not contribute time loss")
    expected_lengths = {
        "A0-TokenStatic": 133,
        "A0-noStatic": 129,
        "A1-TokenStatic": 133,
        "A1-noStatic": 129,
        "Med-BERT-Paper": 133,
        "Med-BERT-Matched-S": 133,
    }
    if protocol.get("sequence_lengths") != expected_lengths:
        raise ValueError(f"Track R sequence lengths must be {expected_lengths}")
    if protocol["redundancy_policy"]["main_track_numeric_features"]:
        raise ValueError("height/weight must not accompany BMI bins in the primary Track R input")
    if protocol["protocol_id"] == "track_r_v2_2":
        if protocol.get("data_protocol_id") != "track_r_v2_1":
            raise ValueError("Track R v2.2 must reuse the frozen v2.1 data contract")
        if protocol.get("locked_test_read") is not False or protocol.get("test_authorized") is not False:
            raise ValueError("Track R v2.2 must keep locked test access disabled")
        if protocol.get("status") not in ("implementation_ready", "ready_for_training"):
            raise ValueError("Track R v2.2 status must describe its manifest-freeze lifecycle")
        if protocol.get("seeds") != [42, 43, 44]:
            raise ValueError("Track R v2.2 seeds must remain [42, 43, 44]")
        bootstrap = protocol.get("bootstrap_resample")
        if bootstrap != {
            "unit": "patient",
            "replicates": 10000,
            "seed": 20260815,
            "percentile": [2.5, 97.5],
            "shared_index_across_seeds": True,
            "per_seed_delta_then_average_deltas": True,
        }:
            raise ValueError("Track R v2.2 bootstrap contract is not frozen")
        mechanism = protocol.get("mechanism_gate", {})
        if mechanism.get("aggregation") != "per_seed_scalar_not_cross_seed_average":
            raise ValueError("Track R v2.2 mechanism aggregation must remain per-seed")
        if int(mechanism.get("required_seed_passes", 0)) != len(protocol.get("seeds", [])):
            raise ValueError("Track R v2.2 mechanism gate must require every seed")
        integrity = protocol.get("result_integrity", {})
        integrity_fields = (
            "gate_outcome_is_final",
            "report_regardless_of_pass_or_fail",
            "rerun_only_for_implementation_bugs",
            "no_post_hoc_threshold_or_parameter_change",
            "rerun_requires_audit_log",
        )
        if any(integrity.get(field) is not True for field in integrity_fields):
            raise ValueError("Track R v2.2 result-integrity contract is incomplete")
        wavelength = protocol.get("wavelength_contract", {})
        if wavelength.get("require_manifest_log_scale_match") is not True:
            raise ValueError("Track R v2.2 must cross-check manifest scale bounds")
        expected_wavelength_hash = wavelength.get("expected_sha256")
        if expected_wavelength_hash is not None and not _is_sha256(expected_wavelength_hash):
            raise ValueError("wavelength_contract.expected_sha256 must be a SHA-256 digest")
        if protocol["status"] == "ready_for_training" and not _is_sha256(expected_wavelength_hash):
            raise ValueError("ready_for_training requires wavelength_contract.expected_sha256")
        primary = protocol.get("pretraining_fidelity", {}).get("primary_contrast", {})
        if primary != {
            "left": "A2",
            "right": "A0",
            "direction": "lower",
            "ci_gate": "upper_below_zero",
            "required_seed_directions": 3,
        }:
            raise ValueError("Track R v2.2 primary contrast is not frozen")
        if protocol.get("clinical_gate", {}).get("contrast") != {"left": "A2", "right": "A0"}:
            raise ValueError("Track R v2.2 clinical contrast is not frozen")
        descriptive = protocol.get("descriptive_contrasts", [])
        if descriptive != [{
            "name": "A2_minus_A2_noAge",
            "metric": "observed_positive_gap_exponential_nll",
            "left": "A2",
            "right": "A2-noAge",
            "formal_gate": False,
        }]:
            raise ValueError("Track R v2.2 descriptive ablation contract is not frozen")
        expected_hash = protocol.get("protocol_manifest_sha256")
        if not expected_hash or expected_hash != protocol_contract_sha256(protocol):
            raise ValueError("Track R v2.2 protocol manifest SHA-256 mismatch")
    return protocol


def bmi_category(value: object) -> str:
    if value is None or str(value).strip() == "":
        return "missing"
    bmi = float(value)
    if not np.isfinite(bmi):
        return "missing"
    if bmi <= 0:
        return "missing"
    if bmi < 18.5:
        return "lt18.5"
    if bmi < 25.0:
        return "18.5_to_lt25"
    if bmi < 30.0:
        return "25_to_lt30"
    return "gte30"


def static_token_keys(categories: Mapping[str, str], order: Sequence[str] = REQUIRED_STATIC_ORDER) -> list[str]:
    return [f"static:{name}:{categories.get(name, 'missing')}" for name in order]


def prepend_static_context(
    x: torch.Tensor,
    age: torch.Tensor,
    y: torch.Tensor,
    target_age: torch.Tensor,
    prefix_token_ids: torch.Tensor,
    anchor_age_days: torch.Tensor,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepend static context and a required BOS that predicts the first retained clinical event."""
    if prefix_token_ids.ndim != 2 or prefix_token_ids.size(0) != x.size(0):
        raise ValueError("prefix_token_ids must have shape [batch, static_prefix_length]")
    if anchor_age_days.shape != (x.size(0),):
        raise ValueError("anchor_age_days must have shape [batch]")
    prefix_len = prefix_token_ids.size(1)
    prefix_age = anchor_age_days[:, None].expand(-1, prefix_len).to(age.dtype)
    masked_targets = torch.full_like(prefix_token_ids, -1)
    masked_target_age = torch.full_like(prefix_age, MASK_TIME)
    clinical = x > 1
    first_position = torch.where(
        clinical,
        torch.arange(x.size(1), device=x.device).view(1, -1),
        torch.full_like(x, x.size(1)),
    ).min(dim=1).values
    has_clinical = first_position < x.size(1)
    safe_position = first_position.clamp_max(max(x.size(1) - 1, 0))
    first_token = x.gather(1, safe_position[:, None]).masked_fill(~has_clinical[:, None], -1)
    first_age = age.gather(1, safe_position[:, None]).masked_fill(~has_clinical[:, None], MASK_TIME)
    bos = torch.full((x.size(0), 1), int(bos_token_id), dtype=x.dtype, device=x.device)
    x_out = torch.cat((prefix_token_ids, bos, x), dim=1)
    age_out = torch.cat((prefix_age, first_age, age), dim=1)
    y_out = torch.cat((masked_targets, first_token, y), dim=1)
    target_age_out = torch.cat((masked_target_age, first_age, target_age), dim=1)
    static_mask = torch.zeros_like(x_out, dtype=torch.bool)
    static_mask[:, :prefix_len] = True
    bos_mask = torch.zeros_like(x_out, dtype=torch.bool)
    bos_mask[:, prefix_len] = True
    next_event_mask = exact_next_event_mask(
        x_out, age_out, y_out, target_age_out, static_mask, bos_mask
    )
    # The first retained clinical event is supervised exactly once through BOS.
    has_predecessor = has_clinical & (first_position > 0)
    rows = torch.arange(x.size(0), device=x.device)[has_predecessor]
    predecessor = prefix_len + first_position[has_predecessor]
    next_event_mask[rows, predecessor] = False
    return x_out, age_out, y_out, target_age_out, static_mask, bos_mask, next_event_mask


def exact_next_event_mask(
    x: torch.Tensor,
    age: torch.Tensor,
    targets: torch.Tensor,
    targets_age: torch.Tensor,
    static_token_mask: torch.Tensor,
    bos_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Frozen Track R CE mask; time losses add the strict positive-time condition."""
    if not (x.shape == age.shape == targets.shape == targets_age.shape == static_token_mask.shape == bos_token_mask.shape):
        raise ValueError("Track R loss tensors must have identical [batch, sequence] shapes")
    source_is_dynamic = ((x > 0) & ~static_token_mask & ~bos_token_mask) | bos_token_mask
    target_is_dynamic_event = targets > 1
    target_age_is_valid = targets_age > MASK_TIME / 2
    return source_is_dynamic & target_is_dynamic_event & target_age_is_valid


def exact_time_loss_mask(
    next_event_mask: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    bos_token_mask: torch.Tensor,
) -> torch.Tensor:
    if not (next_event_mask.shape == age.shape == targets_age.shape == bos_token_mask.shape):
        raise ValueError("Track R time-loss tensors must have identical shapes")
    return next_event_mask & ~bos_token_mask & (targets_age > age)


def apply_static_temporal_visibility(
    attention_mask: torch.Tensor,
    age: torch.Tensor,
    static_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Hide recruitment-time context from queries that occur before recruitment."""
    if attention_mask.ndim != 4 or attention_mask.size(1) != 1:
        raise ValueError("attention_mask must have shape [batch,1,query,key]")
    if age.shape != static_token_mask.shape:
        raise ValueError("age and static_token_mask must have identical shapes")
    query_age = age[:, None, :, None]
    key_age = age[:, None, None, :]
    static_key = static_token_mask[:, None, None, :]
    query_is_static = static_token_mask[:, None, :, None]
    static_key_is_available = query_is_static | (query_age >= key_age)
    return attention_mask & (~static_key | static_key_is_available)


def medbert_attention_allowed(
    token_ids: torch.Tensor,
    age: torch.Tensor,
    static_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional valid-token attention with recruitment-time static gating."""
    valid = token_ids > 0
    allowed = valid[:, :, None] & valid[:, None, :]
    query_age = age[:, :, None]
    key_age = age[:, None, :]
    static_key = static_token_mask[:, None, :]
    static_query = static_token_mask[:, :, None]
    return allowed & (~static_key | static_query | (query_age >= key_age))


def mask_medbert_inputs(
    token_ids: torch.Tensor,
    static_token_mask: torch.Tensor,
    bos_token_mask: torch.Tensor,
    mask_token_id: int,
    dynamic_vocab_size: int,
    generator: torch.Generator,
    probability: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen Med-BERT 35% MLM and 80/10/10 replacement rule."""
    eligible = (token_ids > 1) & ~static_token_mask & ~bos_token_mask & (token_ids < int(dynamic_vocab_size))
    selected = eligible & (torch.rand(token_ids.shape, generator=generator, device=token_ids.device) < probability)
    labels = token_ids.masked_fill(~selected, -1)
    replacement_draw = torch.rand(token_ids.shape, generator=generator, device=token_ids.device)
    output = token_ids.clone()
    output[selected & (replacement_draw < 0.8)] = int(mask_token_id)
    random_mask = selected & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
    random_tokens = torch.randint(
        2,
        int(dynamic_vocab_size),
        token_ids.shape,
        generator=generator,
        device=token_ids.device,
    )
    output[random_mask] = random_tokens[random_mask]
    return output, labels
