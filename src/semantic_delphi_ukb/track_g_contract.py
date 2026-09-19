from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from semantic_delphi_ukb.track_r_contract import load_track_r_protocol


TRACK_G_MODELS = ("A0", "A2", "ETHOS-Matched", "Foresight-Matched")
TRAINED_TRACK_G_FAMILIES = ("ethos_matched", "foresight_matched")


def protocol_contract_sha256(protocol: Mapping) -> str:
    payload = copy.deepcopy({key: value for key, value in dict(protocol).items() if not str(key).startswith("_")})
    payload.pop("protocol_manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(repo_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_dir / path


def validation_protocol_sha256(protocol: Mapping) -> str | None:
    authorization = protocol.get("locked_test_authorization", {})
    value = authorization.get("validation_protocol_sha256")
    return str(value) if value is not None else None


def accepted_checkpoint_protocol_hashes(protocol: Mapping) -> set[str]:
    values = {str(protocol["protocol_manifest_sha256"])}
    if protocol.get("locked_test_read") is True and protocol.get("test_authorized") is True:
        values.add(str(validation_protocol_sha256(protocol)))
    return values


def load_track_g_protocol(path: Path, repo_dir: Path | None = None) -> dict:
    path = Path(path).resolve()
    repo_dir = Path(repo_dir).resolve() if repo_dir is not None else path.parents[2]
    protocol = json.loads(path.read_text(encoding="utf-8-sig"))
    if protocol.get("protocol_id") != "track_g_v1":
        raise ValueError("expected protocol_id=track_g_v1")
    if protocol.get("status") not in {"implementation_ready_review_required", "ready_for_training"}:
        raise ValueError("Track G status must describe the review/training lifecycle")
    if protocol.get("seeds") != [42, 43, 44]:
        raise ValueError("Track G seeds must remain [42, 43, 44]")
    locked_test_read = protocol.get("locked_test_read")
    test_authorized = protocol.get("test_authorized")
    if locked_test_read not in {True, False} or test_authorized not in {True, False}:
        raise ValueError("Track G locked-test flags must be boolean")
    if locked_test_read != test_authorized:
        raise ValueError("Track G locked-test flags must be enabled together")
    authorization = protocol.get("locked_test_authorization")
    if locked_test_read:
        if protocol.get("status") != "ready_for_training":
            raise ValueError("Track G locked test requires the training-ready frozen protocol")
        if not isinstance(authorization, Mapping):
            raise ValueError("Track G locked-test authorization record is required")
        if authorization.get("scope") != "one_shot_track_g_v1_locked_test":
            raise ValueError("Track G locked-test authorization scope is invalid")
        required_hashes = (
            "validation_protocol_sha256",
            "sampler_selection_sha256",
            "validation_assessment_sha256",
        )
        if any(
            len(str(authorization.get(key, ""))) != 64
            or any(character not in "0123456789abcdef" for character in str(authorization.get(key, "")))
            for key in required_hashes
        ):
            raise ValueError("Track G locked-test authorization hashes are invalid")
    elif authorization is not None:
        raise ValueError("Track G locked-test authorization record requires enabled test flags")

    models = protocol.get("models", [])
    if tuple(model.get("name") for model in models) != TRACK_G_MODELS:
        raise ValueError(f"Track G models must be {TRACK_G_MODELS}")
    trainable = tuple(model.get("family") for model in models if model.get("train"))
    if trainable != TRAINED_TRACK_G_FAMILIES:
        raise ValueError(f"Track G trainable families must be {TRAINED_TRACK_G_FAMILIES}")
    if any(model.get("train") for model in models[:2]):
        raise ValueError("A0/A2 must reuse frozen Track R v2.2 pretraining checkpoints")

    inputs = protocol.get("matched_input_contract", {})
    required_true = (
        "static_prefix",
        "dynamic_bos",
        "same_vocabulary",
        "same_patient_splits",
        "same_next_event_mask",
        "same_time_loss_mask",
        "same_semantic_token_initialization",
        "paper_reported_numbers_are_external_reference_only",
    )
    if any(inputs.get(key) is not True for key in required_true):
        raise ValueError("Track G matched-input contract is incomplete")

    source_path = resolve_repo_path(repo_dir, protocol["source_track_r_protocol"])
    source = load_track_r_protocol(source_path)
    if source.get("protocol_manifest_sha256") != protocol.get("source_track_r_protocol_sha256"):
        raise ValueError("Track G source Track R protocol hash mismatch")
    if source.get("data_protocol_id") != protocol.get("data_protocol_id"):
        raise ValueError("Track G data protocol does not match Track R v2.2")

    sampling = protocol.get("generation_evaluation", {})
    if sampling.get("waiting_time_sampling") != (
        "exponential_total_rate_matching_filtered_track_r_v2_2_evaluator_log_rate"
    ):
        raise ValueError("Track G waiting-time sampler is not frozen")
    if sampling.get("temperature_applies_to_event_identity_only") is not True:
        raise ValueError("Track G temperature must not alter the frozen time rate")
    if sampling.get("death_logit_bias_applies_to_event_identity_only") is not True:
        raise ValueError("Track G death bias must not alter the frozen time rate")
    if sampling.get("waiting_time_rate_uses_untempered_unbiased_logits") is not True:
        raise ValueError("Track G waiting-time rate must use original logits")

    rates = protocol.get("time_rate_contracts", {})
    if rates.get("historical_training_objective") != (
        "full_vocabulary_logsumexp_in_non_validation_training_mode"
    ):
        raise ValueError("Track G historical training-rate contract is incomplete")
    if rates.get("generation_and_evaluation_objective") != (
        "filtered_dynamic_clinical_event_logsumexp_excluding_ignore_tokens_and_no_event"
    ):
        raise ValueError("Track G evaluator-rate contract is incomplete")
    if rates.get("historical_a0_a2_checkpoints_are_not_retrained_or_reinterpreted") is not True:
        raise ValueError("Track G must preserve historical A0/A2 checkpoints")
    if rates.get("same_day_targets_are_excluded_from_waiting_time_nll") is not True:
        raise ValueError("Track G same-day waiting-time exclusion is not frozen")

    selection = protocol.get("sampler_selection", {})
    if selection.get("family_members", {}).get("carope") != ["A0", "A2"]:
        raise ValueError("Track G A0/A2 must explicitly share the carope sampler")
    if selection.get("tie_breakers") != [
        "first_event_top10",
        "time_mae_days",
        "death_brier",
        "lower_temperature",
        "lower_top_p",
        "death_logit_bias_closest_to_zero",
    ]:
        raise ValueError("Track G sampler tie-breakers are not fully frozen")
    if sampling.get("common_random_numbers_scope") != (
        "matched_only_until_trajectory_state_or_active_set_diverges"
    ):
        raise ValueError("Track G common-random-number scope is not frozen")
    if sampling.get("report_minimum_wait_clamp_rate") is not True:
        raise ValueError("Track G must report minimum-wait clamping")

    integrity = protocol.get("integrity", {})
    integrity_true = (
        "validation_only_until_explicit_test_authorization",
        "do_not_select_or_drop_training_seeds_post_hoc",
        "report_seed_44_even_if_time_generation_is_unstable",
        "no_test_commands_in_default_launcher",
        "no_remote_deletion",
        "no_participant_ids_in_logs",
    )
    if any(integrity.get(key) is not True for key in integrity_true):
        raise ValueError("Track G integrity contract is incomplete")

    expected = protocol.get("protocol_manifest_sha256")
    actual = protocol_contract_sha256(protocol)
    if expected is not None and expected != actual:
        raise ValueError("Track G protocol manifest SHA-256 mismatch")
    protocol["_path"] = path
    protocol["_repo_dir"] = repo_dir
    protocol["_source_track_r"] = source
    protocol["_source_track_r_path"] = source_path
    return protocol


def assert_split_allowed(protocol: Mapping, split: str) -> None:
    if split not in {"val", "test"}:
        raise ValueError("Track G evaluation split must be val or test")
    if split == "test" and (
        protocol.get("test_authorized") is not True or protocol.get("locked_test_read") is not True
    ):
        raise PermissionError("Track G locked test is not authorized by the frozen protocol")


def assert_training_allowed(protocol: Mapping) -> None:
    if protocol.get("status") != "ready_for_training":
        raise PermissionError(
            "Track G training is blocked until review changes status to ready_for_training and rehashes the protocol"
        )


def model_spec(protocol: Mapping, name: str) -> dict:
    for value in protocol["models"]:
        if value["name"] == name:
            return dict(value)
    raise ValueError(f"unregistered Track G model: {name}")
