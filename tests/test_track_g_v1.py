import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_track_g_v1 import (
    cohort_jobs,
    finalize_jobs,
    parse_pairs,
    sampler_grid_jobs,
    training_jobs,
    validation_jobs,
)
from run_track_g_locked_test import locked_test_jobs
import semantic_delphi_ukb.finalize_track_g as track_g_finalizer
from semantic_delphi_ukb.finalize_track_g import bootstrap_delta, waiting_time_valid_patient_ids
from semantic_delphi_ukb.evaluate_track_g_generation import (
    dynamic_vocab_logits,
    one_step_waiting_time_metrics,
    rollout_many,
    topk_metrics,
)
from semantic_delphi_ukb.select_track_g_sampler import select_candidate
from semantic_delphi_ukb.track_g_contract import (
    accepted_checkpoint_protocol_hashes,
    assert_split_allowed,
    assert_training_allowed,
    load_track_g_protocol,
    protocol_contract_sha256,
)
from semantic_delphi_ukb.track_g_generation import (
    GeneratedEvent,
    clinical_candidate_mask,
    frozen_log_rate,
    sample_event_and_wait,
    sample_event_and_wait_with_diagnostics,
    trajectory_validity_metrics,
    waiting_time_nll,
)
from semantic_delphi_ukb.track_g_models import (
    MATCHED_FAMILIES,
    build_matched_model,
    checkpoint_state,
    inter_event_gap_days,
    model_config_payload,
)
from semantic_delphi_ukb.train_track_g_baseline import resume_global_step


PROTOCOL = ROOT / "configs/track_g_v1/TRACK_G_v1.json"


def tiny_model(family):
    architecture = {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "dropout": 0.0,
        "ethos_gap_bucket_upper_days": [1.0, 7.0, 30.0],
    }
    embeddings = np.zeros((32, 8), dtype=np.float32)
    return build_matched_model(
        family,
        block_size=8,
        vocab_size=32,
        semantic_embedding_dim=8,
        pretrained_token_embeddings=embeddings,
        num_diseases=2,
        architecture=architecture,
        horizons_years=(1.0, 5.0),
    )


def test_protocol_is_locked_test_authorized_hashed_and_uses_all_three_seeds():
    protocol = load_track_g_protocol(PROTOCOL, ROOT)
    assert protocol["status"] == "ready_for_training"
    assert protocol["locked_test_read"] is True
    assert protocol["test_authorized"] is True
    assert protocol["seeds"] == [42, 43, 44]
    assert protocol["protocol_manifest_sha256"] == protocol_contract_sha256(protocol)
    assert [model["name"] for model in protocol["models"]] == [
        "A0", "A2", "ETHOS-Matched", "Foresight-Matched"
    ]
    assert_split_allowed(protocol, "test")
    assert_split_allowed(protocol, "val")
    assert_training_allowed(protocol)
    assert protocol["locked_test_authorization"]["validation_protocol_sha256"] in (
        accepted_checkpoint_protocol_hashes(protocol)
    )
    review_protocol = dict(protocol)
    review_protocol["status"] = "implementation_ready_review_required"
    with pytest.raises(PermissionError, match="blocked until review"):
        assert_training_allowed(review_protocol)


def test_locked_test_launcher_has_frozen_one_shot_jobs_only():
    protocol = load_track_g_protocol(PROTOCOL, ROOT)
    jobs = locked_test_jobs(PROTOCOL, protocol, "cpu")
    rendered = [" ".join(job) for job in jobs]
    assert len(jobs) == 15
    assert "--require-trained-baselines" in rendered[0]
    assert "build_track_g_cohort.py" in rendered[1] and "--split test" in rendered[1]
    assert sum("evaluate_track_g_generation.py" in command for command in rendered) == 12
    assert all("--split test" in command for command in rendered[2:-1])
    assert "finalize_track_g.py" in rendered[-1] and "--split test" in rendered[-1]
    assert all("sampler-grid" not in command and "sampler-select" not in command for command in rendered)


def test_launcher_has_six_training_jobs_144_grid_jobs_and_no_test_lane():
    protocol = load_track_g_protocol(PROTOCOL, ROOT)
    train = training_jobs(PROTOCOL, protocol, "cpu", None)
    grid = sampler_grid_jobs(PROTOCOL, protocol, "cpu", None)
    validate = validation_jobs(PROTOCOL, protocol, "cpu", None)
    finalize = finalize_jobs(PROTOCOL, protocol)
    cohort = cohort_jobs(PROTOCOL, protocol, None)
    assert len(train) == 6
    assert len(grid) == 144
    assert len(validate) == 12
    assert len(finalize) == 1
    assert len(cohort) == 1
    rendered = "\n".join(" ".join(job) for job in cohort + train + grid + validate + finalize)
    assert "--split test" not in rendered
    assert rendered.count("seed44") > 0
    assert all("risk" not in str(job[job.index("--checkpoint") + 1]) for job in grid if "--checkpoint" in job)


def test_launcher_pair_filter_only_accepts_trainable_registered_pairs():
    protocol = load_track_g_protocol(PROTOCOL, ROOT)
    pairs = parse_pairs(["42:ETHOS-Matched", "44:Foresight-Matched"], protocol)
    jobs = training_jobs(PROTOCOL, protocol, "cpu", None, pairs)
    assert len(jobs) == 2
    with pytest.raises(ValueError, match="unregistered"):
        parse_pairs(["42:A2"], protocol)
    with pytest.raises(ValueError, match="duplicate"):
        parse_pairs(["42:ETHOS-Matched", "42:ETHOS-Matched"], protocol)
    resume = training_jobs(PROTOCOL, protocol, "cpu", None, pairs, resume_existing=True)
    assert all("--resume" in job and Path(job[-1]).as_posix().endswith("checkpoints/last.pt") for job in resume)


def test_frozen_rate_and_nll_match_registered_formula():
    logits = torch.tensor([2.0, -3.0, 0.5, 1.0])
    filtered = logits.clone(); filtered[[0, 1]] = -torch.inf
    raw_lse = torch.logsumexp(filtered, dim=-1)
    expected_log_rate = -torch.log(torch.exp(-raw_lse) + 0.1)
    actual = frozen_log_rate(logits, ignore_tokens=(0,), t_min=0.1)
    assert torch.allclose(actual, expected_log_rate, rtol=0.0, atol=1e-7)
    expected_nll = -expected_log_rate + torch.exp(expected_log_rate) * (12.0 + 0.1)
    assert torch.allclose(
        waiting_time_nll(logits, 12.0, ignore_tokens=(0,), t_min=0.1),
        expected_nll,
        rtol=0.0,
        atol=1e-7,
    )


def test_protocol_distinguishes_historical_training_and_filtered_evaluator_rates():
    protocol = load_track_g_protocol(PROTOCOL, ROOT)
    rates = protocol["time_rate_contracts"]
    assert rates["historical_training_objective"].startswith("full_vocabulary_logsumexp")
    assert rates["generation_and_evaluation_objective"].startswith("filtered_dynamic_clinical_event")
    assert rates["historical_a0_a2_checkpoints_are_not_retrained_or_reinterpreted"] is True

    logits = torch.tensor([4.0, 3.0, 0.5, 1.0])
    historical_full_vocab = torch.logsumexp(logits, dim=-1)
    filtered = frozen_log_rate(logits, ignore_tokens=(0,), t_min=0.1)
    expected_filtered = -torch.log(
        torch.exp(-torch.logsumexp(logits[2:], dim=-1)) + 0.1
    )
    assert torch.allclose(filtered, expected_filtered)
    assert not torch.allclose(filtered, -torch.log(torch.exp(-historical_full_vocab) + 0.1))


def test_same_day_target_is_excluded_from_waiting_time_nll():
    logits = torch.tensor([0.0, 0.0, 1.0, 2.0])
    tied = one_step_waiting_time_metrics(logits, 100.0, 100.0, (0,), 0.1)
    future = one_step_waiting_time_metrics(logits, 101.0, 100.0, (0,), 0.1)
    assert tied == {"waiting_time_nll": None, "waiting_time_nll_valid": 0}
    assert future["waiting_time_nll_valid"] == 1
    assert np.isfinite(future["waiting_time_nll"])


def test_temperature_changes_event_distribution_but_not_waiting_time_rate():
    logits = torch.tensor([-5.0, -5.0, 0.2, 1.4])
    candidates = torch.tensor([False, False, True, True])
    common = dict(
        logits=logits,
        candidate_mask=candidates,
        ignore_tokens=(0,),
        t_min=0.1,
        top_p=1.0,
        death_token_mask=None,
        death_logit_bias=0.0,
        minimum_wait_days=0.0,
        event_uniform=torch.tensor(0.6),
        wait_uniform=torch.tensor(0.2),
    )
    _, wait_cold = sample_event_and_wait(temperature=0.5, **common)
    _, wait_hot = sample_event_and_wait(temperature=2.0, **common)
    assert wait_cold == pytest.approx(wait_hot)


def test_rate_candidate_mask_removes_excluded_terminal_hazard():
    logits = torch.tensor([-5.0, -5.0, 0.0, 10.0])
    candidates = torch.tensor([False, False, True, False])
    common = dict(
        logits=logits,
        candidate_mask=candidates,
        ignore_tokens=(0,),
        t_min=0.1,
        temperature=1.0,
        top_p=1.0,
        death_token_mask=torch.tensor([False, False, False, True]),
        death_logit_bias=0.0,
        minimum_wait_days=0.0,
        event_uniform=torch.tensor(0.5),
        wait_uniform=torch.tensor(0.2),
    )
    token_full_rate, wait_full_rate = sample_event_and_wait(**common)
    token_masked_rate, wait_masked_rate = sample_event_and_wait(
        rate_candidate_mask=candidates,
        **common,
    )
    assert token_full_rate == token_masked_rate == 2
    assert wait_masked_rate > wait_full_rate


def test_wait_clamp_and_trajectory_validity_are_measured_from_events():
    logits = torch.tensor([-5.0, -5.0, 5.0, 1.0])
    _, wait_days, clamped = sample_event_and_wait_with_diagnostics(
        logits,
        candidate_mask=torch.tensor([False, False, True, True]),
        ignore_tokens=(0,),
        t_min=0.1,
        temperature=1.0,
        top_p=1.0,
        death_token_mask=None,
        death_logit_bias=0.0,
        minimum_wait_days=1.0,
        event_uniform=torch.tensor(0.2),
        wait_uniform=torch.tensor(0.999999),
    )
    assert clamped is True
    assert wait_days == pytest.approx(1.0)

    events = [
        GeneratedEvent(2, 10.0, "diagnosis", wait_was_clamped=True),
        GeneratedEvent(3, 11.0, "death"),
        GeneratedEvent(4, 12.0, "procedure"),
        GeneratedEvent(5, 11.0, "diagnosis"),
    ]
    validity = trajectory_validity_metrics(events)
    assert validity["minimum_wait_clamp_rate"] == pytest.approx(0.25)
    assert validity["post_death_event_rate"] == pytest.approx(0.5)
    assert validity["nonmonotonic_time_rate"] == pytest.approx(1.0 / 3.0)


def test_sampler_family_and_all_deterministic_tie_breakers_are_frozen():
    protocol = load_track_g_protocol(PROTOCOL, ROOT)
    assert protocol["sampler_selection"]["family_members"]["carope"] == ["A0", "A2"]
    base = {
        "diagnosis_jaccard": 0.4,
        "first_event_top10": 0.7,
        "time_mae_days": 20.0,
        "death_brier": 0.1,
        "temperature": 1.2,
        "top_p": 1.0,
        "death_logit_bias": -3.0,
    }
    lower_temperature = {**base, "temperature": 0.8}
    assert select_candidate([base, lower_temperature]) == lower_temperature

    high_top_p = {**base, "temperature": 0.8, "top_p": 1.0}
    low_top_p = {**high_top_p, "top_p": 0.9}
    assert select_candidate([high_top_p, low_top_p]) == low_top_p

    biased = {**low_top_p, "death_logit_bias": -3.0}
    unbiased = {**low_top_p, "death_logit_bias": 0.0}
    assert select_candidate([biased, unbiased]) == unbiased


def test_tmux_wrapper_explicitly_invokes_bash_for_bash_quoting():
    wrapper = (ROOT / "scripts/launch_track_g_tmux.sh").read_text(encoding="utf-8")
    assert "exec bash -lc" in wrapper


def test_ethos_gap_embedding_uses_nonnegative_consecutive_intervals():
    age = torch.tensor([[200.0, 200.0, 100.0, 130.0, 125.0, 500.0]])
    assert inter_event_gap_days(age).tolist() == [[0.0, 0.0, 0.0, 30.0, 0.0, 375.0]]


@pytest.mark.parametrize("family", MATCHED_FAMILIES)
def test_matched_models_share_track_r_masks_and_checkpoint_roundtrip(family):
    model = tiny_model(family).eval()
    x = torch.tensor([[20, 21, 22, 23, 24, 4, 5, 6]])
    age = torch.tensor([[200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 130.0, 160.0]])
    static = torch.zeros_like(x, dtype=torch.bool); static[:, :4] = True
    bos = torch.zeros_like(x, dtype=torch.bool); bos[:, 4] = True
    logits, *_ = model(x, age, None, static_token_mask=static, bos_token_mask=bos)
    assert logits.shape == (1, 8, 32)
    state = {
        "model": model.state_dict(),
        "model_args": model_config_payload(model),
        "track_g_family": family,
    }
    loaded, loaded_family = checkpoint_state(state, expected_family=family)
    loaded.eval()
    loaded_logits, *_ = loaded(x, age, None, static_token_mask=static, bos_token_mask=bos)
    assert loaded_family == family
    assert torch.equal(logits, loaded_logits)


def test_bootstrap_keeps_seed_deltas_separate_before_averaging():
    patient_ids = [0, 1]
    left = {
        42: {0: {"metric": 3.0}, 1: {"metric": 5.0}},
        43: {0: {"metric": 4.0}, 1: {"metric": 6.0}},
        44: {0: {"metric": 7.0}, 1: {"metric": 9.0}},
    }
    right = {
        seed: {0: {"metric": 1.0}, 1: {"metric": 1.0}} for seed in (42, 43, 44)
    }
    result = bootstrap_delta(left, right, patient_ids, "metric", 100, 9)
    assert result["per_seed_delta"] == {"42": 3.0, "43": 4.0, "44": 7.0}
    assert result["delta"] == pytest.approx(14.0 / 3.0)
    assert result["patient_count"] == 2


def test_bootstrap_filters_none_and_nan_for_every_metric_and_reports_subset_count():
    patient_ids = [0, 1, 2]
    left = {
        seed: {
            0: {"waiting_time_nll": None, "first_event_time_mae_days": 4.0},
            1: {"waiting_time_nll": 3.0, "first_event_time_mae_days": float("nan")},
            2: {"waiting_time_nll": 5.0, "first_event_time_mae_days": 8.0},
        }
        for seed in (42, 43, 44)
    }
    right = {
        seed: {
            0: {"waiting_time_nll": None, "first_event_time_mae_days": 2.0},
            1: {"waiting_time_nll": 2.0, "first_event_time_mae_days": 3.0},
            2: {"waiting_time_nll": 3.0, "first_event_time_mae_days": 5.0},
        }
        for seed in (42, 43, 44)
    }

    nll = bootstrap_delta(left, right, patient_ids, "waiting_time_nll", 100, 9)
    time_mae = bootstrap_delta(left, right, patient_ids, "first_event_time_mae_days", 100, 9)
    assert len(patient_ids) == 3
    assert nll["patient_count"] == 2
    assert time_mae["patient_count"] == 2
    assert nll["delta"] == pytest.approx(1.5)
    assert time_mae["delta"] == pytest.approx(2.5)


def test_bootstrap_empty_metric_subset_returns_null_interval_without_warning():
    values = {
        seed: {0: {"metric": None}, 1: {"metric": float("nan")}}
        for seed in (42, 43, 44)
    }
    result = bootstrap_delta(values, values, [0, 1], "metric", 100, 9)
    assert result == {
        "delta": None,
        "ci95": [None, None],
        "ci95_low": None,
        "ci95_high": None,
        "per_seed_delta": {"42": None, "43": None, "44": None},
        "patient_count": 0,
    }


def test_waiting_time_validity_is_data_invariant_across_models_and_seeds():
    patient_ids = [0, 1]
    rows = {
        model: {
            seed: {
                0: {"waiting_time_nll_valid": 0},
                1: {"waiting_time_nll_valid": 1},
            }
            for seed in (42, 43, 44)
        }
        for model in ("A0", "A2", "ETHOS-Matched", "Foresight-Matched")
    }
    assert waiting_time_valid_patient_ids(rows, patient_ids) == [1]
    rows["A2"][44][0]["waiting_time_nll_valid"] = 1
    with pytest.raises(ValueError, match="not model/seed invariant"):
        waiting_time_valid_patient_ids(rows, patient_ids)


def test_final_assessment_distinguishes_full_and_metric_patient_counts(tmp_path, monkeypatch):
    models = ["A0", "A2", "ETHOS-Matched", "Foresight-Matched"]
    protocol = {
        "protocol_id": "track_g_v1_test",
        "protocol_manifest_sha256": "test-hash",
        "output_root": "unused",
        "locked_test_read": False,
        "seeds": [42, 43, 44],
        "models": [{"name": name} for name in models],
        "statistics": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 9,
            "primary_contrast": {"left": "A2", "right": "A0"},
            "external_contrasts": [
                {"left": "A2", "right": "ETHOS-Matched"},
                {"left": "A2", "right": "Foresight-Matched"},
            ],
        },
    }
    monkeypatch.setattr(track_g_finalizer, "load_track_g_protocol", lambda *_: protocol)
    monkeypatch.setattr(track_g_finalizer, "assert_training_allowed", lambda *_: None)
    monkeypatch.setattr(track_g_finalizer, "assert_split_allowed", lambda *_: None)

    input_root = tmp_path / "val"
    for seed in protocol["seeds"]:
        for model_index, model in enumerate(models):
            rows = [
                {
                    "patient_index": 0,
                    "hit_at_10": 1.0,
                    "waiting_time_nll": None,
                    "waiting_time_nll_valid": 0,
                    "diagnosis_jaccard": 0.2,
                    "first_event_time_mae_days": 5.0,
                    "event_count_mae": 1.0,
                    "death_brier": 0.1,
                },
                {
                    "patient_index": 1,
                    "hit_at_10": 1.0,
                    "waiting_time_nll": 3.0 + model_index,
                    "waiting_time_nll_valid": 1,
                    "diagnosis_jaccard": 0.3,
                    "first_event_time_mae_days": 4.0,
                    "event_count_mae": 1.0,
                    "death_brier": 0.1,
                },
            ]
            path = input_root / f"seed{seed}" / model / "patient_rows.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "assessment"
    assert track_g_finalizer.main([
        "--input-root", str(input_root),
        "--out-dir", str(out_dir),
    ]) == 0
    assessment = json.loads((out_dir / "assessment.json").read_text(encoding="utf-8"))
    assert assessment["patient_count"] == 2
    assert assessment["waiting_time_nll_data_valid_patient_count"] == 1
    assert assessment["contrasts"]["A2_minus_A0"]["waiting_time_nll"]["patient_count"] == 1
    assert assessment["contrasts"]["A2_minus_A0"]["hit_at_10"]["patient_count"] == 2


def test_resume_restarts_at_checkpointed_completed_step_without_skipping():
    assert resume_global_step({"iteration": 500, "global_step": 500}) == 500
    with pytest.raises(ValueError, match="mismatch"):
        resume_global_step({"iteration": 499, "global_step": 500})


def test_clinical_candidate_mask_excludes_static_bos_and_no_event():
    labels = ["Padding", "No event", "diag:I10", "proc:K40", "static:sex:male", "dynamic:BOS"]
    assert clinical_candidate_mask(labels, "cpu").tolist() == [False, False, True, True, False, False]


def test_full_vocab_logits_are_restricted_to_dynamic_vocabulary_for_generation_metrics():
    logits = torch.tensor([0.0, 0.0, 3.0, 2.0, 100.0, 90.0])
    candidates = torch.tensor([False, False, True, True])
    restricted = dynamic_vocab_logits(logits, candidates)
    assert restricted.tolist() == [0.0, 0.0, 3.0, 2.0]
    assert topk_metrics(logits, 2, candidates)["hit_at_1"] == 1
    with pytest.raises(ValueError, match="outside the dynamic vocabulary"):
        topk_metrics(logits, 4, candidates)


def test_rollout_batches_sampling_replicates_and_preserves_monotonic_time():
    model = tiny_model("foresight_matched").eval()
    labels = ["Padding", "No event"] + [f"diag:D{index}" for index in range(2, 32)]
    for token_id in (20, 21, 22, 23):
        labels[token_id] = f"static:test:{token_id}"
    labels[24] = "dynamic:BOS"
    tokens = torch.tensor([[20, 21, 22, 23, 24, 4, 5, 6]])
    ages = torch.tensor([[200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 130.0, 160.0]])
    static = torch.zeros_like(tokens, dtype=torch.bool); static[:, :4] = True
    bos = torch.zeros_like(tokens, dtype=torch.bool); bos[:, 4] = True
    settings = {
        "sampling_seed": 17,
        "followup_years": 1.0,
        "max_new_tokens": 3,
        "minimum_wait_days": 1.0,
        "stop_after_death": True,
    }
    sampler = {"temperature": 1.0, "top_p": 1.0, "death_logit_bias": 0.0, "num_rollouts": 4}
    generated = rollout_many(
        model,
        labels,
        tokens,
        ages,
        static,
        bos,
        clinical_candidate_mask(labels, "cpu"),
        settings,
        sampler,
        patient_index=9,
    )
    assert len(generated) == 4
    for events in generated:
        assert all(events[index].age_days > events[index - 1].age_days for index in range(1, len(events)))
