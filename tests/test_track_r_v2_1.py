import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_track_r_v2_1 import baseline_jobs, paths, transformer_jobs
from semantic_delphi_ukb.calibration_auc import build_horizon_case_control, horizon_case_control_at_age
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory
from semantic_delphi_ukb.medbert_r_model import MedBERTR, MedBERTRConfig, parameter_report
from semantic_delphi_ukb.track_r_contract import (
    apply_static_temporal_visibility,
    bmi_category,
    exact_time_loss_mask,
    load_track_r_protocol,
    mask_medbert_inputs,
    medbert_attention_allowed,
    prepend_static_context,
)
from semantic_delphi_ukb.track_r_rows import read_json, read_json_bytes, resolve_rows_path, write_json_gzip
from semantic_delphi_ukb.track_r_baselines import main as baseline_main, standardize_for_cox


PROTOCOL_PATH = ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json"


def test_protocol_freezes_nonchronological_anchor_redundancy_and_loss_contract():
    protocol = load_track_r_protocol(PROTOCOL_PATH)
    anchor = protocol["static_prefix"]["age_anchor"]
    assert anchor["allow_anchor_after_first_dynamic_event"] is True
    assert anchor["sequence_order_is_not_chronological"] is True
    assert anchor["forbid_age_sort_after_prefix"] is True
    assert protocol["redundancy_policy"]["main_track_numeric_features"] == []
    assert set(protocol["redundancy_policy"]["exclude_from_main_track"]) == {"height", "weight"}
    assert protocol["loss_contract"]["static_to_first_dynamic_transition"] is False
    assert protocol["dynamic_bos"]["required"] is True
    assert protocol["dynamic_bos"]["age_anchor"] == "first_dynamic_clinical_event_age_in_retained_window"
    assert protocol["loss_contract"]["dynamic_bos_to_first_dynamic_ce"] is True
    assert protocol["loss_contract"]["dynamic_bos_to_first_dynamic_time_loss"] is False
    assert protocol["sequence_lengths"]["A1-TokenStatic"] == 133
    assert protocol["sequence_lengths"]["A0-noStatic"] == 129
    assert protocol["sequence_lengths"]["A1-noStatic"] == 129
    assert protocol["validation_reporting_policy"]["validation_metrics_are_selection_biased_descriptive"] is True
    assert protocol["static_fields"]["alcohol"]["coding"]["2"] == "three_or_four_times_per_week"
    assert protocol["static_fields"]["alcohol"]["coding"]["3"] == "once_or_twice_per_week"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "missing"),
        (0.0, "missing"),
        (-3.0, "missing"),
        (18.49, "lt18.5"),
        (18.5, "18.5_to_lt25"),
        (25.0, "25_to_lt30"),
        (30.0, "gte30"),
    ],
)
def test_bmi_bins_are_frozen(value, expected):
    assert bmi_category(value) == expected


def test_prefix_loss_masks_exclude_every_static_transition_and_require_positive_time():
    x = torch.tensor([[10, 11, 12]])
    age = torch.tensor([[100.0, 200.0, 300.0]])
    y = torch.tensor([[11, 12, 13]])
    target_age = torch.tensor([[200.0, 300.0, 400.0]])
    prefix = torch.tensor([[20, 21, 22, 23]])
    x_out, age_out, y_out, target_out, static_mask, bos_mask, next_mask = prepend_static_context(
        x, age, y, target_age, prefix, torch.tensor([250.0]), bos_token_id=24
    )
    assert x_out.tolist() == [[20, 21, 22, 23, 24, 10, 11, 12]]
    assert age_out.tolist()[:1] == [[250.0, 250.0, 250.0, 250.0, 100.0, 100.0, 200.0, 300.0]]
    assert y_out[0, :4].tolist() == [-1, -1, -1, -1]
    assert not next_mask[0, :4].any()
    assert bos_mask.tolist() == [[False, False, False, False, True, False, False, False]]
    assert y_out[0, 4].item() == 10
    assert next_mask[0, 4:].tolist() == [True, True, True, True]
    time_mask = exact_time_loss_mask(next_mask, age_out, target_out, bos_mask)
    assert time_mask[0, 4:].tolist() == [False, True, True, True]


def test_dynamic_bos_restores_single_event_ce_without_time_loss():
    x = torch.tensor([[0, 10]])
    age = torch.tensor([[-10000.0, 100.0]])
    y = torch.tensor([[0, 1]])
    target_age = torch.tensor([[-10000.0, 100.0]])
    prefix = torch.tensor([[20, 21, 22, 23]])
    _, age_out, y_out, target_out, _, bos_mask, next_mask = prepend_static_context(
        x, age, y, target_age, prefix, torch.tensor([250.0]), bos_token_id=24
    )
    assert y_out[bos_mask].tolist() == [10]
    assert next_mask[bos_mask].tolist() == [True]
    assert exact_time_loss_mask(next_mask, age_out, target_out, bos_mask)[bos_mask].tolist() == [False]


def test_dynamic_bos_deduplicates_first_event_after_no_event_source():
    x = torch.tensor([[1, 10, 11]])
    age = torch.tensor([[50.0, 100.0, 200.0]])
    y = torch.tensor([[10, 11, 12]])
    target_age = torch.tensor([[100.0, 200.0, 300.0]])
    prefix = torch.tensor([[20, 21, 22, 23]])
    _, _, y_out, _, _, bos_mask, next_mask = prepend_static_context(
        x, age, y, target_age, prefix, torch.tensor([250.0]), bos_token_id=24
    )
    assert y_out[bos_mask].tolist() == [10]
    assert next_mask[0, 5:].tolist() == [False, True, True]


def test_horizon_case_control_excludes_events_after_followup():
    label, eligible = horizon_case_control_at_age(100.0, [200.0], 150.0, 1.0)
    assert (label, eligible) == (0, False)
    labels, mask = build_horizon_case_control(
        np.asarray([0]),
        np.asarray([100.0]),
        [[np.asarray([200.0])]],
        np.asarray([150.0]),
        1.0,
        0,
    )
    assert labels.tolist() == [0]
    assert mask.tolist() == [False]


def test_track_r_rows_round_trip_deterministic_gzip(tmp_path):
    rows = [{"patient_index": 1, "label": 0, "score": 0.25}]
    first = write_json_gzip(tmp_path / "rows.json.gz", rows)
    second = write_json_gzip(tmp_path / "rows-copy.json.gz", rows)
    assert first.read_bytes() == second.read_bytes()
    assert resolve_rows_path(tmp_path / "rows.json") == first
    assert read_json(first) == rows
    assert json.loads(read_json_bytes(first)) == rows


def test_cox_standardization_caps_sparse_feature_outliers():
    train = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 100.0]], dtype=np.float32)
    evaluate = np.asarray([[3.0, 1000.0]], dtype=np.float32)
    train_scaled, eval_scaled, mean, std = standardize_for_cox(train, evaluate)
    assert train_scaled.dtype == np.float64
    assert eval_scaled.dtype == np.float64
    assert np.isfinite(train_scaled).all()
    assert np.isfinite(eval_scaled).all()
    assert np.abs(train_scaled).max() <= 20.0
    assert np.abs(eval_scaled).max() <= 20.0
    assert (std >= 0.05).all()
    assert mean.shape == (2,)


def test_logistic_baseline_writes_compressed_rows_summary_and_status(tmp_path):
    rng = np.random.default_rng(7)
    count = 24
    features = rng.normal(size=(count, 4)).astype(np.float32)
    labels = np.zeros((count, 3, 2), dtype=np.float32)
    labels[::2, :, 0] = 1.0
    labels[1::3, :, 1] = 1.0
    mask = np.ones_like(labels)
    artifact = {
        "x": features,
        "labels": labels,
        "label_mask": mask,
        "survival_duration": np.ones((count, 2), dtype=np.float32),
        "survival_event": labels[:, 0, :],
        "patient_index": np.arange(count, dtype=np.int64),
        "prediction_age_days": np.full(count, 60.0 * 365.25, dtype=np.float32),
        "age_start_years": np.full(count, 60.0, dtype=np.float32),
        "sex": np.asarray(["female", "male"] * (count // 2)),
    }
    for split in ("train", "val"):
        path = tmp_path / f"{split}.npz"
        np.savez_compressed(path, **artifact)
        path.with_suffix(".schema.json").write_text(json.dumps({
            "protocol_id": "track_r_v2_1",
            "split": "val",
            "disease_ids": ["d0", "d1"],
            "horizons_years": [1.0, 5.0, 10.0],
        }), encoding="utf-8")
    checkpoint = tmp_path / "logistic.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump({
            "models": {f"{horizon}:{disease}": {"constant_prior": 0.25} for horizon in range(3) for disease in range(2)},
            "train_mean": np.zeros(4, dtype=np.float32),
            "train_std": np.ones(4, dtype=np.float32),
            "shape": (3, 2),
        }, handle)
    output = tmp_path / "logistic"
    assert baseline_main([
        "--model", "logistic",
        "--train-features", str(tmp_path / "train.npz"),
        "--val-features", str(tmp_path / "val.npz"),
        "--eval-features", str(tmp_path / "val.npz"),
        "--output-dir", str(output),
        "--checkpoint", str(checkpoint),
        "--predict-only",
    ]) == 0
    assert len(read_json(output / "rows.json.gz")) == count * 3 * 2
    assert json.loads((output / "status.json").read_text())["status"] == "finished"
    assert json.loads((output / "summary.json").read_text())["patient_rows"] == count * 3 * 2


def test_static_prefix_visibility_is_gated_by_recruitment_age_not_prefix_position():
    mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    ages = torch.tensor([[60.0, 50.0, 70.0]])
    static = torch.tensor([[True, False, False]])
    visible = apply_static_temporal_visibility(mask, ages, static)
    assert visible[0, 0, 1, 0].item() is False
    assert visible[0, 0, 2, 0].item() is True
    allowed = medbert_attention_allowed(torch.tensor([[5, 6, 7]]), ages, static)
    assert allowed[0, 1, 0].item() is False
    assert allowed[0, 2, 0].item() is True


def test_car_rope_explicit_track_r_masks_make_empty_loss_finite():
    model = CARoPEHorizonMedTrajectory(
        CARoPEConfig(
            block_size=4,
            vocab_size=32,
            n_layer=1,
            n_head=4,
            n_embd=32,
            static_dim=0,
            num_tte_tasks=2,
            num_horizons=3,
            use_relative_horizon_query=False,
        )
    )
    x = torch.tensor([[5, 6, 7, 8]])
    age = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    targets = torch.tensor([[6, 7, 8, 9]])
    static_mask = torch.ones_like(x, dtype=torch.bool)
    empty = torch.zeros_like(x, dtype=torch.bool)
    _, parts, *_ = model(
        x,
        age,
        None,
        targets,
        age + 1.0,
        static_token_mask=static_mask,
        next_event_mask=empty,
        time_loss_mask=empty,
    )
    assert torch.isfinite(parts["loss_ce"])
    assert torch.isfinite(parts["loss_dt"])


def test_medbert_variants_report_capacity_separately_and_mlm_never_masks_static():
    paper = MedBERTR(MedBERTRConfig(vocab_size=100, n_layer=6, n_head=6, hidden_size=192, intermediate_size=64))
    matched = MedBERTR(MedBERTRConfig(vocab_size=100, n_layer=6, n_head=8, hidden_size=64, intermediate_size=256))
    paper_report = parameter_report(paper)
    matched_report = parameter_report(matched)
    assert paper_report["total_trainable_parameters"] > matched_report["total_trainable_parameters"]
    assert paper_report["transformer_block_parameters"] != paper_report["total_trainable_parameters"]
    tokens = torch.tensor([[50, 2, 3, 1]])
    static = torch.tensor([[True, False, False, False]])
    bos = torch.tensor([[False, True, False, False]])
    generator = torch.Generator().manual_seed(7)
    _, labels = mask_medbert_inputs(tokens, static, bos, mask_token_id=99, dynamic_vocab_size=90, generator=generator, probability=1.0)
    assert labels[0, 0].item() == -1
    assert labels[0, 1].item() == -1
    assert labels[0, 3].item() == -1
    assert labels[0, 2].item() == 3


def test_medbert_padding_queries_do_not_create_nan_hidden_states():
    model = MedBERTR(MedBERTRConfig(
        vocab_size=100,
        max_sequence_length=6,
        n_layer=2,
        n_head=4,
        hidden_size=32,
        intermediate_size=64,
        dropout=0.0,
    )).eval()
    tokens = torch.tensor([[50, 2, 3, 0, 0, 0]])
    age = torch.tensor([[100.0, 100.0, 120.0, -10000.0, -10000.0, -10000.0]])
    static = torch.tensor([[True, False, False, False, False, False]])
    allowed = medbert_attention_allowed(tokens, age, static)
    labels = torch.tensor([[-1, -1, 3, -1, -1, -1]])
    output = model(tokens, age, static, attention_allowed=allowed, mlm_labels=labels)
    assert torch.isfinite(output["hidden"]).all()
    assert torch.isfinite(output["loss_mlm"])


def test_dry_run_transformer_lane_has_static_and_true_no_static_variants():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    rendered = [" ".join(job) for job in transformer_jobs(PROTOCOL_PATH, protocol, paths(protocol), "cuda")]
    assert len(rendered) == 12
    assert any("A1-TokenStatic" in job and "--include-static-prefix true" in job for job in rendered)
    assert any(
        "A0-noStatic" in job
        and "--include-static-prefix false" in job
        and "--use-age-encoding true" in job
        and "--use-age-rope false" in job
        for job in rendered
    )
    assert any("A1-noStatic" in job and "--include-static-prefix false" in job for job in rendered)


def test_baseline_jobs_use_registered_model_names_for_run_directories():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    rendered = [" ".join(job) for job in baseline_jobs(PROTOCOL_PATH, paths(protocol))]
    assert any("Logistic-R\\validation" in job or "Logistic-R/validation" in job for job in rendered)
    assert any("Cox-R\\validation" in job or "Cox-R/validation" in job for job in rendered)
    assert any("MDRMF-Clinical-R\\validation" in job or "MDRMF-Clinical-R/validation" in job for job in rendered)
