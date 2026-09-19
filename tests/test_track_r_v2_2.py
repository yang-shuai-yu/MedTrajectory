import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_track_r_v2_2 import evaluation_jobs, parse_pairs, training_jobs
from semantic_delphi_ukb.car_rope_model import (
    AdditiveAgeRoPEV22,
    CARoPECausalSelfAttention,
    CARoPEConfig,
    CARoPEHorizonMedTrajectory,
)
from semantic_delphi_ukb.finalize_track_r_v2_2 import (
    bootstrap_patient_mean,
    patient_nll_payload,
    seed_averaged_bootstrap_delta,
)
from semantic_delphi_ukb.track_r_contract import (
    load_track_r_protocol,
    protocol_contract_sha256,
)
from semantic_delphi_ukb.track_r_batch import validate_track_r_data_manifest
from semantic_delphi_ukb.track_r_rows import write_json_gzip
from semantic_delphi_ukb.track_r_v2_2 import (
    build_wavelength_manifest,
    canonical_sha256,
    deterministic_patient_batch,
    event_time_nll_rows,
    load_wavelength_manifest,
    positive_consecutive_gaps,
    write_wavelength_manifest,
)


PROTOCOL = ROOT / "configs/track_r_v2_2/TRACK_R_v2_2.json"


def test_v2_2_protocol_freezes_independent_lane_and_two_stage_last_checkpoints():
    protocol = load_track_r_protocol(PROTOCOL)
    assert protocol["protocol_id"] == "track_r_v2_2"
    assert protocol["data_protocol_id"] == "track_r_v2_1"
    assert protocol["locked_test_read"] is False
    assert protocol["test_authorized"] is False
    assert protocol["seeds"] == [42, 43, 44]
    assert protocol["pretraining_fidelity"]["nll_equivalence_atol"] == 1e-6
    checkpoints = protocol["checkpoint_contract"]
    assert checkpoints["main_nll"].endswith("pretraining/{model}/checkpoints/last.pt")
    assert checkpoints["clinical_auroc"].endswith("risk/{model}/checkpoints/last.pt")


def test_v2_2_validates_the_reused_v2_1_data_protocol_manifest(tmp_path):
    protocol = load_track_r_protocol(PROTOCOL)
    manifest = {
        "protocol_id": "track_r_v2_1",
        "loss_contract": protocol["loss_contract"],
        "static_prefix": {"age_anchor": protocol["static_prefix"]["age_anchor"]},
        "dynamic_bos": protocol["dynamic_bos"],
        "dynamic_bos_token_id": 123,
    }
    (tmp_path / "prepare_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_track_r_data_manifest(tmp_path, protocol)["protocol_id"] == "track_r_v2_1"
    manifest["protocol_id"] = "track_r_v2_2"
    (tmp_path / "prepare_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        validate_track_r_data_manifest(tmp_path, protocol)


def test_wavelength_grid_is_log_spaced_and_interleaved_across_heads():
    manifest = build_wavelength_manifest(np.geomspace(2.0, 1000.0, 1000), n_heads=8, head_dim=8)
    global_grid = np.asarray(manifest["wavelengths_days_global_sorted"])
    head_major = np.asarray(manifest["wavelengths_days_head_major"]).reshape(8, 4)
    assert len(global_grid) == 32
    assert np.allclose(np.diff(np.log(global_grid)), np.diff(np.log(global_grid))[0])
    for head in range(8):
        assert np.allclose(head_major[head], global_grid[head::8])
        assert head_major[head, 0] < head_major[head, -1]


def test_positive_gaps_sort_each_patient_by_age_before_differencing():
    ordered = np.asarray([
        [0, 100, 2], [0, 200, 3], [0, 350, 4],
        [1, 50, 2], [1, 75, 3], [1, 75, 4], [1, 125, 5],
    ], dtype=np.uint32)
    shuffled = ordered[[2, 0, 1, 5, 3, 6, 4]]
    p2i = np.asarray([[0, 3], [3, 4]], dtype=np.int64)
    assert np.array_equal(
        np.sort(positive_consecutive_gaps(shuffled, p2i)),
        np.asarray([25.0, 50.0, 100.0, 150.0]),
    )


def _nll_row(patient, source_ordinal, nll):
    return {
        "patient_index": patient,
        "original_source_event_ordinal": source_ordinal,
        "original_target_event_ordinal": source_ordinal + 1,
        "context_window_start": 0,
        "local_source_position": source_ordinal + 5,
        "source_token_id": 10 + source_ordinal,
        "source_age_days": 100.0 + source_ordinal,
        "target_token_id": 11 + source_ordinal,
        "target_age_days": 101.0 + source_ordinal,
        "positive_gap_days": 1.0,
        "nll": nll,
    }


def test_patient_nll_payload_preserves_pairing_identity_and_requires_frozen_order():
    rows = [_nll_row(0, 0, 1.0), _nll_row(0, 1, 3.0), _nll_row(1, 0, 5.0)]
    payload = patient_nll_payload(rows)
    same_events = patient_nll_payload([
        _nll_row(0, 0, 2.0), _nll_row(0, 1, 4.0), _nll_row(1, 0, 6.0)
    ])
    assert payload["identity_sha256"] == same_events["identity_sha256"]
    assert payload["patients"].tolist() == [0, 1]
    assert payload["patient_mean_nll"].tolist() == [2.0, 5.0]
    with pytest.raises(ValueError, match="strictly ordered"):
        patient_nll_payload(list(reversed(rows)))


def test_patient_bootstrap_uses_shared_indices_across_seeds_and_is_deterministic():
    delta = np.asarray([[1.0, 4.0, 9.0], [10.0, 40.0, 90.0], [100.0, 400.0, 900.0]])
    result = bootstrap_patient_mean(delta, replicates=5, seed=7, percentile=[0.0, 100.0])
    repeated = bootstrap_patient_mean(delta, replicates=5, seed=7, percentile=[0.0, 100.0])
    rng = np.random.default_rng(7)
    expected = []
    for _ in range(5):
        selected = rng.integers(0, 3, size=3)
        expected.append(np.mean([values[selected].mean() for values in delta]))
    assert result == repeated
    assert result["ci95_low"] == pytest.approx(min(expected))
    assert result["ci95_high"] == pytest.approx(max(expected))
    assert result["delta"] == pytest.approx(delta.mean(axis=1).mean())


def test_clinical_bootstrap_subtracts_each_seed_before_averaging():
    names = ["seed42:A0", "seed42:A2", "seed43:A0", "seed43:A2", "seed44:A0", "seed44:A2"]
    values = np.asarray([
        [0.50, 0.60, 0.70, 0.65, 0.80, 0.82],
        [0.40, 0.45, 0.60, 0.70, 0.90, 0.87],
    ])
    per_seed, result = seed_averaged_bootstrap_delta(
        values, names, [42, 43, 44], "A2", "A0", [0.0, 100.0]
    )
    expected = np.asarray([[0.10, -0.05, 0.02], [0.05, 0.10, -0.03]])
    assert np.allclose(per_seed, expected)
    assert result["ci95_low"] == pytest.approx(expected.mean(axis=1).min())
    assert result["ci95_high"] == pytest.approx(expected.mean(axis=1).max())


def test_wavelength_manifest_rejects_expected_hash_and_scale_mismatches(tmp_path):
    manifest = build_wavelength_manifest(np.geomspace(2.0, 1000.0, 1000), n_heads=8, head_dim=8)
    path = write_wavelength_manifest(tmp_path / "wavelengths.json", manifest)
    loaded = load_wavelength_manifest(
        path,
        expected_sha256=manifest["sha256"],
        scale_factor_bounds=[0.5, 2.0],
        require_log_scale_match=True,
    )
    assert loaded == manifest
    with pytest.raises(ValueError, match="expected_sha256"):
        load_wavelength_manifest(path, expected_sha256="0" * 64)
    altered = dict(manifest)
    altered["log_scale_bounds"] = [float(np.log(0.6)), float(np.log(2.0))]
    altered["sha256"] = canonical_sha256(altered)
    altered_path = write_wavelength_manifest(tmp_path / "altered.json", altered)
    with pytest.raises(ValueError, match="scale_factor_bounds"):
        load_wavelength_manifest(
            altered_path,
            expected_sha256=altered["sha256"],
            scale_factor_bounds=[0.5, 2.0],
            require_log_scale_match=True,
        )


def test_ready_for_training_protocol_requires_manifest_hash_and_frozen_seeds(tmp_path):
    protocol = load_track_r_protocol(PROTOCOL)
    protocol.pop("extends", None)
    protocol["status"] = "ready_for_training"
    protocol["wavelength_contract"].pop("expected_sha256", None)
    protocol["protocol_manifest_sha256"] = protocol_contract_sha256(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_sha256"):
        load_track_r_protocol(path)

    protocol["status"] = "implementation_ready"
    protocol["seeds"] = [42, 43]
    protocol["protocol_manifest_sha256"] = protocol_contract_sha256(protocol)
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="seeds"):
        load_track_r_protocol(path)


def test_additive_rope_is_translation_invariant_and_scale_is_bounded():
    rope = AdditiveAgeRoPEV22(8, 2, tuple(np.geomspace(7.0, 3650.0, 8)))
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    age = torch.tensor([[10.0, 100.0, 400.0, 900.0]])
    q1, k1 = rope(q, k, age)
    q2, k2 = rope(q, k, age + 10000.0)
    assert torch.allclose(q1 @ k1.transpose(-2, -1), q2 @ k2.transpose(-2, -1), atol=2e-4)
    with torch.no_grad():
        rope.log_scale_raw[0].fill_(-100.0)
        rope.log_scale_raw[1].fill_(100.0)
    scale = rope.scale_factor()
    assert torch.allclose(scale[0], torch.full_like(scale[0], 0.5))
    assert torch.allclose(scale[1], torch.full_like(scale[1], 2.0))


def test_pair_logit_selection_leaves_static_and_bos_pairs_on_base_logits():
    config = CARoPEConfig(
        block_size=4, vocab_size=32, n_layer=1, n_head=2, n_embd=8,
        use_age_encoding=True, use_age_rope=True, age_rope_variant="additive_v2_2",
        rope_wavelengths_days=(7.0, 70.0, 14.0, 140.0),
    )
    attention = CARoPECausalSelfAttention(config).eval()
    q, k, _ = attention._project(torch.randn(1, 4, 8))
    clinical = torch.tensor([[False, False, True, True]])
    base, rope, selected, pair_mask = attention.additive_pair_logits(
        q, k, torch.tensor([[50.0, 50.0, 100.0, 300.0]]), clinical
    )
    expanded = pair_mask.expand_as(selected)
    assert torch.equal(selected[~expanded], base[~expanded])
    assert torch.equal(selected[expanded], rope[expanded])
    assert not torch.allclose(base[expanded], rope[expanded])


def test_a0_has_no_rope_attention_path():
    attention = CARoPECausalSelfAttention(CARoPEConfig(
        block_size=4, vocab_size=16, n_layer=1, n_head=2, n_embd=8,
        use_age_encoding=True, use_age_rope=False,
    ))
    assert attention.rope is None
    assert attention.additive_rope_v2_2 is None


def test_evaluator_rows_mean_equals_frozen_next_event_loss_atol_1e_6():
    model = CARoPEHorizonMedTrajectory(CARoPEConfig(
        block_size=7, vocab_size=32, n_layer=1, n_head=2, n_embd=8,
        static_dim=0, num_tte_tasks=2, num_horizons=2,
        use_age_encoding=True, use_age_rope=False, use_relative_horizon_query=False,
        dropout=0.0,
    )).eval()
    x = torch.tensor([[20, 21, 22, 5, 6, 7, 8]])
    age = torch.tensor([[250.0, 250.0, 250.0, 100.0, 200.0, 300.0, 400.0]])
    targets = torch.tensor([[-1, -1, 5, 6, 7, 8, 9]])
    targets_age = torch.tensor([[-10000.0, -10000.0, 100.0, 200.0, 300.0, 400.0, 500.0]])
    static = torch.tensor([[True, True, False, False, False, False, False]])
    bos = torch.tensor([[False, False, True, False, False, False, False]])
    next_mask = torch.tensor([[False, False, True, True, True, True, True]])
    time_mask = torch.tensor([[False, False, False, True, True, True, True]])
    logits, parts, *_ = model(
        x, age, None, targets, targets_age, validation_loss_mode=True,
        static_token_mask=static, bos_token_mask=bos,
        next_event_mask=next_mask, time_loss_mask=time_mask,
    )
    attn_mask = model.build_track_r_attention_mask(x, age, targets_age, static)
    rows, time_valid = event_time_nll_rows(
        logits, x, age, targets, targets_age, attn_mask,
        ignore_tokens=model.config.ignore_tokens, t_min=model.config.t_min,
        mask_ties=model.config.mask_ties,
        next_event_mask=next_mask, time_loss_mask=time_mask,
    )
    assert torch.allclose(rows[time_valid].mean(), parts["loss_dt"], rtol=0.0, atol=1e-6)


def test_deterministic_right_window_preserves_original_ordinals_and_gzip(tmp_path):
    data = np.asarray([[0, age, token] for age, token in zip((10, 20, 30, 40, 50), (2, 3, 4, 5, 6))], dtype=np.uint32)
    batch = deterministic_patient_batch(
        data, np.asarray([[0, 5]]), np.asarray([[20, 21, 22, 23]]), np.asarray([25.0]), 0,
        dynamic_context_length=3, bos_token_id=24,
    )
    assert batch["context_window_start"] == 1
    assert batch["source_ordinals"].tolist() == [1, 2, 3]
    assert batch["target_ordinals"].tolist() == [2, 3, 4]
    assert batch["local_positions"].tolist() == [5, 6, 7]
    payload = [{"source": int(value)} for value in batch["source_ordinals"]]
    first = write_json_gzip(tmp_path / "rows.json.gz", payload)
    second = write_json_gzip(tmp_path / "rows-copy.json.gz", payload)
    assert first.read_bytes() == second.read_bytes()


def test_launcher_uses_matching_pretraining_last_for_risk_and_separate_last_for_evaluation():
    protocol = load_track_r_protocol(PROTOCOL)
    train = [" ".join(job) for job in training_jobs(PROTOCOL, protocol, "cpu")]
    evaluate = [" ".join(job) for job in evaluation_jobs(PROTOCOL, protocol, "cpu")]
    assert len(train) == 18
    assert len(evaluate) == 19
    assert all("best_" not in item for item in train + evaluate)
    normalized_train = [item.replace("\\", "/") for item in train]
    normalized_evaluate = [item.replace("\\", "/") for item in evaluate]
    risk = [item for item in normalized_train if "train_car_rope.py" in item]
    assert all("--init-from-ckpt" in item and "pretraining" in item and "checkpoints/last.pt" in item for item in risk)
    nll = [item for item in normalized_evaluate if "evaluate_track_r_v2_2_pretraining" in item]
    clinical = [item for item in normalized_evaluate if "semantic_delphi_ukb.evaluate_track_r " in item]
    final = [item for item in normalized_evaluate if "semantic_delphi_ukb.finalize_track_r_v2_2" in item]
    assert all("pretraining" in item and "checkpoints/last.pt" in item for item in nll)
    assert all("risk" in item and "checkpoints/last.pt" in item for item in clinical)
    assert len(final) == 1
    assert normalized_evaluate[-1] == final[0]


def test_launcher_pair_filter_keeps_each_pretraining_risk_dependency_together():
    protocol = load_track_r_protocol(PROTOCOL)
    pairs = parse_pairs(["42:A0", "44:A2-noAge"], protocol)
    jobs = [" ".join(job).replace("\\", "/") for job in training_jobs(PROTOCOL, protocol, "cuda", pairs)]
    assert len(jobs) == 4
    assert sum("seed42_additiverope1" in job and "/A0" in job for job in jobs) == 2
    assert sum("seed44_additiverope1" in job and "/A2-noAge" in job for job in jobs) == 2
    with pytest.raises(ValueError, match="unregistered"):
        parse_pairs(["45:A0"], protocol)
    with pytest.raises(ValueError, match="duplicate"):
        parse_pairs(["42:A0", "42:A0"], protocol)
