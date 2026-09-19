import pytest
import torch

from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory, MultiScaleAgeRoPE
from semantic_delphi_ukb.train_car_rope import compose_training_loss, load_frozen_a0_base


def test_relative_rope_is_translation_invariant_in_age_origin():
    torch.manual_seed(7)
    rope = MultiScaleAgeRoPE(head_dim=8, n_heads=2)
    q = torch.randn(2, 2, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    age = torch.arange(5, dtype=torch.float32).repeat(2, 1) * 365.25 + 50 * 365.25
    q1, k1 = rope(q, k, age)
    q2, k2 = rope(q, k, age + 1000 * 365.25)
    scores_1 = q1 @ k1.transpose(-2, -1)
    scores_2 = q2 @ k2.transpose(-2, -1)
    assert torch.allclose(scores_1, scores_2, atol=2e-4, rtol=2e-4)


def test_car_rope_forward_has_time_gap_and_monotonic_outputs():
    torch.manual_seed(11)
    config = CARoPEConfig(
        block_size=12,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        static_dim=3,
        static_hidden_dim=32,
        num_tte_tasks=3,
        num_horizons=2,
    )
    model = CARoPEHorizonMedTrajectory(config)
    idx = torch.randint(2, 64, (4, 12))
    age = torch.sort(torch.rand(4, 12) * 30000, dim=1).values
    targets = torch.randint(2, 64, (4, 12))
    target_age = age + 30.0
    static = torch.randn(4, 3)
    horizon_event = torch.randint(0, 2, risk_shape := (4, 12, 2, 3), dtype=torch.float32)
    horizon_mask = torch.ones(risk_shape)
    logits, parts, attention, tte, risk = model(
        idx, age, static, targets, target_age,
        horizon_event=horizon_event, horizon_mask=horizon_mask,
    )
    assert logits.shape == (4, 12, 64)
    assert attention.shape[:3] == (2, 4, 4)
    assert tte.shape == (4, 12, 3)
    assert risk.shape == (4, 12, 2, 3)
    assert parts is not None and torch.isfinite(parts["loss"]) and "loss_gap" in parts
    assert torch.all(risk[:, :, 1:] >= risk[:, :, :-1])
    parts["loss"].backward()
    assert model.horizon_query.grad is not None


def test_car_rope_forward_applies_static_temporal_visibility_and_explicit_loss_masks():
    torch.manual_seed(13)
    config = CARoPEConfig(
        block_size=6,
        vocab_size=32,
        n_layer=1,
        n_head=4,
        n_embd=32,
        static_dim=0,
        num_tte_tasks=2,
        num_horizons=2,
        dropout=0.0,
    )
    model = CARoPEHorizonMedTrajectory(config).eval()
    idx = torch.tensor([[20, 21, 22, 2, 3, 4]])
    age = torch.tensor([[100.0, 100.0, 100.0, 50.0, 100.0, 150.0]])
    targets = torch.tensor([[-1, -1, -1, 3, 4, 5]])
    target_age = torch.tensor([[-10000.0, -10000.0, -10000.0, 50.0, 150.0, 200.0]])
    static_mask = torch.tensor([[True, True, True, False, False, False]])
    next_mask = torch.tensor([[False, False, False, True, False, True]])
    time_mask = torch.tensor([[False, False, False, False, False, True]])

    _, parts, attention, _, _ = model(
        idx,
        age,
        targets=targets,
        targets_age=target_age,
        static_token_mask=static_mask,
        next_event_mask=next_mask,
        time_loss_mask=time_mask,
    )

    # The dynamic query at age 50 cannot read recruitment-time static tokens at age 100.
    assert torch.allclose(attention[0, 0, :, 3, :3], torch.zeros(4, 3))
    assert parts is not None and torch.isfinite(parts["loss"])
    assert "loss_gap" in parts


def test_car_rope_model_is_invariant_to_age_origin_shift():
    torch.manual_seed(19)
    config = CARoPEConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        static_dim=1,
        static_hidden_dim=32,
        num_tte_tasks=2,
        num_horizons=2,
        dropout=0.0,
        static_dropout=0.0,
    )
    model = CARoPEHorizonMedTrajectory(config).eval()
    idx = torch.randint(2, 32, (3, 8))
    age = torch.sort(torch.rand(3, 8) * 20000, dim=1).values
    static = torch.randn(3, 1)
    risk_1 = model(idx, age, static)[4]
    risk_2 = model(idx, age + 20 * 365.25, static)[4]
    assert torch.allclose(risk_1, risk_2, atol=2e-4, rtol=2e-4)


def test_post_transformer_static_residual_does_not_change_rope_attention():
    torch.manual_seed(29)
    config = CARoPEConfig(
        block_size=6,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        static_dim=4,
        static_hidden_dim=32,
        static_fusion_stage="post_transformer",
        num_tte_tasks=2,
        num_horizons=2,
        dropout=0.0,
        static_dropout=0.0,
    )
    model = CARoPEHorizonMedTrajectory(config).eval()
    idx = torch.randint(2, 32, (2, 6))
    age = torch.sort(torch.rand(2, 6) * 20000, dim=1).values
    available = torch.tensor([[False, False, True, True, True, True], [False, True, True, True, True, True]])
    output_a = model(idx, age, torch.zeros(2, 4), static_feature_mask=available)
    output_b = model(idx, age, torch.ones(2, 4), static_feature_mask=available)

    assert torch.allclose(output_a[2], output_b[2])
    assert torch.allclose(output_a[4][~available], output_b[4][~available])
    assert not torch.allclose(output_a[4][available], output_b[4][available])


def test_gap_loss_weight_is_applied_once():
    parts = {
        "loss_ce": torch.tensor(0.0),
        "loss_dt": torch.tensor(0.0),
        "loss_gap": torch.tensor(1.0),
        "loss_horizon": torch.tensor(0.0),
    }
    loss = compose_training_loss(parts, next_event_weight=0.2, horizon_weight=1.0, gap_weight=0.2)
    assert loss.item() == pytest.approx(0.2)


@pytest.mark.parametrize("n_embd,n_head", [(120, 12), (128, 8)])
def test_car_rope_projects_fixed_semantic_embeddings_to_larger_trunk(n_embd, n_head):
    torch.manual_seed(23)
    semantic = torch.randn(64, 64)
    config = CARoPEConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=n_head,
        n_embd=n_embd,
        semantic_embedding_dim=64,
        static_dim=3,
        static_hidden_dim=n_embd,
        num_tte_tasks=2,
        num_horizons=3,
        horizon_years=(1.0, 5.0, 10.0),
        use_relative_horizon_query=False,
    )
    model = CARoPEHorizonMedTrajectory(config, pretrained_token_embeddings=semantic)
    idx = torch.randint(2, 64, (2, 8))
    age = torch.sort(torch.rand(2, 8) * 30000, dim=1).values
    logits, _, attention, _, risk = model(idx, age, torch.randn(2, 3))
    assert model.transformer.wte.weight.shape == (64, 64)
    assert model.token_input_proj.weight.shape == (n_embd, 64)
    assert logits.shape == (2, 8, 64)
    assert attention.shape == (2, 2, n_head, 8, 8)
    assert risk.shape == (2, 8, 3, 2)


def _residual_test_config(mode="learned", fixed_alpha=1.0):
    return CARoPEConfig(
        block_size=7,
        vocab_size=40,
        n_layer=2,
        n_head=4,
        n_embd=32,
        static_dim=0,
        num_tte_tasks=2,
        num_horizons=2,
        use_age_encoding=True,
        use_age_rope=False,
        use_relative_horizon_query=False,
        residual_rope_mode=mode,
        residual_rope_initial_alpha=0.01,
        residual_rope_fixed_alpha=fixed_alpha,
        dropout=0.0,
    )


def _residual_test_batch():
    idx = torch.tensor([[30, 31, 32, 33, 34, 4, 5]])
    age = torch.tensor([[1000.0, 1000.0, 1000.0, 1000.0, 800.0, 800.0, 1200.0]])
    static_mask = torch.tensor([[True, True, True, True, False, False, False]])
    bos_mask = torch.tensor([[False, False, False, False, True, False, False]])
    return idx, age, static_mask, bos_mask


def test_residual_rope_alpha_zero_exactly_matches_a0_base():
    torch.manual_seed(41)
    base_config = _residual_test_config(mode="none")
    base = CARoPEHorizonMedTrajectory(base_config).eval()
    residual = CARoPEHorizonMedTrajectory(_residual_test_config()).eval()
    missing, unexpected = residual.load_state_dict(base.state_dict(), strict=False)
    assert missing and all("residual_rope" in name or "residual_gate_logit" in name for name in missing)
    assert not unexpected
    residual.set_residual_rope_alpha_override(0.0)
    idx, age, static_mask, bos_mask = _residual_test_batch()

    base_output = base(idx, age, static_token_mask=static_mask)
    residual_output = residual(
        idx,
        age,
        static_token_mask=static_mask,
        bos_token_mask=bos_mask,
    )
    for base_value, residual_value in zip(base_output, residual_output):
        if torch.is_tensor(base_value):
            assert torch.equal(base_value, residual_value)


def test_residual_rope_excludes_static_and_bos_queries_and_has_no_inner_gate():
    torch.manual_seed(43)
    model = CARoPEHorizonMedTrajectory(_residual_test_config()).eval()
    idx, age, static_mask, bos_mask = _residual_test_batch()
    model.set_residual_rope_alpha_override(0.0)
    attention_a0 = model(
        idx, age, static_token_mask=static_mask, bos_token_mask=bos_mask
    )[2]
    model.set_residual_rope_alpha_override(1.0)
    attention_rope = model(
        idx, age, static_token_mask=static_mask, bos_token_mask=bos_mask
    )[2]

    # Static prefix rows and dynamic:BOS are outside the residual pair mask.
    assert torch.equal(attention_a0[:, :, :, :5], attention_rope[:, :, :, :5])
    assert not torch.equal(attention_a0[:, :, :, 5:], attention_rope[:, :, :, 5:])
    diagnostics = model.residual_rope_diagnostics()
    assert diagnostics["inner_gate"] is False
    assert all(block.attn.residual_rope.gate_logit is None for block in model.transformer.h)


def test_frozen_a0_upgrade_only_trains_new_residual_parameters(tmp_path):
    torch.manual_seed(47)
    base = CARoPEHorizonMedTrajectory(_residual_test_config(mode="none"))
    checkpoint = tmp_path / "a0.pt"
    torch.save({"model": base.state_dict(), "model_args": base.config.__dict__.copy()}, checkpoint)
    residual = CARoPEHorizonMedTrajectory(_residual_test_config())

    load_frozen_a0_base(residual, checkpoint, "cpu")
    trainable = {name for name, parameter in residual.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("residual_rope" in name or name.endswith("residual_gate_logit") for name in trainable)
    diagnostics = residual.residual_rope_diagnostics()
    assert all(alpha == pytest.approx(0.01) for layer in diagnostics["layers"] for alpha in layer["alpha_by_head"])
