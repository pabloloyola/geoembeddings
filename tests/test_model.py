from __future__ import annotations

import pytest
import torch

from geoembeddings.model import build_model
from geoembeddings.training import _validate_batch_values


def _model_fixture():
    vocabularies = {
        "service_id": {"<PAD>": 0, "<UNK>": 1, "location": 2},
        "action_type": {"<PAD>": 0, "<UNK>": 1, "ping": 2},
    }
    config = {
        "model": {
            "categorical_embedding_dim": 4,
            "event_dim": 8,
            "hidden_dim": 6,
            "user_embedding_dim": 5,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {"next_service": 1.0, "next_action": 1.0},
    }
    return config, vocabularies


def test_single_vector_shapes() -> None:
    vocabularies = {
        "service_id": {"<PAD>": 0, "<UNK>": 1, "location": 2},
        "action_type": {"<PAD>": 0, "<UNK>": 1, "ping": 2},
        "object_category": {"<PAD>": 0, "<UNK>": 1, "unknown": 2},
        "region_id": {"<PAD>": 0, "<UNK>": 1, "nerima": 2},
        "geohash_5": {"<PAD>": 0, "<UNK>": 1, "xn76g": 2},
        "geohash_7": {"<PAD>": 0, "<UNK>": 1, "xn76gur": 2},
    }
    config = {
        "model": {
            "categorical_embedding_dim": 8,
            "event_dim": 16,
            "hidden_dim": 12,
            "user_embedding_dim": 10,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {
            "next_service": 1.0,
            "next_action": 1.0,
            "next_category": 1.0,
            "next_region": 1.0,
            "next_geohash_5": 1.0,
            "next_geohash_7": 1.0,
        },
    }
    model = build_model(vocabularies, continuous_dim=8, config=config)
    categorical = torch.full((3, 5, len(vocabularies)), 2, dtype=torch.long)
    continuous = torch.zeros((3, 5, 8))
    lengths = torch.tensor([5, 4, 2])
    embedding, logits = model(categorical, continuous, lengths)
    assert embedding.shape == (3, 10)
    assert logits["next_region"].shape == (3, 3)


def test_padding_after_valid_history_does_not_change_embedding() -> None:
    vocabularies = {
        "service_id": {"<PAD>": 0, "<UNK>": 1, "location": 2},
        "action_type": {"<PAD>": 0, "<UNK>": 1, "ping": 2},
        "object_category": {"<PAD>": 0, "<UNK>": 1, "unknown": 2},
        "region_id": {"<PAD>": 0, "<UNK>": 1, "nerima": 2},
        "geohash_5": {"<PAD>": 0, "<UNK>": 1, "xn76g": 2},
        "geohash_7": {"<PAD>": 0, "<UNK>": 1, "xn76gur": 2},
    }
    config = {
        "model": {
            "categorical_embedding_dim": 8,
            "event_dim": 16,
            "hidden_dim": 12,
            "user_embedding_dim": 10,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {"next_region": 1.0},
    }
    model = build_model(vocabularies, continuous_dim=8, config=config).eval()
    short_categorical = torch.full((1, 2, len(vocabularies)), 2, dtype=torch.long)
    short_continuous = torch.zeros((1, 2, 8))
    padded_categorical = torch.nn.functional.pad(short_categorical, (0, 0, 0, 3))
    padded_continuous = torch.nn.functional.pad(short_continuous, (0, 0, 0, 3))

    with torch.no_grad():
        short = model.encode(short_categorical, short_continuous, torch.tensor([2]))
        padded = model.encode(padded_categorical, padded_continuous, torch.tensor([2]))

    torch.testing.assert_close(short, padded)


def test_zero_length_history_fails_before_gru() -> None:
    vocabularies = {
        "service_id": {"<PAD>": 0, "<UNK>": 1, "location": 2},
    }
    config = {
        "model": {
            "categorical_embedding_dim": 8,
            "event_dim": 16,
            "hidden_dim": 12,
            "user_embedding_dim": 10,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {},
    }
    model = build_model(vocabularies, continuous_dim=8, config=config)
    categorical = torch.full((1, 2, 1), 2, dtype=torch.long)
    continuous = torch.zeros((1, 2, 8))

    with pytest.raises(ValueError, match="Invalid sequence lengths"):
        model.encode(categorical, continuous, torch.tensor([0]))


def test_last_valid_state_selection_backpropagates() -> None:
    vocabularies = {
        "service_id": {"<PAD>": 0, "<UNK>": 1, "location": 2},
    }
    config = {
        "model": {
            "categorical_embedding_dim": 8,
            "event_dim": 16,
            "hidden_dim": 12,
            "user_embedding_dim": 10,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {"next_service": 1.0},
    }
    model = build_model(vocabularies, continuous_dim=8, config=config)
    categorical = torch.full((3, 5, 1), 2, dtype=torch.long)
    continuous = torch.randn((3, 5, 8), requires_grad=True)
    lengths = torch.tensor([5, 3, 2])

    embedding = model.encode(categorical, continuous, lengths)
    embedding.square().mean().backward()

    assert continuous.grad is not None
    assert torch.isfinite(continuous.grad).all()


def test_categorical_tensor_order_is_explicit_not_json_key_order() -> None:
    # write_json(sort_keys=True) serializes these keys as object_category,
    # observation_mode. The event tensor contract intentionally uses the
    # opposite order. This reproduces the v0.3.2 failure where category IDs
    # were interpreted as observation-mode IDs.
    vocabularies = {
        "object_category": {
            "<PAD>": 0,
            "<UNK>": 1,
            **{f"category_{index}": index + 2 for index in range(14)},
        },
        "observation_mode": {
            "<PAD>": 0,
            "<UNK>": 1,
            "passive": 2,
            "user_triggered": 3,
            "system_generated": 4,
        },
    }
    categorical_fields = ["observation_mode", "object_category"]
    config = {
        "model": {
            "categorical_embedding_dim": 8,
            "event_dim": 16,
            "hidden_dim": 12,
            "user_embedding_dim": 10,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {},
    }
    model = build_model(
        vocabularies,
        continuous_dim=1,
        config=config,
        categorical_fields=categorical_fields,
    )
    categorical = torch.tensor([[[4, 15], [3, 14]]], dtype=torch.long)
    continuous = torch.zeros((1, 2, 1))
    lengths = torch.tensor([2])
    batch = {
        "user_id": ["user_000184"],
        "categorical": categorical,
        "continuous": continuous,
        "lengths": lengths,
        "early_categorical": categorical[:, :1],
        "early_continuous": continuous[:, :1],
        "early_lengths": torch.tensor([1]),
        "late_categorical": categorical[:, 1:],
        "late_continuous": continuous[:, 1:],
        "late_lengths": torch.tensor([1]),
        "targets": {},
    }

    _validate_batch_values(batch, model, batch_number=1)
    embedding = model.encode(categorical, continuous, lengths)

    assert model.categorical_fields == categorical_fields
    assert embedding.shape == (1, 10)


def test_model_rejects_incomplete_explicit_categorical_schema() -> None:
    vocabularies = {
        "observation_mode": {"<PAD>": 0, "<UNK>": 1},
        "object_category": {"<PAD>": 0, "<UNK>": 1},
    }
    config = {
        "model": {
            "categorical_embedding_dim": 8,
            "event_dim": 16,
            "hidden_dim": 12,
            "user_embedding_dim": 10,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
        },
        "objectives": {},
    }

    with pytest.raises(ValueError, match="Categorical field/vocabulary mismatch"):
        build_model(
            vocabularies,
            continuous_dim=1,
            config=config,
            categorical_fields=["observation_mode"],
        )


def test_single_vector_component_adapter_shapes_finiteness_gradients_and_device():
    from geoembeddings.model import EncoderOutput, SingleVectorOutputAdapter

    vector = torch.randn(3, 7, requires_grad=True)
    output = SingleVectorOutputAdapter()(vector)
    assert isinstance(output, EncoderOutput)
    assert output.persistent.shape == output.context.shape == output.combined.shape == (3, 7)
    assert torch.isfinite(output.persistent).all()
    assert torch.isfinite(output.context).all()
    assert torch.isfinite(output.combined).all()
    assert torch.equal(output.persistent, vector)
    assert torch.equal(output.combined, vector)
    assert torch.count_nonzero(output.context) == 0

    (output.persistent.sum() + output.combined.sum()).backward()
    assert torch.equal(vector.grad, torch.full_like(vector, 2.0))

    moved = output.to(torch.device("cpu"), dtype=torch.float64)
    assert moved.persistent.device.type == "cpu"
    assert moved.context.dtype == moved.combined.dtype == torch.float64


def test_single_vector_encoder_exposes_named_components_without_changing_legacy_vector():
    config, vocabularies = _model_fixture()
    model = build_model(vocabularies, continuous_dim=8, config=config).eval()
    categorical = torch.randint(0, 3, (2, 5, len(vocabularies)))
    continuous = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 3])

    legacy = model.encode(categorical, continuous, lengths)
    components = model.encode_components(categorical, continuous, lengths)
    assert torch.equal(components.persistent, legacy)
    assert torch.equal(components.combined, legacy)
    assert torch.count_nonzero(components.context) == 0


def test_model_registry_defaults_to_legacy_and_validates_unknown_variant():
    from geoembeddings.model import SingleVectorEncoder, configured_model_variant

    config, vocabularies = _model_fixture()
    assert configured_model_variant(config) == "single_vector"
    assert isinstance(build_model(vocabularies, 8, config), SingleVectorEncoder)

    config["model"]["variant"] = "not_registered"
    with pytest.raises(ValueError, match="Unknown model variant.*available variants"):
        build_model(vocabularies, 8, config)


def test_training_rejects_unknown_model_before_creating_output(tmp_path):
    from geoembeddings.training import train_model

    config, _ = _model_fixture()
    config["model"]["variant"] = "truth_aware_model"
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Unknown model variant"):
        train_model(tmp_path / "observed", tmp_path / "prepared", output, config)
    assert not output.exists()


@pytest.mark.parametrize("variant", ["factorized_pc", "persistent_only", "context_only"])
def test_factorized_variants_are_finite_mask_safe_and_backpropagate(variant):
    config, vocabularies = _model_fixture()
    config["model"].update({
        "variant": variant, "recent_history_events": 2,
        "persistent_update_rate": 0.1,
        "loss_routing": {"next_service": "context", "next_action": "combined"},
        "consistency_route": "persistent",
    })
    model = build_model(vocabularies, 8, config)
    categorical = torch.randint(0, 3, (3, 5, 2))
    continuous = torch.randn(3, 5, 8, requires_grad=True)
    lengths = torch.tensor([5, 3, 2])  # deliberately retained as CPU metadata
    output = model.encode_components(categorical, continuous, lengths)
    assert output.persistent.shape == output.context.shape == output.combined.shape == (3, 5)
    assert all(torch.isfinite(value).all() for value in
               (output.persistent, output.context, output.combined))
    output.combined.square().mean().backward()
    assert continuous.grad is not None and torch.isfinite(continuous.grad).all()
    if variant == "persistent_only":
        assert torch.count_nonzero(output.context) == 0
    if variant == "context_only":
        assert torch.count_nonzero(output.persistent) == 0


def test_factorized_loss_routing_uses_named_branch():
    config, vocabularies = _model_fixture()
    config["model"].update({"variant": "factorized_pc", "recent_history_events": 2,
                            "persistent_update_rate": .1,
                            "loss_routing": {"next_service": "context",
                                             "next_action": "persistent"}})
    model = build_model(vocabularies, 8, config).eval()
    assert model.loss_routes == {"next_service": "context", "next_action": "persistent"}
    categorical = torch.randint(0, 3, (2, 4, 2)); continuous = torch.randn(2, 4, 8)
    embedding, logits = model(categorical, continuous, torch.tensor([4, 2]))
    assert embedding.shape == (2, 5)
    assert set(logits) == {"next_service", "next_action"}


def _two_timescale_fixture():
    config, vocabularies = _model_fixture()
    config["model"].update({
        "variant": "two_timescale_pc",
        "persistent_half_life_events": 16.0,
        "context_half_life_events": 2.0,
        "ablation": "fusion",
        "loss_routing": {
            "next_service": "context",
            "next_action": "persistent",
        },
        "consistency_route": "persistent",
    })
    return config, vocabularies


def test_two_timescale_encoder_is_finite_padding_safe_and_backpropagates():
    config, vocabularies = _two_timescale_fixture()
    model = build_model(vocabularies, 8, config).eval()
    categorical = torch.randint(0, 3, (3, 6, 2))
    continuous = torch.randn(3, 6, 8, requires_grad=True)
    lengths = torch.tensor([6, 4, 2])

    output = model.encode_components(categorical, continuous, lengths)

    assert output.persistent.shape == output.context.shape == output.combined.shape == (3, 5)
    assert all(torch.isfinite(value).all() for value in (
        output.persistent, output.context, output.combined
    ))
    output.combined.square().mean().backward()
    assert continuous.grad is not None and torch.isfinite(continuous.grad).all()

    with torch.no_grad():
        short = model.encode_components(categorical[:1, :4], continuous[:1, :4], torch.tensor([4]))
        padded = model.encode_components(categorical[:1], continuous[:1], torch.tensor([4]))
    for name in ("persistent", "context", "combined"):
        torch.testing.assert_close(getattr(short, name), getattr(padded, name))


def test_two_timescale_pooling_declares_distinct_slow_and_fast_recency():
    from geoembeddings.model import TwoTimescalePCEncoder

    lengths = torch.tensor([6, 3])
    slow = TwoTimescalePCEncoder._timescale_weights(
        lengths, 6, 16.0, device=torch.device("cpu"), dtype=torch.float32
    )
    fast = TwoTimescalePCEncoder._timescale_weights(
        lengths, 6, 2.0, device=torch.device("cpu"), dtype=torch.float32
    )

    torch.testing.assert_close(slow.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(fast.sum(dim=1), torch.ones(2))
    assert torch.count_nonzero(slow[1, 3:]) == 0
    assert torch.count_nonzero(fast[1, 3:]) == 0
    assert fast[0, -1] > slow[0, -1]
    assert fast[0, 0] < slow[0, 0]


def test_two_timescale_rejects_inverted_half_lives_and_matches_capacity():
    config, vocabularies = _two_timescale_fixture()
    config["model"]["persistent_half_life_events"] = 1.0
    with pytest.raises(ValueError, match="persistent_half_life_events must be greater"):
        build_model(vocabularies, 8, config)

    config["model"].update({
        "variant": "two_timescale_capacity_matched_single",
        "persistent_half_life_events": 16.0,
        "matched_output_dim": 5,
        "capacity_search_max_hidden_dim": 64,
    })
    control = build_model(vocabularies, 8, config)
    assert control.capacity_match["target_model_variant"] == "two_timescale_pc"
    assert control.capacity_match["relative_error"] < 0.1


def _causal_transformer_fixture():
    config, vocabularies = _two_timescale_fixture()
    config["data"] = {"max_sequence_length": 8}
    config["model"].update({
        "variant": "causal_transformer_pc",
        "transformer_heads": 2,
        "transformer_layers": 1,
        "transformer_feedforward_dim": 16,
        "attention_half_life_events": 4.0,
    })
    return config, vocabularies


def test_causal_transformer_is_finite_padding_safe_and_backpropagates():
    config, vocabularies = _causal_transformer_fixture()
    model = build_model(vocabularies, 8, config).eval()
    categorical = torch.randint(0, 3, (3, 6, 2))
    continuous = torch.randn(3, 6, 8, requires_grad=True)
    lengths = torch.tensor([6, 4, 2])

    output = model.encode_components(categorical, continuous, lengths)

    assert output.persistent.shape == output.context.shape == output.combined.shape == (3, 5)
    assert all(torch.isfinite(value).all() for value in (
        output.persistent, output.context, output.combined
    ))
    output.combined.square().mean().backward()
    assert continuous.grad is not None and torch.isfinite(continuous.grad).all()

    with torch.no_grad():
        short = model.encode_components(categorical[:1, :4], continuous[:1, :4], torch.tensor([4]))
        padded = model.encode_components(categorical[:1], continuous[:1], torch.tensor([4]))
    for name in ("persistent", "context", "combined"):
        torch.testing.assert_close(getattr(short, name), getattr(padded, name))


def test_causal_transformer_does_not_leak_future_events_into_earlier_states():
    config, vocabularies = _causal_transformer_fixture()
    model = build_model(vocabularies, 8, config).eval()
    categorical = torch.randint(0, 3, (1, 5, 2))
    continuous = torch.randn(1, 5, 8)
    changed = continuous.clone()
    changed[:, -1] = changed[:, -1] + 100.0

    with torch.no_grad():
        original_event = model._event_tensor(categorical, continuous, augment=False)
        changed_event = model._event_tensor(categorical, changed, augment=False)
        original = model._transformer_sequence(original_event, torch.tensor([5]))
        perturbed = model._transformer_sequence(changed_event, torch.tensor([5]))

    torch.testing.assert_close(original[:, :-1], perturbed[:, :-1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original[:, -1], perturbed[:, -1])


def test_causal_transformer_validates_configuration_and_matches_capacity():
    config, vocabularies = _causal_transformer_fixture()
    config["model"]["transformer_heads"] = 3
    with pytest.raises(ValueError, match="event_dim must be divisible"):
        build_model(vocabularies, 8, config)

    config["model"].update({
        "variant": "causal_transformer_capacity_matched_single",
        "transformer_heads": 2,
        "matched_output_dim": 5,
        "capacity_search_max_hidden_dim": 128,
    })
    control = build_model(vocabularies, 8, config)
    assert control.capacity_match["target_model_variant"] == "causal_transformer_pc"
    assert control.capacity_match["relative_error"] < 0.1
