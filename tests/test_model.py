from __future__ import annotations

import pytest
import torch

from geoembeddings.model import build_model
from geoembeddings.training import _validate_batch_values


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
