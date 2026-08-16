from __future__ import annotations

import copy

import pandas as pd
import pytest
import torch

from geoembeddings.data import EventWindowDataset, future_profile_objectives
from geoembeddings.model import build_model


def _config(variant: str = "multihorizon_profile_gru") -> dict:
    return {
        "data": {"categorical_fields": ["service_id"], "min_history_events": 1,
                 "max_sequence_length": 8},
        "model": {"variant": variant, "categorical_embedding_dim": 4, "event_dim": 8,
                  "hidden_dim": 8, "user_embedding_dim": 8, "gru_layers": 1,
                  "dropout": 0.0, "event_dropout": 0.0,
                  "persistent_half_life_events": 4.0, "context_half_life_events": 2.0,
                  "ablation": "fusion", "loss_routing": {"next_service": "context"}},
        "objectives": {"next_service": 1.0, "cross_window_consistency": 0.0},
        "representation_objectives": {
            "schema_version": "geoembeddings-multihorizon-profile/1.0",
            "horizons": {"short": {"events": 2, "route": "context"},
                         "long": {"events": 3, "route": "persistent"}},
            "fields": {"service_id": 1.0, "time_bin_4h": 0.5, "day_type": 0.5},
        },
    }


def _vocabularies() -> dict[str, dict[str, int]]:
    return {"service_id": {"<UNK>": 0, "location": 1, "commerce": 2}}


def test_profile_heads_receive_gradients_but_detached_control_does_not_shape_encoder() -> None:
    categorical = torch.tensor([[[1], [2]]])
    continuous = torch.zeros((1, 2, 1))
    lengths = torch.tensor([2])
    for variant, expects_encoder_gradient in (
        ("multihorizon_profile_gru", True),
        ("multihorizon_profile_detached_control", False),
    ):
        model = build_model(_vocabularies(), 1, _config(variant), ["service_id"])
        _, _, profile_logits = model.forward_with_profiles(categorical, continuous, lengths)
        loss = sum(value.sum() for value in profile_logits.values())
        loss.backward()
        assert all(head.weight.grad is not None for head in model.profile_heads.values())
        assert (model.gru.weight_ih_l0.grad is not None) is expects_encoder_gradient


def test_same_timestamp_events_are_never_used_as_each_others_history() -> None:
    dataset = EventWindowDataset.__new__(EventWindowDataset)
    dataset.metadata = {"train_end": "2026-01-01T12:00:00+00:00",
                        "validation_end": "2026-01-02T12:00:00+00:00"}
    dataset.user_roles = None
    dataset.profile_objectives = future_profile_objectives(_config(), _vocabularies())
    events = pd.DataFrame({
        "user_id": ["u", "u", "u", "u"],
        "timestamp": pd.to_datetime(["2026-01-01T09:00:00Z", "2026-01-01T10:00:00Z",
                                      "2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"]),
    })
    references = dataset._make_references(events, "train", _config()["data"])
    simultaneous = [reference for reference in references if reference.target_index in {1, 2}]
    assert len(simultaneous) == 2
    assert all(reference.context_indices == (0,) for reference in simultaneous)
    assert all(1 not in reference.future_indices and 2 not in reference.future_indices
               for reference in references if reference.target_index == 0)


def test_profile_targets_are_split_local_and_strictly_future() -> None:
    dataset = EventWindowDataset.__new__(EventWindowDataset)
    dataset.metadata = {"train_end": "2026-01-01T12:00:00+00:00",
                        "validation_end": "2026-01-01T23:00:00+00:00"}
    dataset.user_roles = None
    dataset.profile_objectives = future_profile_objectives(_config(), _vocabularies())
    events = pd.DataFrame({
        "user_id": ["u"] * 5,
        "timestamp": pd.to_datetime(["2026-01-01T09:00:00Z", "2026-01-01T10:00:00Z",
                                      "2026-01-01T11:00:00Z", "2026-01-01T13:00:00Z",
                                      "2026-01-02T09:00:00Z"]),
        "service_id": ["location", "commerce", "location", "commerce", "location"],
    })
    references = dataset._make_references(events, "train", _config()["data"])
    assert references
    target = references[0]
    assert target.target_index == 1
    assert target.future_indices == (2,)


def test_unknown_profile_field_is_rejected() -> None:
    config = copy.deepcopy(_config())
    config["representation_objectives"]["fields"] = {"truth_intent": 1.0}
    with pytest.raises(ValueError, match="Unknown representation objective field"):
        future_profile_objectives(config, _vocabularies())
