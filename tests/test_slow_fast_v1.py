from __future__ import annotations

from copy import deepcopy

import torch

from geoembeddings.model import build_model


def _fixture() -> tuple[dict, dict]:
    fields = (
        "service_id", "action_type", "observation_mode", "object_category",
        "region_id", "geohash_5", "geohash_7",
    )
    vocabularies = {
        field: {"<PAD>": 0, "<UNK>": 1, f"{field}_value": 2}
        for field in fields
    }
    config = {
        "data": {"max_sequence_length": 8},
        "model": {
            "variant": "slow_fast_v1",
            "categorical_embedding_dim": 4,
            "event_dim": 8,
            "hidden_dim": 6,
            "user_embedding_dim": 5,
            "gru_layers": 1,
            "dropout": 0.0,
            "event_dropout": 0.0,
            "persistent_decay_horizon_hours": 168.0,
            "context_recent_history_events": 2,
            "loss_routing": {
                "persistent_future_category_histogram": "persistent",
                "next_time_bucket": "context",
                "next_elapsed_time_bucket": "context",
                "next_category": "combined",
            },
        },
        "targets": {
            "persistent_future_category_histogram": {"horizon_days": 7},
            "next_time_bucket": {"edges_hours": [0, 12, 24]},
            "next_elapsed_time_bucket": {"edges_hours": [0, 1, 24]},
        },
        "objectives": {
            "persistent_future_category_histogram": 0.5,
            "next_time_bucket": 0.25,
            "next_elapsed_time_bucket": 0.25,
            "next_category": 1.0,
        },
    }
    return config, vocabularies


def test_slow_fast_has_private_branches_and_normalized_lossless_fusion() -> None:
    config, vocabularies = _fixture()
    model = build_model(vocabularies, 3, config).eval()
    assert not hasattr(model, "gru")
    assert model.persistent_branch is not model.context_branch
    assert model.persistent_branch.gru is not model.context_branch.gru
    assert model.persistent_decay_horizon_hours == 168.0
    assert not any(name.startswith("fusion_gate") for name, _ in model.named_parameters())

    categorical = torch.full((3, 5, len(vocabularies)), 2, dtype=torch.long)
    continuous = torch.randn(3, 5, 3)
    elapsed = torch.tensor([[0.0, 4.0, 8.0, 24.0, 48.0]] * 3)
    with torch.no_grad():
        output = model.encode_components(categorical, continuous, torch.tensor([5, 4, 2]), elapsed_hours=elapsed)
    assert output.persistent.shape == output.context.shape == (3, 5)
    assert output.combined.shape == (3, 10)
    torch.testing.assert_close(
        output.combined,
        torch.cat((torch.nn.functional.normalize(output.persistent, dim=1),
                   torch.nn.functional.normalize(output.context, dim=1)), dim=1),
    )
    assert model.heads["persistent_future_category_histogram"].in_features == 5
    assert model.heads["next_time_bucket"].in_features == 5
    assert model.heads["next_category"].in_features == 10


def test_slow_fast_and_control_are_capacity_matched() -> None:
    config, vocabularies = _fixture()
    control_config = deepcopy(config)
    control_config["model"] = {
        **config["model"],
        "variant": "slow_fast_capacity_matched_single",
        "matched_output_dim": 5,
        "capacity_search_max_hidden_dim": 64,
    }
    candidate = build_model(vocabularies, 3, config)
    control = build_model(vocabularies, 3, control_config)
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    control_count = sum(parameter.numel() for parameter in control.parameters())
    assert abs(candidate_count - control_count) / candidate_count < 0.05
    assert control.capacity_match["target_model_variant"] == "slow_fast_v1"
