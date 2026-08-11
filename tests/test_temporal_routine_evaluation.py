import numpy as np
import pandas as pd
import pytest
import inspect
from geoembeddings import baseline, export, model, prepare, training
from geoembeddings.cli import build_parser
import geoembeddings.temporal_routine_evaluation as temporal_evaluator

from geoembeddings.temporal_routine_evaluation import (
    cyclic_bin, deterministic_user_split, episode_duration_hours,
    periodic_retrieval, select_repeated_and_one_off,
)


def test_cyclic_boundaries_wrap_exact_period():
    assert cyclic_bin(0, [0, 6, 12, 24], 24) == 0
    assert cyclic_bin(24, [0, 6, 12, 24], 24) == 0
    assert cyclic_bin(-1, [0, 6, 12, 24], 24) == 2
    assert cyclic_bin(6, [0, 6, 12, 24], 24) == 1


def test_duration_validation():
    frame = pd.DataFrame({"start_time": pd.to_datetime(["2025-01-01T00:00Z"]),
                          "end_time": pd.to_datetime(["2025-01-01T02:30Z"])})
    assert episode_duration_hours(frame).tolist() == [2.5]
    frame["end_time"] = frame["start_time"]
    with pytest.raises(ValueError, match="positive"):
        episode_duration_hours(frame)


def test_deterministic_split_is_order_independent():
    users = [f"u{i}" for i in range(30)]
    first = dict(zip(users, deterministic_user_split(users, .7, 9)))
    second = dict(zip(reversed(users), deterministic_user_split(list(reversed(users)), .7, 9)))
    assert first == second


def test_periodic_retrieval_recovers_user_and_state():
    rows = pd.DataFrame({"embedding_index": range(4), "user_id": ["a", "a", "b", "b"],
                         "temporal_bin": ["morning", "morning", "evening", "evening"]})
    embeddings = np.asarray([[1, 0], [.9, .1], [0, 1], [.1, .9]])
    result = periodic_retrieval(rows, embeddings)
    assert result["user_top1"] == result["state_top1"] == 1


def test_repeated_routine_and_one_off_selection_and_empty_classes():
    episodes = pd.DataFrame({"user_id": ["u"] * 4, "episode_id": list("abcd"),
                             "primary_intent": ["routine", "routine", "travel", "travel"]})
    selected = select_repeated_and_one_off(episodes, 2)
    assert selected["routine_class"].tolist() == ["repeated_routine"] * 2
    assert select_repeated_and_one_off(episodes.iloc[2:], 2).empty


def test_periodic_retrieval_zero_coverage():
    result = periodic_retrieval(pd.DataFrame(columns=["embedding_index", "user_id", "temporal_bin"]), np.empty((0, 2)))
    assert result["status"] == "insufficient_rows" and result["queries"] == 0


def test_temporal_truth_join_is_evaluator_only_and_cli_is_wired():
    assert "episodes_truth.csv.gz" in inspect.getsource(temporal_evaluator.load_episode_evaluation_inputs)
    for module in (baseline, export, model, prepare, training):
        assert "episodes_truth.csv.gz" not in inspect.getsource(module)
    args = build_parser().parse_args(["evaluate", "--temporal-routine", "--kind", "baseline",
                                      "--run-dir", "run", "--experiment-dir", "experiment"])
    assert args.temporal_routine and args.kind == "baseline"
