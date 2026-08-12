from __future__ import annotations

import numpy as np
import pytest

from geoembeddings.online import AtomicOnlineState, OnlineRepresentation, canonical_hash


FIELDS = ("user_id", "timestamp", "value")


def _computer(frame):
    return {"combined": np.asarray([frame["value"].astype(float).sum()], dtype=np.float32)}


def _row(user="u1", timestamp="2026-01-01T00:00:00Z", value=1.0):
    return {"user_id": user, "timestamp": timestamp, "value": value}


def _state():
    return AtomicOnlineState(field_order=FIELDS, component_names=("combined",),
        compute=_computer, maximum_history=3, identity={"checkpoint": "fixed"})


def test_workload_hash_is_deterministic_and_percentile_method_is_stable():
    from geoembeddings.benchmark import latency_statistics
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert latency_statistics([4, 1, 3, 2])["p50_seconds"] == 2.5


def test_outputs_are_immutable_and_duplicates_are_noops():
    state = _state(); result = state.append([_row()])
    vector = result.representations[0].components[0][1]
    with pytest.raises(ValueError): vector[0] = 9
    duplicate = state.append([_row()])
    assert duplicate.accepted_events == 0 and duplicate.duplicate_events == 1


def test_invalid_batch_and_out_of_order_event_roll_back_all_users():
    state = _state(); state.append([_row()]); before = state.outputs["u1"].components[0][1].copy()
    with pytest.raises(ValueError, match="timestamp"):
        state.append([_row("u1", "bad", 2), _row("u2", "2026-01-02T00:00:00Z", 3)])
    assert "u2" not in state.outputs and np.array_equal(state.outputs["u1"].components[0][1], before)
    with pytest.raises(ValueError, match="out-of-order"):
        state.append([_row("u1", "2025-12-31T00:00:00Z", 4)])
    assert np.array_equal(state.outputs["u1"].components[0][1], before)


def test_correctness_mismatch_aborts_without_mutation():
    state = _state()
    def wrong(frame): return {"combined": _computer(frame)["combined"] + 1}
    with pytest.raises(ValueError, match="correctness mismatch"):
        state.append([_row()], oracle=wrong)
    assert state.outputs == {}


def test_component_schema_failure_is_atomic():
    state = _state()
    with pytest.raises(ValueError, match="component schema"):
        state.append([_row()], oracle=lambda frame: {"other": np.ones(1, dtype=np.float32)})
    assert state.outputs == {}
