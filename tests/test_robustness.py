from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geoembeddings.baseline import export_statistical_baseline
from geoembeddings.config import load_config
from geoembeddings.robustness import deterministic_event_removal, export_robustness_views


def _events(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"user_id": [f"u{i % 4}" for i in range(n)],
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "service_id": [f"s{i % 3}" for i in range(n)], "value": range(n)})


def test_selection_is_deterministic_order_invariant_and_chronological() -> None:
    events = _events()
    left, left_mask = deterministic_event_removal(events, source_hash="a" * 64, seed=7, rate=.4)
    right, right_mask = deterministic_event_removal(events.sample(frac=1, random_state=4),
                                                     source_hash="a" * 64, seed=7, rate=.4)
    assert left.equals(right)
    assert left_mask.equals(right_mask)
    assert left.equals(left.sort_values(["user_id", "timestamp"]).reset_index(drop=True))


@pytest.mark.parametrize("rate", [-.1, 1.1, float("nan")])
def test_rate_validation(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        deterministic_event_removal(_events(), source_hash="x", seed=1, rate=rate)


def test_zero_full_duplicate_and_source_hash_edges() -> None:
    events = _events(200)
    zero, _ = deterministic_event_removal(events, source_hash="a", seed=3, rate=0)
    full, _ = deterministic_event_removal(events, source_hash="a", seed=3, rate=1)
    assert len(zero) == len(events) and full.empty
    _, a = deterministic_event_removal(events, source_hash="a", seed=3, rate=.5)
    _, b = deterministic_event_removal(events, source_hash="b", seed=3, rate=.5)
    assert not a["removed"].equals(b["removed"])
    with pytest.raises(ValueError, match="duplicate event keys"):
        deterministic_event_removal(pd.concat([events, events.iloc[[0]]]), source_hash="a", seed=3, rate=.5)


def test_sparse_histories_are_reported_unencodable_and_masks_match_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    run, experiment = root / "smoke/run", root / "smoke/experiment"
    config = load_config(root / "configs/embedding/single_vector.yaml")
    config["evaluation"]["event_removal"] = {"algorithm_version": "sha256-event-removal/1.0",
                                              "seed": 19, "rates": [0.0, 1.0]}
    baseline = export_robustness_views(run / "observed", experiment / "prepared", Path("unused"),
        tmp_path, config, kind="baseline")
    def fake_learned(observed_dir, prepared_dir, checkpoint_path, output_path, config, **kwargs):
        return export_statistical_baseline(observed_dir, prepared_dir, output_path, config, **kwargs)
    monkeypatch.setattr("geoembeddings.robustness.export_embeddings", fake_learned)
    learned = export_robustness_views(run / "observed", experiment / "prepared",
        Path("fake-checkpoint"), tmp_path, config, kind="learned")
    for a, b in zip(baseline["artifacts"], learned["artifacts"]):
        assert a["removed_events"] == b["removed_events"]
        assert a["encoded_keys"] == b["encoded_keys"]
    assert baseline["artifacts"][1]["path"] is None
    assert baseline["artifacts"][1]["unencodable_keys"]
    assert baseline["information_boundary"].endswith("observed/ only")
