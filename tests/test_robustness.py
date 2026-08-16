from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from geoembeddings.baseline import export_dense_statistical_baseline, export_statistical_baseline
from geoembeddings.config import load_config
from geoembeddings.robustness import deterministic_event_removal, export_robustness_views, perturb_view
from geoembeddings.robustness import _specs
from geoembeddings.representation_schema import EXPORT_SCHEMA_VERSION, load_embedding_export
from geoembeddings.io import sha256_file


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


@pytest.mark.parametrize("view,parameters", [
    ("gps", {"sigma_m": 50}), ("timestamp", {"max_jitter_seconds": 600}),
    ("leave-one-service-out", {"service_id": "s1"}),
    ("recent-truncation", {"remove_recent_events": 2}),
])
def test_views_are_deterministic_and_row_order_independent(view: str, parameters: dict) -> None:
    events = _events()
    events["latitude"] = 35.0; events["longitude"] = 139.0
    left, left_details, left_mask = perturb_view(events, source_hash="a" * 64, seed=9,
                                                  kind=view, parameters=parameters)
    right, right_details, right_mask = perturb_view(events.sample(frac=1, random_state=2),
        source_hash="a" * 64, seed=9, kind=view, parameters=parameters)
    assert left.equals(right)
    assert left_details == right_details
    assert left_mask.equals(right_mask)
    assert left.equals(left.sort_values(["user_id", "timestamp"]).reset_index(drop=True))
    if view == "gps":
        assert left.latitude.between(-90, 90).all() and left.longitude.between(-180, 180).all()
    if view == "leave-one-service-out": assert "s1" not in set(left.service_id)
    if view == "recent-truncation": assert left.groupby("user_id").size().eq(8).all()


def test_sparse_histories_are_reported_unencodable_and_masks_match_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    run, experiment = root / "smoke/run", root / "smoke/experiment"
    config = load_config(root / "configs/embedding/single_vector.yaml")
    config["evaluation"]["robustness"] = {"schema_version": "robustness-spec/1.0", "seed": 19,
        "views": {"recent-truncation": [{"remove_recent_events": 0}, {"remove_recent_events": 10000}]}}
    baseline = export_robustness_views(run / "observed", experiment / "prepared", Path("unused"),
        tmp_path, config, kind="baseline")
    def fake_learned(observed_dir, prepared_dir, checkpoint_path, output_path, config, **kwargs):
        return export_statistical_baseline(observed_dir, prepared_dir, output_path, config, **kwargs)
    monkeypatch.setattr("geoembeddings.robustness.export_embeddings", fake_learned)
    learned = export_robustness_views(run / "observed", experiment / "prepared",
        Path("fake-checkpoint"), tmp_path, config, kind="learned")
    for a, b in zip(baseline["artifacts"], learned["artifacts"]):
        assert a["changed_events"] == b["changed_events"]
        assert a["encoded_keys"] == b["encoded_keys"]
    assert baseline["artifacts"][1]["path"] is None
    assert baseline["artifacts"][1]["unencodable_keys"]
    assert baseline["artifacts"][0]["view_id"] == learned["artifacts"][0]["view_id"]
    assert baseline["information_boundary"].endswith("observed/ only")


def test_statistical_baseline_uses_authenticated_component_export_schema(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/embedding/single_vector.yaml")
    path = tmp_path / "baseline.npz"
    export_statistical_baseline(
        root / "smoke/run/observed", root / "smoke/experiment/prepared", path, config
    )

    loaded = load_embedding_export(path)
    assert loaded.schema_version == EXPORT_SCHEMA_VERSION
    assert loaded.arrays["model_variant"].item() == "statistical_baseline"
    assert loaded.components["persistent"].shape == loaded.components["combined"].shape
    assert not loaded.components["context"].any()


def test_dense_statistical_baseline_carries_calibration_identity(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    prepared = root / "smoke/experiment/prepared"
    path = tmp_path / "dense-baseline.npz"
    export_dense_statistical_baseline(
        root / "smoke/run/observed", prepared, path,
        load_config(root / "configs/embedding/single_vector.yaml"), event_stride=4,
    )

    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["preparation_hash"].item()) == sha256_file(
            prepared / "prepared_metadata.json"
        )
        assert len(payload["source_file_names"]) == len(payload["source_hashes"])
        assert len(payload["source_file_names"]) > 0


def test_context_session_configs_use_supported_robustness_schema_and_unknown_versions_fail() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "context_session_contrastive_candidate.yaml",
        "context_session_contrastive_detached_control.yaml",
    ):
        config = load_config(root / "configs/embedding" / name)
        robustness = config["evaluation"]["robustness"]
        assert robustness["schema_version"] == "robustness-spec/1.0"
        specs, _ = _specs(config, ["gps"])
        assert specs[0]["schema_version"] == "robustness-spec/1.0"

        invalid = dict(config)
        invalid["evaluation"] = dict(config["evaluation"])
        invalid["evaluation"]["robustness"] = dict(robustness)
        invalid["evaluation"]["robustness"]["schema_version"] = "robustness-spec/2.0"
        with pytest.raises(ValueError, match="Unsupported robustness specification version"):
            _specs(invalid, ["gps"])
