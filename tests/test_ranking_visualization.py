from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoembeddings.contract import OBSERVED_FILES
from geoembeddings.io import sha256_file
from geoembeddings.ranking import RANKING_PREDICTION_SCHEMA, RANKING_REPORT_SCHEMA
from geoembeddings.ranking_visualization import DEFAULT_MODELS, render_ranking_explanation
from geoembeddings.recommendation import IMPRESSION_FIELDS, INTERACTION_FIELDS, POI_FIELDS, REQUEST_FIELDS


def _sources(root: Path, *, interaction: bool = True) -> Path:
    observed = root / "observed"
    observed.mkdir(parents=True)
    rows = {
        "poi_catalog": [
            ("a", "hakone", "kanagawa", "cafe", 35.0, 139.0, 2, .8, "indoor", .7, "09:00", "18:00", "2026-01-01T00:00:00"),
            ("b", "hakone", "kanagawa", "museum", 35.1, 139.1, 3, .9, "indoor", .6, "09:00", "18:00", "2026-01-01T00:00:00"),
            ("x", "hakone", "kanagawa", "park", 35.2, 139.2, 1, 1.0, "outdoor", .4, "09:00", "18:00", "2026-01-01T00:00:00"),
        ],
        "recommendation_requests": [("r", "u", "2026-01-02T10:00:00", "hakone", 35.0, 139.0, "arrival")],
        "impressions": [("r", "a", 1, 1, 1, 1, 5.0, "2026-01-02T09:59:00"),
                        ("r", "b", 2, 1, 0, 0, 10.0, "2026-01-02T09:59:00"),
                        ("r", "x", 3, 0, 0, 0, 15.0, "2026-01-02T09:59:00")],
        "interactions": ([('i', 'r', 'a', '2026-01-02T10:05:00', 'click')] if interaction else []),
    }
    fields = {"poi_catalog": POI_FIELDS, "recommendation_requests": REQUEST_FIELDS,
              "impressions": IMPRESSION_FIELDS, "interactions": INTERACTION_FIELDS}
    for name, values in rows.items():
        pd.DataFrame(values, columns=fields[name]).to_csv(observed / OBSERVED_FILES[name], index=False)
    return observed


def _artifacts(observed: Path, ranking: Path, *, candidate_hash: str = "candidates", tie_order=("a", "b")) -> None:
    ranking.mkdir(parents=True)
    source_hashes = {name: sha256_file(observed / OBSERVED_FILES[name]) for name in
                     ("poi_catalog", "recommendation_requests", "impressions", "interactions")}
    for model in DEFAULT_MODELS:
        np.savez_compressed(ranking / f"{model}.npz", schema_version=np.asarray(RANKING_PREDICTION_SCHEMA),
            model=np.asarray(model), request_id=np.asarray(["r", "r"]), poi_id=np.asarray(tie_order),
            rank=np.asarray([1, 2]), score=np.asarray([1.0, 1.0]), request_hash=np.asarray("requests"),
            candidate_hash=np.asarray(candidate_hash))
        (ranking / f"{model}.json").write_text(json.dumps({"schema_version": RANKING_REPORT_SCHEMA,
            "model": model, "request_hash": "requests", "candidate_hash": candidate_hash,
            "source_hashes": source_hashes, "coverage": {"unscorable_requests": {}}}))


def test_renderer_rejects_mismatched_candidate_sets_and_truth_paths(tmp_path: Path) -> None:
    observed, ranking = _sources(tmp_path), tmp_path / "experiment" / "ranking"
    _artifacts(observed, ranking)
    report = json.loads((ranking / "nearest.json").read_text())
    report["candidate_hash"] = "different"
    (ranking / "nearest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="candidate_hash mismatch"):
        render_ranking_explanation(observed, ranking, ranking / "visualization")
    with pytest.raises(ValueError, match="only the canonical observed"):
        render_ranking_explanation(tmp_path / "truth", ranking, ranking / "visualization")


def test_renderer_handles_unavailable_and_missing_interactions(tmp_path: Path) -> None:
    observed, ranking = _sources(tmp_path, interaction=False), tmp_path / "experiment" / "ranking"
    _artifacts(observed, ranking)
    result = render_ranking_explanation(observed, ranking, ranking / "visualization")
    rendered = (ranking / "visualization" / "ranking_explanation.html").read_text()
    assert result["candidate_identity"]["unavailable_poi_ids"] == ["x"]
    assert "unavailable (excluded)" in rendered
    assert "none recorded" in rendered
    assert result["protected_utility"]["status"] == "unavailable"


def test_renderer_authenticates_deterministic_ties(tmp_path: Path) -> None:
    observed, ranking = _sources(tmp_path), tmp_path / "experiment" / "ranking"
    _artifacts(observed, ranking, tie_order=("b", "a"))
    with pytest.raises(ValueError, match="deterministic score/POI tie ordering"):
        render_ranking_explanation(observed, ranking, ranking / "visualization")
