from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from geoembeddings.layout import DatasetLayout, ExperimentLayout
from geoembeddings.representation_schema import EXPORT_SCHEMA_VERSION
from geoembeddings.user_journey_visualization import (
    annotate_boundaries, build_user_journey_report, deterministic_rows,
    reject_protected_columns,
)


def test_temporal_ordering_and_deterministic_truncation():
    frame = pd.DataFrame({"timestamp": ["2026-01-02", "2026-01-01", "2026-01-01"],
                          "event_id": ["z", "b", "a"]})
    selected = deterministic_rows(frame, timestamp="timestamp", limit=2, tie_breakers=("event_id",))
    assert selected.event_id.tolist() == ["a", "b"]


def test_boundary_annotation_is_exact_and_ordered():
    episodes = pd.DataFrame({"episode_id": ["b", "a"], "start_time": ["2026-01-02", "2026-01-01"],
                             "end_time": ["2026-01-03", "2026-01-02"]})
    marks = annotate_boundaries(["2026-01-02"], episodes)[0]
    assert [(m["kind"], m["episode_id"]) for m in marks] == [("end", "a"), ("start", "b")]


def test_protected_input_rejection():
    with pytest.raises(ValueError, match="protected fields"):
        reject_protected_columns(pd.DataFrame({"timestamp": [], "true_utility": []}), source="events")


def _artifact_tree(tmp_path):
    run = DatasetLayout.from_path(tmp_path / "run"); exp = ExperimentLayout.from_path(tmp_path / "experiment")
    run.observed.mkdir(parents=True); exp.root.mkdir(parents=True)
    tables = {
      "events": pd.DataFrame(columns=["event_id","user_id","timestamp","service_id","action","category","region_id","latitude","longitude"]),
      "users": pd.DataFrame({"user_id":["other"]}),
      "poi_catalog": pd.DataFrame(columns=["poi_id","category"]),
      "recommendation_requests": pd.DataFrame(columns=["request_id","user_id","timestamp","region_id","context_source"]),
      "impressions": pd.DataFrame(columns=["request_id","poi_id","position","is_available","travel_time_minutes"]),
      "interactions": pd.DataFrame(columns=["interaction_id","request_id","poi_id","timestamp","interaction_type"]),
    }
    hashes={}
    for name, frame in tables.items():
        path=run.observed_file(name); frame.to_csv(path,index=False,compression="gzip")
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    run.manifest_path.write_text(json.dumps({"dataset_contract":{"name":"geoembeddings-dataset","version":"2.0"}}))
    names=np.asarray(list(hashes)); digests=np.asarray([hashes[n] for n in names])
    values=np.asarray([[1.,2.],[2.,3.]])
    np.savez_compressed(exp.dense_embeddings, user_id=np.asarray(["other","missing"]),
      timestamp=np.asarray(["2026-01-01","2026-01-01"]), schema_version=np.asarray(EXPORT_SCHEMA_VERSION),
      model_variant=np.asarray("factorized_pc"), categorical_fields=np.asarray([],dtype=str), continuous_fields=np.asarray([],dtype=str),
      preparation_hash=np.asarray("x"), source_file_names=names, source_hashes=digests,
      train_end=np.asarray("2026-01-01"), validation_end=np.asarray("2026-01-02"), export_cutoffs=np.asarray([],dtype=str),
      compatibility=np.asarray("combined alias"), component_names=np.asarray(["persistent","context","combined"]),
      component_dimensions=np.asarray([2,2,2]), component_persistent=values, component_context=values,
      component_combined=values, embedding=values)
    return run,exp


@pytest.mark.integration
def test_observed_only_missing_history_never_resolves_truth(tmp_path, monkeypatch):
    run,exp=_artifact_tree(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("truth path was opened")
    monkeypatch.setattr(DatasetLayout, "truth_file", forbidden)
    result=build_user_journey_report(run,exp,user_id="missing",start="2026-01-01",end="2026-01-02")
    assert result["counts"]["events"] == 0
    assert result["counts"]["embedding_timestamps"] == 1
    assert "SYNTHETIC PROTECTED TRUTH" not in exp.user_journey_report("missing").read_text()
