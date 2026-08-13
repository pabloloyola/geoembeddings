from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest

from geoembeddings.contract import OBSERVED_FILES, TRUTH_FILES
from scripts.kanto_trajectory_explorer import (
    EmptySelectionError, ExplorerData, deterministic_users, filter_events, load_data, main, render_map, resolve_paths,
)


def _data() -> ExplorerData:
    users = pd.DataFrame([{"user_id":"u1","age_group":"20-29","household_type":"single"},{"user_id":"u2","age_group":"40-49","household_type":"family"}])
    events = pd.DataFrame([
      {"user_id":"u1","timestamp":"2026-01-01T00:00:00Z","service_id":"location","action_type":"ping","observation_mode":"passive","region_id":"tokyo","latitude":35.,"longitude":139.,"location_accuracy_m":20},
      {"user_id":"u1","timestamp":"2026-01-02T00:00:00Z","service_id":"travel","action_type":"search","observation_mode":"active","region_id":"hakone","latitude":35.2,"longitude":139.1,"location_accuracy_m":50},
      {"user_id":"u2","timestamp":"2026-01-01T00:00:00Z","service_id":"ecommerce","action_type":"view","observation_mode":"active","region_id":"tokyo","latitude":35.1,"longitude":139.2,"location_accuracy_m":100},])
    events["timestamp"] = pd.to_datetime(events.timestamp, utc=True).dt.tz_convert("Asia/Tokyo")
    pois = pd.DataFrame([{"poi_id":"p1","latitude":35.2,"longitude":139.1,"category":"cafe","region_id":"hakone"}])
    requests = pd.DataFrame([{"request_id":"r1","user_id":"u1","request_timestamp":"2026-01-02","request_latitude":35.2,"request_longitude":139.1,"region_id":"hakone"}])
    return ExplorerData(users, events, pois, requests)


def _run(tmp_path: Path) -> Path:
    run=tmp_path/"run"; (run/"observed").mkdir(parents=True); (run/"truth").mkdir()
    (run/"manifest.json").write_text(json.dumps({"dataset_contract":{"name":"geoembeddings-dataset","version":"2.0"},"source_identity":"fixture"}))
    data=_data()
    tables={"users":data.users,"events":data.events,"poi_catalog":data.pois,"recommendation_requests":data.requests,"impressions":pd.DataFrame({"x":[]}),"interactions":pd.DataFrame({"x":[]})}
    for name,file in OBSERVED_FILES.items(): tables[name].to_csv(run/"observed"/file,index=False)
    truth=pd.DataFrame([{"trajectory_id":"t","user_id":"u1","timestamp":"2026-01-01T00:00:00Z","episode_id":"e","activity":"home","true_region_id":"tokyo","true_latitude":35.,"true_longitude":139.}])
    anchors=pd.DataFrame([{"user_id":"u1","home_latitude":35.,"home_longitude":139.,"work_latitude":35.1,"work_longitude":139.1}])
    for name,file in TRUTH_FILES.items(): (truth if name=="trajectories" else anchors).to_csv(run/"truth"/file,index=False)
    return run


def test_paths_are_canonical_and_default_excludes_truth(tmp_path: Path) -> None:
    run=_run(tmp_path); paths=resolve_paths(run)
    assert paths["events"] == (run/"observed"/OBSERVED_FILES["events"]).resolve()
    assert not any("truth" in path.parts for path in paths.values())


def test_truth_access_is_explicitly_gated(tmp_path: Path) -> None:
    run=_run(tmp_path)
    with pytest.raises(ValueError, match="requires evaluator-only"):
        load_data(run, include_anchors=True)
    data,_=load_data(run, include_truth=True, include_anchors=True)
    assert data.truth is not None and data.anchors is not None


def test_deterministic_sampling_and_filtering() -> None:
    events=pd.concat([_data().events.assign(user_id=f"u{i}") for i in range(10)],ignore_index=True)
    first=deterministic_users(events,3,42); second=deterministic_users(events.sample(frac=1),3,42)
    assert first == second and first[1]
    selected=filter_events(_data(),{"age_group":"20-29","service":"travel","region":"hakone","date":"2026-01-02"})
    assert selected.user_id.tolist()==["u1"]


def test_empty_selection_does_not_write(tmp_path: Path) -> None:
    with pytest.raises(EmptySelectionError):
        render_map(_data(), _data().events.iloc[0:0], tmp_path/"x.html", manifest={}, filters={}, seed=1,
                   max_users=1,max_markers=2,max_path_points=2,include_truth=False,include_anchors=False)
    assert not (tmp_path/"x.html").exists()


def test_output_immutability_and_metadata(tmp_path: Path) -> None:
    pytest.importorskip("folium"); run=_run(tmp_path)
    assert main(["--run-dir",str(run),"--output",str(run/"map.html")]) == 2
    output=tmp_path/"outside.html"
    assert main(["--run-dir",str(run),"--output",str(output),"--max-users","1","--max-markers","1","--max-path-points","1","--seed","9"]) == 0
    metadata=json.loads((tmp_path/"outside.html.metadata.json").read_text())
    assert metadata["truth_access"] is False and metadata["seed"]==9 and metadata["truncated"] is True
    assert "Display truncated" in output.read_text()
    assert main(["--run-dir",str(run),"--output",str(output)]) == 2
