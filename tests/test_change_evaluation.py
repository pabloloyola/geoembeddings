from __future__ import annotations

import gzip
import csv
from pathlib import Path

import numpy as np
import pytest

from geoembeddings.evaluation import evaluate_change
from geoembeddings.simulate_pair import simulate_pair


CONFIG = Path("configs/simulation/kanto_v1.yaml")


def _dense(run: Path, experiment: Path, name: str) -> None:
    with gzip.open(run / "observed/observed_events.csv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = list({(row["user_id"], row["timestamp"]): row for row in rows}.values())
    # Deterministic vectors make the test about alignment/coverage, not a model.
    vectors = np.asarray([[1.0, (index + 1) / max(1, len(rows))] for index in range(len(rows))])
    experiment.mkdir(parents=True)
    np.savez_compressed(experiment / name, user_id=np.asarray([r["user_id"] for r in rows]),
        timestamp=np.asarray([r["timestamp"] for r in rows]), cutoff_kind=np.asarray(["observed_event"] * len(rows)),
        embedding=vectors, history_event_count=np.arange(1, len(rows) + 1))


def test_temporary_change_reports_recovery_and_immutable_outputs(tmp_path: Path) -> None:
    result = simulate_pair(CONFIG, tmp_path / "ref", tmp_path / "changed", tmp_path / "pair",
                           intervention="temporary-trip", users=10, days=9, seed=20260811)
    roots = []
    for side, run in (("ref", tmp_path / "ref"), ("changed", tmp_path / "changed")):
        baseline, learned = tmp_path / f"{side}-baseline", tmp_path / f"{side}-learned"
        _dense(run, baseline, "dense_statistical_baseline.npz")
        _dense(run, learned, "dense_embeddings.npz")
        roots.append((baseline, learned))
    report = evaluate_change(result["pair_manifest"], [roots[0][0], roots[1][0]], [roots[0][1], roots[1][1]])
    assert report["representations"]["baseline"]["censoring"]["right_censored"] is False
    assert report["representations"]["baseline"]["coverage"]["eligible_users"] > 0
    with pytest.raises(FileExistsError, match="immutable"):
        evaluate_change(result["pair_manifest"], [roots[0][0], roots[1][0]], [roots[0][1], roots[1][1]])


def test_change_evaluation_rejects_stale_pair_artifact(tmp_path: Path) -> None:
    result = simulate_pair(CONFIG, tmp_path / "ref", tmp_path / "changed", tmp_path / "pair",
                           intervention="sustained-preference", users=10, days=9, seed=20260811)
    manifest = Path(result["pair_manifest"])
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="stale"):
        evaluate_change(manifest, [tmp_path / "a", tmp_path / "b"], [tmp_path / "c", tmp_path / "d"])
