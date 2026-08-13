from pathlib import Path

import numpy as np
import pytest

from geoembeddings.cli import _named_roots, build_parser
from geoembeddings.layout import PrivacyEvidenceLayout, UtilityReportLayout
from geoembeddings.privacy import construct_sensitive_probe_population
from geoembeddings.representation_schema import LoadedEmbeddingExport


def test_audit_privacy_accepts_only_canonical_roots() -> None:
    args = build_parser().parse_args([
        "audit-privacy", "--run-dir", "run",
        "--experiment-dir", "statistical_baseline=experiments/baseline",
        "--experiment-dir", "capacity_matched_single=experiments/learned",
        "--evidence-dir", "evidence", "--utility-report-dir", "utility",
        "--output-dir", "audit",
    ])
    assert args.command == "audit-privacy"
    assert _named_roots(args.experiment_dir) == {
        "statistical_baseline": Path("experiments/baseline"),
        "capacity_matched_single": Path("experiments/learned"),
    }
    assert not hasattr(args, "truth_path")
    assert not hasattr(args, "observed_path")


def test_privacy_root_layouts_resolve_internal_files(tmp_path: Path) -> None:
    evidence = PrivacyEvidenceLayout.from_path(tmp_path / "evidence")
    utility = UtilityReportLayout.from_path(tmp_path / "utility")
    assert evidence.evidence_index == (tmp_path / "evidence" / "evidence_index.json").resolve()
    assert utility.report("control") == (tmp_path / "utility" / "control.json").resolve()
    with pytest.raises(ValueError, match="Invalid utility-report name"):
        utility.report("../control")


def test_named_experiment_roots_reject_duplicates_and_paths_without_names() -> None:
    with pytest.raises(ValueError, match="unique NAME=ROOT"):
        _named_roots(["control=one", "control=two"])
    with pytest.raises(ValueError, match="unique NAME=ROOT"):
        _named_roots(["experiment"])


def test_sensitive_probe_population_runs_without_membership_labels() -> None:
    """End-to-end population fixture for independent sensitive applicability."""
    users = np.asarray([f"u{i}" for i in range(30) for _ in range(3)])
    cutoffs = np.tile(np.asarray(["train_end", "validation_end", "test_end"]), 30)
    vectors = np.arange(len(users), dtype=float)[:, None]
    export = LoadedEmbeddingExport(
        arrays={"user_id": users, "cutoff": cutoffs},
        components={"combined": vectors}, schema_version="fixture", compatibility="fixture",
    )
    provenance = {f"u{i}": {"history_event_count": float(i)} for i in range(30)}
    population = construct_sensitive_probe_population(
        export, target_model_lineage="authenticated-lineage", eligible_users=tuple(provenance),
        provenance_by_user=provenance,
        cutoff_order=("train_end", "validation_end", "test_end"),
        component_order=("combined",), split_seed=17,
    )
    assert population.status == "available"
    assert all(record.membership is None for record in population.records)
    assert set(record.split for record in population.records) == {"train", "validation", "test"}
    assert dict(population.user_set_hashes)["eligible"]
