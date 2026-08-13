from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
import yaml

from geoembeddings.privacy import (PrivacyPopulation, PrivacyPopulationRecord,
                                   construct_privacy_population, run_privacy_attacks_from_config,
                                   seeded_matched_representation_delta_bootstrap,
                                   seeded_stratified_user_bootstrap)
from geoembeddings.privacy_evaluation import PRIVACY_CONFIG_SCHEMA, load_privacy_config, validate_privacy_config
from geoembeddings.representation_schema import LoadedEmbeddingExport


CONFIG = Path("configs/privacy/diagnostic_v1.yaml")


def raw_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_diagnostic_privacy_config_is_typed_and_complete() -> None:
    config = load_privacy_config(CONFIG)
    assert config.schema_version == PRIVACY_CONFIG_SCHEMA
    assert config.split.train_fraction + config.split.validation_fraction + config.split.test_fraction == 1
    assert config.membership["statistical_baseline"]["training_membership_status"] == "not_applicable"
    assert config.sensitive_attributes[0].derivation.version.endswith("/1.0")


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.update({"surprise": True}), "unknown fields"),
    (lambda c: c["splits"].update({"test_fraction": 0.3}), "sum to 1"),
    (lambda c: c["matching"]["variables"][0].update({"bin_boundaries": [0, 10, 10]}), "overlapping"),
    (lambda c: c["support"].update({"minimum_total": 0}), "positive"),
    (lambda c: c["sensitive_attributes"][0]["derivation"].update({"version": "latest"}), "versioned"),
    (lambda c: c["membership"].update({"nonmember_definition": c["membership"]["member_definition"]}), "overlap"),
])
def test_privacy_config_rejects_invalid_protocol(mutation, match: str) -> None:
    config = deepcopy(raw_config())
    mutation(config)
    with pytest.raises(ValueError, match=match):
        validate_privacy_config(config)


def test_privacy_config_rejects_unfrozen_derivation() -> None:
    config = raw_config()
    config["sensitive_attributes"][0]["derivation"]["method"] = "quantiles_fit_on_all_labels"
    with pytest.raises(ValueError, match="not frozen"):
        validate_privacy_config(config)


def _export(rows):
    users, cutoffs = zip(*rows)
    user_values = np.asarray([float(user.removeprefix("u").removeprefix("m").removeprefix("n"))
                              if user.lstrip("umn").isdigit() else float(sum(map(ord, user)))
                              for user in users], dtype=np.float32)
    cutoff_values = np.asarray([float(cutoff == "test_end") for cutoff in cutoffs], dtype=np.float32)
    components = {
        "persistent": user_values[:, None],
        "context": cutoff_values[:, None],
        "combined": np.column_stack((user_values, cutoff_values)),
    }
    return LoadedEmbeddingExport({"user_id": np.asarray(users), "cutoff": np.asarray(cutoffs)},
                                 components, "test", "test")


@pytest.fixture
def small_synthetic_exports(tmp_path: Path) -> tuple[Path, Path]:
    """Two equivalent on-disk exports avoid dependence on historical artifact bytes."""
    rows = [(f"u{i}", cutoff) for i in range(8)
            for cutoff in ("train_end", "test_end")]
    paths = (tmp_path / "forward.npz", tmp_path / "reversed.npz")
    for path, ordered in zip(paths, (rows, list(reversed(rows))), strict=True):
        np.savez_compressed(
            path,
            user_id=np.asarray([user for user, _ in ordered]),
            cutoff=np.asarray([cutoff for _, cutoff in ordered]),
            embedding=np.asarray([
                [float(user.removeprefix("u")), float(cutoff == "test_end")]
                for user, cutoff in ordered
            ], dtype=np.float32),
        )
    return paths


def test_on_disk_synthetic_exports_have_order_independent_population_identity(
        small_synthetic_exports: tuple[Path, Path]) -> None:
    membership = {f"u{i}": i % 2 == 0 for i in range(8)}
    kwargs = dict(
        target_model_lineage="fixture-lineage", membership_by_user=membership,
        provenance_by_user={user: {"history": 1.0} for user in membership},
        cutoff_order=("train_end", "test_end"), component_order=("combined",),
        matching_boundaries={"history": (0, 2)}, audit_seed=31, split_seed=32,
    )
    forward = construct_privacy_population(small_synthetic_exports[0], **kwargs)
    reversed_rows = construct_privacy_population(small_synthetic_exports[1], **kwargs)
    assert forward.user_set_hashes == reversed_rows.user_set_hashes
    assert sorted((r.user_id, r.membership, r.split) for r in forward.records) == sorted(
        (r.user_id, r.membership, r.split) for r in reversed_rows.records
    )


def test_population_groups_cutoffs_masks_matches_and_splits_whole_users() -> None:
    export = _export([("m1", "train_end"), ("m1", "test_end"), ("n1", "train_end"),
                      ("m2", "train_end"), ("n2", "train_end")])
    membership = {"m1": True, "m2": True, "n1": False, "n2": False}
    provenance = {user: {"history": 5.0} for user in membership}
    result = construct_privacy_population(
        export, target_model_lineage="checkpoint-sha", membership_by_user=membership,
        provenance_by_user=provenance, cutoff_order=("train_end", "test_end"),
        component_order=("combined", "persistent"), matching_boundaries={"history": (0, 10)},
        audit_seed=7, split_seed=8,
    )
    assert result.status == "available"
    assert len(result.records) == 4
    m1 = next(record for record in result.records if record.user_id == "m1")
    assert m1.membership is True
    assert m1.missing_cutoff_mask == (0, 0)
    assert m1.missing_component_mask == (0, 0, 0, 0)
    n1 = next(record for record in result.records if record.user_id == "n1")
    assert n1.missing_cutoff_mask == (0, 1)
    assert n1.missing_component_mask == (0, 0, 1, 1)
    assert len({record.user_id for record in result.records}) == len(result.records)
    assert set(result.provenance_feature_names).isdisjoint(result.vector_feature_names)


def test_population_is_unavailable_without_common_support_and_authenticates_sets() -> None:
    result = construct_privacy_population(
        _export([("member", "train_end"), ("nonmember", "train_end")]),
        target_model_lineage="lineage", membership_by_user={"member": True, "nonmember": False},
        provenance_by_user={"member": {"history": 1}, "nonmember": {"history": 11}},
        cutoff_order=("train_end",), component_order=("combined",),
        matching_boundaries={"history": (0, 10, 20)}, audit_seed=1, split_seed=2,
    )
    assert result.status == "unavailable"
    assert result.reason == "membership_classes_inadequate"
    with pytest.raises(ValueError, match="Post-hoc 'matched' user-set change"):
        construct_privacy_population(
            _export([("member", "train_end"), ("nonmember", "train_end")]),
            target_model_lineage="lineage", membership_by_user={"member": True, "nonmember": False},
            provenance_by_user={"member": {"history": 1}, "nonmember": {"history": 1}},
            cutoff_order=("train_end",), component_order=("combined",),
            matching_boundaries={"history": (0, 10)}, audit_seed=1, split_seed=2,
            expected_user_set_hashes={"matched": "changed"},
        )


def _attack_population() -> PrivacyPopulation:
    records = []
    splits = ["train"] * 12 + ["validation"] * 6 + ["test"] * 6
    for index, split in enumerate(splits):
        member = index % 2 == 0
        records.append(PrivacyPopulationRecord(
            f"u{index}", member, split, (float(member), float(index % 3)), (0,), (0,),
            (float(index % 4),), (0,),
        ))
    return PrivacyPopulation("test", "available", None, "lineage", ("test",), ("combined",),
                             ("v0", "v1"), ("history",), tuple(records), (), ())


def test_attack_families_report_frozen_test_metrics_and_undefined_values() -> None:
    result = run_privacy_attacks_from_config(_attack_population(), None, load_privacy_config(CONFIG))
    assert result["status"] == "available"
    assert set(result["attacks"]) == {"deterministic_random", "majority_base_rate",
                                      "provenance_logistic", "vector_logistic",
                                      "vector_plus_provenance_logistic", "bounded_nonlinear"}
    vector = result["attacks"]["vector_logistic"]
    assert vector["test_scoring_passes"] == 1
    assert vector["preprocessing_fit_split"] == "train"
    assert vector["selection_split"] == "validation"
    assert vector["roc_auc"]["value"] == pytest.approx(1.0)
    assert vector["threshold_rule"] == "score_greater_than_or_equal_to_threshold"


def test_sensitive_binary_metrics_and_undefined_auc_are_explicit() -> None:
    population = _attack_population()
    labels = {record.user_id: ("a" if record.split == "test" else ("a" if i % 2 else "b"))
              for i, record in enumerate(population.records)}
    result = run_privacy_attacks_from_config(population, labels, load_privacy_config(CONFIG),
                                             task="sensitive_attribute")
    metric = result["attacks"]["majority_prior"]
    assert metric["roc_auc"]["value"] is None
    assert metric["roc_auc"]["undefined_reason"] == "test_split_lacks_both_classes"
    assert metric["per_class_support"] == {"a": 6, "b": 0}


def test_seeded_user_bootstrap_is_deterministic_and_preserves_classes() -> None:
    users = [f"u{i}" for i in range(12)]
    labels = np.asarray([i % 2 for i in range(12)])
    scores = labels * 0.7 + np.asarray([i % 3 for i in range(12)]) * 0.05
    strata = [(i % 2, i % 3) for i in range(12)]
    auc = lambda y, score: float(np.mean(score[y == 1]) - np.mean(score[y == 0]))
    first = seeded_stratified_user_bootstrap(
        users, labels, scores, strata=strata, metric=auc, replicates=100, seed=91,
    )
    second = seeded_stratified_user_bootstrap(
        list(reversed(users)), labels[::-1], scores[::-1], strata=list(reversed(strata)),
        metric=auc, replicates=100, seed=91,
    )
    assert first == second
    assert first["interval"] == second["interval"]
    assert first["successful_replicates"] == 100
    assert first["degenerate_replicates"] == first["excluded_replicates"] == 0
    assert first["requested_replicates"] == first["replicate_count"] == 100
    assert first["confidence_level"] == 0.95


def test_matched_delta_bootstrap_resamples_identical_users_deterministically() -> None:
    users = [f"u{i}" for i in range(10)]
    labels = np.asarray([i % 2 for i in range(10)])
    reference = labels * 0.4 + np.arange(10) * 0.01
    candidate = reference + labels * 0.2
    metric = lambda y, score: float(np.mean(score[y == 1]) - np.mean(score[y == 0]))
    kwargs = dict(reference="baseline", strata=labels, metric=metric,
                  replicates=80, seed=1234, confidence_level=0.95)
    first = seeded_matched_representation_delta_bootstrap(
        users, labels, {"baseline": reference, "learned": candidate}, **kwargs,
    )
    second = seeded_matched_representation_delta_bootstrap(
        users, labels, {"baseline": reference, "learned": candidate}, **kwargs,
    )
    assert first == second
    assert first["deltas"]["learned"]["point_estimate"] == pytest.approx(0.2)
    assert first["deltas"]["learned"]["interval"] == pytest.approx([0.2, 0.2])


def test_primary_attack_metrics_include_frozen_bootstrap_protocol() -> None:
    result = run_privacy_attacks_from_config(_attack_population(), None, load_privacy_config(CONFIG))
    bootstrap = result["attacks"]["vector_logistic"]["roc_auc"]["bootstrap"]
    assert bootstrap["seed"] == 20260818
    assert bootstrap["method"] == "stratified_user_percentile"
    assert bootstrap["replicate_count"] == 1000
    assert bootstrap["successful_replicates"] == 1000


def test_membership_population_is_deterministic_under_export_and_mapping_order() -> None:
    """Membership is a user property; neither NPZ rows nor dict order may derive it."""
    rows = [(f"u{i}", cutoff) for i in range(12)
            for cutoff in ("train_end", "test_end")]
    membership = {f"u{i}": i % 2 == 0 for i in range(12)}
    provenance = {user: {"history": float(index % 3)}
                  for index, user in enumerate(membership)}
    kwargs = dict(
        target_model_lineage="frozen-checkpoint", cutoff_order=("train_end", "test_end"),
        component_order=("combined",), matching_boundaries={"history": (0, 1, 2, 3)},
        audit_seed=19, split_seed=23,
    )
    first = construct_privacy_population(
        _export(rows), membership_by_user=membership,
        provenance_by_user=provenance, **kwargs,
    )
    second = construct_privacy_population(
        _export(list(reversed(rows))),
        membership_by_user=dict(reversed(list(membership.items()))),
        provenance_by_user=dict(reversed(list(provenance.items()))), **kwargs,
    )
    identity = lambda population: sorted(
        (r.user_id, r.membership, r.split, r.vector_features, r.missing_cutoff_mask)
        for r in population.records
    )
    assert identity(first) == identity(second)
    assert first.user_set_hashes == second.user_set_hashes


def test_matching_is_balanced_without_replacement_and_splits_are_disjoint() -> None:
    membership = {**{f"m{i}": True for i in range(8)},
                  **{f"n{i}": False for i in range(5)}}
    population = construct_privacy_population(
        _export([(user, "train_end") for user in membership]),
        target_model_lineage="lineage", membership_by_user=membership,
        provenance_by_user={user: {"history": 1.0} for user in membership},
        cutoff_order=("train_end",), component_order=("combined",),
        matching_boundaries={"history": (0, 2)}, audit_seed=2, split_seed=3,
    )
    records = population.records
    assert len(records) == 10
    assert sum(record.membership for record in records) == 5
    assert len({record.user_id for record in records}) == len(records)
    split_sets = {
        split: {record.user_id for record in records if record.split == split}
        for split in ("train", "validation", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    assert set().union(*split_sets.values()) == {record.user_id for record in records}


def test_attack_is_unavailable_for_imbalanced_or_missing_split_classes() -> None:
    population = _attack_population()
    all_members = PrivacyPopulation(
        population.schema_version, population.status, population.reason,
        population.target_model_lineage, population.cutoff_order,
        population.component_order, population.vector_feature_names,
        population.provenance_feature_names,
        tuple(PrivacyPopulationRecord(
            r.user_id, True, r.split, r.vector_features, r.missing_cutoff_mask,
            r.missing_component_mask, r.provenance_covariates, r.matching_stratum,
        ) for r in population.records), population.excluded_users,
        population.user_set_hashes,
    )
    result = run_privacy_attacks_from_config(all_members, None, load_privacy_config(CONFIG))
    assert result == {
        "status": "unavailable", "reason": "attack_split_support_inadequate", "attacks": {}
    }


def test_common_support_threshold_returns_explicit_unavailable_result() -> None:
    membership = {f"u{i}": i % 2 == 0 for i in range(8)}
    population = construct_privacy_population(
        _export([(user, "train_end") for user in membership]),
        target_model_lineage="lineage", membership_by_user=membership,
        provenance_by_user={user: {"history": 1.0} for user in membership},
        cutoff_order=("train_end",), component_order=("combined",),
        matching_boundaries={"history": (0, 2)}, audit_seed=5, split_seed=6,
        minimum_total=10,
    )
    assert population.status == "unavailable"
    assert population.reason == "common_support_inadequate"


def test_bootstrap_reports_degenerate_replicates_instead_of_hiding_them() -> None:
    result = seeded_stratified_user_bootstrap(
        ["u0", "u1"], np.asarray([0, 1]), np.asarray([0.1, 0.9]),
        strata=["only", "only"], metric=lambda y, score: (
            float(score[y == 1].mean() - score[y == 0].mean())
            if len(np.unique(y)) == 2 else float("nan")
        ), replicates=40, seed=9,
    )
    assert result["requested_replicates"] == 40
    # Class is part of the effective stratum, so both singleton classes are
    # retained in every replicate rather than being silently degenerate.
    assert result["degenerate_replicates"] == 0
    assert result["excluded_replicates"] == 0
    assert result["successful_replicates"] == 40
