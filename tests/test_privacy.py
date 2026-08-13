from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
import yaml

from geoembeddings.privacy import (PrivacyPopulation, PrivacyPopulationRecord,
                                   construct_privacy_population, run_privacy_attacks_from_config)
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
    components = {
        "persistent": np.arange(len(rows), dtype=np.float32)[:, None],
        "context": (10 + np.arange(len(rows), dtype=np.float32))[:, None],
        "combined": np.column_stack((np.arange(len(rows)), np.arange(len(rows)) + 1)).astype(np.float32),
    }
    return LoadedEmbeddingExport({"user_id": np.asarray(users), "cutoff": np.asarray(cutoffs)},
                                 components, "test", "test")


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
