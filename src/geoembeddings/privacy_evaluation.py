"""Typed configuration boundary for the protected R12 privacy evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PRIVACY_CONFIG_SCHEMA = "geoembeddings-privacy-diagnostic-config/1.0"


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _fields(value: Any, path: str, required: set[str]) -> dict[str, Any]:
    result = _mapping(value, path)
    unknown = set(result) - required
    missing = required - set(result)
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return result


def _integer(value: Any, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _ordered_unique_text(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty ordered list")
    result = tuple(_text(item, f"{path}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{path} contains overlapping definitions")
    return result


def _boundaries(value: Any, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{path} must contain at least two boundaries")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{path} boundaries must be numeric")
    result = tuple(float(item) for item in value)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{path} contains overlapping definitions")
    return result


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class SupportConfig:
    minimum_total: int
    minimum_per_class: int
    minimum_per_stratum: int
    minimum_sensitive_label_cell: int


@dataclass(frozen=True)
class MatchingVariable:
    name: str
    bin_boundaries: tuple[float, ...]


@dataclass(frozen=True)
class AttributeDerivation:
    version: str
    method: str
    source_field: str
    labels: tuple[str, ...]
    bin_boundaries: tuple[float, ...] | None
    categories: tuple[str, ...] | None


@dataclass(frozen=True)
class SensitiveAttribute:
    name: str
    source: str
    derivation: AttributeDerivation


@dataclass(frozen=True)
class PrivacyAuditConfig:
    schema_version: str
    audit_seed: int
    split_seed: int
    membership: dict[str, Any]
    cutoff_order: tuple[str, ...]
    component_order: tuple[str, ...]
    feature_construction: str
    feature_missingness: str
    matching_variables: tuple[MatchingVariable, ...]
    split: SplitConfig
    support: SupportConfig
    sensitive_attributes: tuple[SensitiveAttribute, ...]
    attacks: dict[str, Any]
    imbalance: dict[str, Any]
    bootstrap: dict[str, Any]
    reporting: dict[str, Any]


def _parse_attribute(value: Any, index: int) -> SensitiveAttribute:
    path = f"sensitive_attributes[{index}]"
    item = _fields(value, path, {"name", "source", "derivation"})
    derivation = _mapping(item["derivation"], f"{path}.derivation")
    common = {"version", "method", "source_field", "labels"}
    method = derivation.get("method")
    method_field = "bin_boundaries" if method == "fixed_edges" else "categories" if method == "categorical_mapping" else None
    if method_field is None:
        raise ValueError(f"{path}.derivation.method is not frozen to a supported derivation")
    derivation = _fields(derivation, f"{path}.derivation", common | {method_field})
    labels = _ordered_unique_text(derivation["labels"], f"{path}.derivation.labels")
    boundaries = _boundaries(derivation[method_field], f"{path}.derivation.bin_boundaries") if method_field == "bin_boundaries" else None
    categories = _ordered_unique_text(derivation[method_field], f"{path}.derivation.categories") if method_field == "categories" else None
    expected = len(boundaries) - 1 if boundaries else len(categories or ())
    if len(labels) != expected:
        raise ValueError(f"{path}.derivation labels do not match the frozen cells")
    version = _text(derivation["version"], f"{path}.derivation.version")
    if "/" not in version:
        raise ValueError(f"{path}.derivation.version must be versioned")
    return SensitiveAttribute(_text(item["name"], f"{path}.name"), _text(item["source"], f"{path}.source"),
                              AttributeDerivation(version, method, _text(derivation["source_field"], f"{path}.derivation.source_field"), labels, boundaries, categories))


def validate_privacy_config(raw: Any) -> PrivacyAuditConfig:
    """Reject underspecified or mutable privacy-audit protocols."""
    root_fields = {"schema_version", "seeds", "membership", "features", "matching", "splits", "support",
                   "sensitive_attributes", "attacks", "imbalance", "bootstrap", "reporting"}
    config = _fields(raw, "privacy", root_fields)
    if config["schema_version"] != PRIVACY_CONFIG_SCHEMA:
        raise ValueError("Unsupported privacy configuration schema_version")
    seeds = _fields(config["seeds"], "seeds", {"audit", "split"})
    membership = _fields(config["membership"], "membership", {"unit", "member_definition", "nonmember_definition", "statistical_baseline"})
    for key in ("unit", "member_definition", "nonmember_definition"): _text(membership[key], f"membership.{key}")
    if membership["member_definition"] == membership["nonmember_definition"]:
        raise ValueError("membership definitions overlap")
    baseline = _fields(membership["statistical_baseline"], "membership.statistical_baseline", {"training_membership_status", "reason"})
    if baseline["training_membership_status"] != "not_applicable" or baseline["reason"] != "no_learned_target_parameters":
        raise ValueError("statistical baseline membership must be frozen as not_applicable")
    features = _fields(config["features"], "features", {"cutoff_order", "component_order", "construction", "missingness"})
    if features["construction"] != "concatenate_cutoff_then_component" or features["missingness"] != "append_binary_mask":
        raise ValueError("feature construction is not the frozen ordered construction")
    matching = _fields(config["matching"], "matching", {"without_replacement", "variables"})
    if matching["without_replacement"] is not True or not isinstance(matching["variables"], list) or not matching["variables"]:
        raise ValueError("matching must define variables and be without replacement")
    variables = []
    for index, value in enumerate(matching["variables"]):
        item = _fields(value, f"matching.variables[{index}]", {"name", "bin_boundaries"})
        variables.append(MatchingVariable(_text(item["name"], f"matching.variables[{index}].name"), _boundaries(item["bin_boundaries"], f"matching.variables[{index}].bin_boundaries")))
    if len({item.name for item in variables}) != len(variables): raise ValueError("matching variables contain overlapping definitions")
    fractions = _fields(config["splits"], "splits", {"train_fraction", "validation_fraction", "test_fraction"})
    fraction_values = [float(fractions[name]) for name in ("train_fraction", "validation_fraction", "test_fraction")]
    if any(not 0 < value < 1 for value in fraction_values) or abs(sum(fraction_values) - 1.0) > 1e-12:
        raise ValueError("attack split fractions must lie in (0, 1) and sum to 1")
    support_raw = _fields(config["support"], "support", {"minimum_total", "minimum_per_class", "minimum_per_stratum", "minimum_sensitive_label_cell"})
    support_values = [_integer(support_raw[name], f"support.{name}", positive=True) for name in support_raw]
    attributes_raw = config["sensitive_attributes"]
    if not isinstance(attributes_raw, list) or not attributes_raw: raise ValueError("sensitive_attributes must not be empty")
    attributes = tuple(_parse_attribute(value, index) for index, value in enumerate(attributes_raw))
    if len({item.name for item in attributes}) != len(attributes): raise ValueError("sensitive attributes overlap")
    attacks = _fields(config["attacks"], "attacks", {"linear", "nonlinear"})
    linear = _fields(attacks["linear"], "attacks.linear", {"family", "regularization", "inverse_regularization_strengths"})
    if linear["family"] != "logistic_regression" or linear["regularization"] != "l2": raise ValueError("linear attack is not frozen")
    strengths = linear["inverse_regularization_strengths"]
    if not isinstance(strengths, list) or not strengths or any(float(value) <= 0 for value in strengths): raise ValueError("linear regularization search space must be positive")
    nonlinear = _fields(attacks["nonlinear"], "attacks.nonlinear", {"family", "hidden_units", "maximum_parameters", "epochs", "tuning_budget"})
    if nonlinear["family"] != "one_hidden_layer_mlp": raise ValueError("nonlinear architecture must have one hidden layer")
    if not isinstance(nonlinear["hidden_units"], list) or not nonlinear["hidden_units"] or any(_integer(v, "attacks.nonlinear.hidden_units[]", positive=True) < 1 for v in nonlinear["hidden_units"]): raise ValueError("hidden_units must be positive")
    for name in ("maximum_parameters", "epochs", "tuning_budget"): _integer(nonlinear[name], f"attacks.nonlinear.{name}", positive=True)
    imbalance = _fields(config["imbalance"], "imbalance", {"class_weighting", "secondary_one_to_one"})
    if imbalance["class_weighting"] != "inverse_frequency_from_attack_train": raise ValueError("class weighting is not frozen")
    secondary = _fields(imbalance["secondary_one_to_one"], "imbalance.secondary_one_to_one", {"enabled", "policy", "seeds"})
    if not isinstance(secondary["enabled"], bool) or secondary["policy"] != "deterministic_downsample_without_replacement": raise ValueError("secondary 1:1 policy is not frozen")
    if not isinstance(secondary["seeds"], list) or not secondary["seeds"]: raise ValueError("secondary 1:1 seeds must be frozen")
    for seed in secondary["seeds"]: _integer(seed, "imbalance.secondary_one_to_one.seeds[]")
    bootstrap = _fields(config["bootstrap"], "bootstrap", {"method", "replicates", "confidence_level", "seed"})
    if bootstrap["method"] != "stratified_user_percentile" or not 0 < float(bootstrap["confidence_level"]) < 1: raise ValueError("bootstrap method or confidence level is invalid")
    _integer(bootstrap["replicates"], "bootstrap.replicates", positive=True); _integer(bootstrap["seed"], "bootstrap.seed")
    reporting = _fields(config["reporting"], "reporting", {"primary_endpoints", "aggregate_selection", "aggregate_winner"})
    endpoints = _fields(reporting["primary_endpoints"], "reporting.primary_endpoints", {"membership", "sensitive_attribute"})
    if endpoints != {"membership": "test_roc_auc", "sensitive_attribute": "test_macro_f1"} or reporting["aggregate_selection"] != "prohibited" or reporting["aggregate_winner"] is not None:
        raise ValueError("primary endpoints and prohibited aggregate selection must remain frozen")
    return PrivacyAuditConfig(PRIVACY_CONFIG_SCHEMA, _integer(seeds["audit"], "seeds.audit"), _integer(seeds["split"], "seeds.split"), membership,
        _ordered_unique_text(features["cutoff_order"], "features.cutoff_order"), _ordered_unique_text(features["component_order"], "features.component_order"), features["construction"], features["missingness"], tuple(variables),
        SplitConfig(*fraction_values), SupportConfig(*support_values), attributes, attacks, imbalance, bootstrap, reporting)


def load_privacy_config(path: str | Path) -> PrivacyAuditConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_privacy_config(yaml.safe_load(handle))
