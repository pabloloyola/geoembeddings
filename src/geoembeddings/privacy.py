"""Fail-closed input authentication and label handling for privacy evaluation.

Protected labels are read only by :func:`load_protected_labels`, after public
representation evidence has authenticated.  The loader accepts only a
canonical, evaluator-resolved truth path and returns an opaque capability plus
aggregate disclosure-safe metadata; it never serializes user-level labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .artifact_index import SCHEMA_VERSION as EVIDENCE_INDEX_SCHEMA_VERSION
from .contract import TRUTH_FILES
from .io import read_json, sha256_file
from .layout import PairLayout
from .representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION, load_embedding_export
from .runtime_metadata import RuntimeMetadata


FACTORIZATION_INDEX_SCHEMA_VERSION = "geoembeddings-factorization-evidence-index/1.0"
PRIVACY_INPUT_SCHEMA_VERSION = "geoembeddings-privacy-input/1.0"
SELECTION_ROLE = "diagnostic_control"
BASELINE_CHECKPOINT_IDENTITY = "not_applicable"
PRIVACY_POPULATION_SCHEMA_VERSION = "geoembeddings-privacy-population/1.0"
PROTECTED_LABEL_SCHEMA_VERSION = "geoembeddings-protected-labels/1.0"
PRIVACY_AUDIT_SCHEMA_VERSION = "geoembeddings-privacy-audit/1.0"

PRIVACY_AUDIT_SECTIONS = (
    "threat_model", "inputs", "lineage", "splits", "membership_population",
    "sensitive_attributes", "attacks", "membership_metrics",
    "sensitive_probe_metrics", "utility_privacy_axes", "coverage", "exclusions",
    "selection", "limitations", "command", "timestamps", "runtime_metadata",
)


def _validate_privacy_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable report envelope and prohibited conclusions."""
    value = dict(report)
    if value.get("schema_version") != PRIVACY_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"privacy audit schema_version must be {PRIVACY_AUDIT_SCHEMA_VERSION!r}")
    missing = [section for section in PRIVACY_AUDIT_SECTIONS if section not in value]
    if missing:
        raise ValueError(f"privacy audit is missing required sections: {missing}")
    prohibited = {"aggregate_score", "aggregate_privacy_score", "aggregate_utility_score", "aggregate_winner", "winner"}

    def inspect(item: Any, path: str = "report") -> None:
        if isinstance(item, Mapping):
            found = prohibited.intersection(map(str, item))
            if found:
                raise ValueError(f"privacy audit must not emit aggregate scores or winners ({path}: {sorted(found)})")
            for key, child in item.items():
                inspect(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                inspect(child, f"{path}[{index}]")

    inspect(value)
    membership = value["membership_population"]
    if not isinstance(membership, Mapping):
        raise ValueError("membership_population must be an object")
    statistical = membership.get("statistical_baseline")
    if not isinstance(statistical, Mapping) or statistical.get("status") != "not_applicable":
        raise ValueError("statistical baseline membership must be not_applicable")
    # An unsupported analysis is absence of evidence, never a failed or numeric result.
    for name, result in membership.items():
        if name == "statistical_baseline" or not isinstance(result, Mapping):
            continue
        if result.get("supported") is False and result.get("status") != "unavailable":
            raise ValueError(f"unsupported membership analysis {name!r} must be unavailable")
    selection = value["selection"]
    conclusion = selection.get("selection_dependent_privacy_conclusion") if isinstance(selection, Mapping) else None
    if conclusion != {"status": "unavailable", "reason": "no_selected_candidate"}:
        raise ValueError("selection-dependent privacy conclusion must be unavailable: no_selected_candidate")
    return value


def render_privacy_markdown(report: Mapping[str, Any]) -> str:
    """Render the authoritative JSON content without deriving alternate results."""
    value = _validate_privacy_audit(report)
    # Embedding the complete canonical payload makes omissions and drift between
    # the human and machine views mechanically detectable.
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    return (
        "# R12 privacy audit\n\n"
        "The JSON artifact is authoritative. This document renders the same results. "
        "No aggregate privacy/utility score or winner is reported.\n\n"
        "```json\n" + payload + "\n```\n"
    )


def write_privacy_audit(report: Mapping[str, Any], audit_dir: str | Path, *,
                        overwrite: bool = False) -> tuple[Path, Path]:
    """Atomically publish canonical JSON and Markdown privacy-audit outputs.

    Both complete byte strings are staged before either destination changes.
    On any publication failure, prior regular files are restored and newly
    created output is removed, so callers never observe a partial final pair.
    """
    value = _validate_privacy_audit(report)
    layout = PairLayout.from_path(audit_dir)
    destinations = (layout.privacy_audit_json, layout.privacy_audit_markdown)
    if any(path.exists() for path in destinations):
        if not overwrite:
            raise FileExistsError("Refusing to overwrite immutable privacy audit")
        if any(path.exists() and (path.is_symlink() or not path.is_file()) for path in destinations):
            raise ValueError("--overwrite targets must be regular privacy audit files")
    contents = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        render_privacy_markdown(value),
    )
    destinations[0].parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    originals: list[bytes | None] = [path.read_bytes() if path.exists() else None for path in destinations]
    try:
        for destination, content in zip(destinations, contents):
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            temporary = Path(name); staged.append(temporary)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content); handle.flush(); os.fsync(handle.fileno())
        for temporary, destination in zip(staged, destinations):
            os.replace(temporary, destination)
    except BaseException:
        for destination, original in zip(destinations, originals):
            try:
                if original is None:
                    destination.unlink(missing_ok=True)
                else:
                    destination.write_bytes(original)
            except OSError:
                pass
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return destinations


@dataclass(frozen=True)
class ProtectedAttributeDeclaration:
    """Predeclared interpretation of one canonical simulator attribute."""

    name: str
    source_field: str
    privacy_rationale: str
    visibility: Literal["evaluator_only", "public_or_observed"]
    derivation_version: str
    derivation_method: Literal["train_quantile_bins", "observed_identity"]
    labels: tuple[str, ...]
    rare_class_mapping: tuple[tuple[str, str], ...] = ()


# This allowlist is intentionally code-versioned and small. Adding a field is a
# privacy-contract change, not something a caller may do through configuration.
SUPPORTED_PROTECTED_ATTRIBUTES: Mapping[str, ProtectedAttributeDeclaration] = {
    "price_sensitivity_group": ProtectedAttributeDeclaration(
        "price_sensitivity_group", "price_sensitivity",
        "A persistent synthetic economic-preference trait may be recoverable from an embedding.",
        "evaluator_only", "price-sensitivity-train-tertiles/1.0", "train_quantile_bins",
        ("low", "medium", "high"),
    ),
    "family_orientation_group": ProtectedAttributeDeclaration(
        "family_orientation_group", "family_orientation",
        "A persistent synthetic household-related trait may reveal modeled family orientation.",
        "evaluator_only", "family-orientation-train-tertiles/1.0", "train_quantile_bins",
        ("low", "medium", "high"),
    ),
    # These declarations document why tempting probes are not protected-label
    # probes: their values are already explicit in users_observed.csv.gz.
    "age_group": ProtectedAttributeDeclaration(
        "age_group", "age_group", "Synthetic age band is demographic information.",
        "public_or_observed", "age-group-observed-identity/1.0", "observed_identity", (),
    ),
    "household_type": ProtectedAttributeDeclaration(
        "household_type", "household_type", "Synthetic household type is demographic information.",
        "public_or_observed", "household-type-observed-identity/1.0", "observed_identity", (),
    ),
}

PROHIBITED_PROTECTED_ATTRIBUTE_TOKENS = (
    "latitude", "longitude", "coordinate", "user_id", "object_id", "session_id",
    "episode_id", "decision_id", "identifier", "chosen", "utility", "text",
    "intersection", "sparse_cell",
)


@dataclass(frozen=True)
class ProtectedAttributeSummary:
    """Aggregate-only label derivation result safe for a privacy report."""

    name: str
    status: Literal["available", "excluded"]
    reason: str | None
    privacy_rationale: str
    visibility: str
    derivation_version: str
    derivation_method: str
    fit_split: str | None
    labels: tuple[str, ...]
    bin_boundaries: tuple[float, ...]
    counts: tuple[tuple[str, int], ...]
    eligible_count: int
    missing_count: int
    unsupported_count: int


class ProtectedLabelBundle:
    """Opaque in-process labels and their aggregate, non-reconstructable summary.

    The private mapping deliberately has no public accessor, iterator, repr, or
    serialization method. Evaluator code can pass the bundle directly to
    :func:`run_protected_attribute_attacks`.
    """

    __slots__ = ("schema_version", "summaries", "_labels")

    def __init__(self, summaries: Sequence[ProtectedAttributeSummary], labels: Mapping[str, Mapping[str, str]]) -> None:
        self.schema_version = PROTECTED_LABEL_SCHEMA_VERSION
        self.summaries = tuple(summaries)
        self._labels = {name: dict(values) for name, values in labels.items()}

    def __repr__(self) -> str:
        return f"ProtectedLabelBundle(schema_version={self.schema_version!r}, attributes={len(self.summaries)})"


def _canonical_hash(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrivacyInput:
    """Declared, non-protected files and identities for one diagnostic control."""

    name: str
    kind: Literal["statistical_baseline", "learned"]
    export_path: Path
    prepared_metadata_path: Path
    utility_report_path: Path
    selection_role: str
    parameter_count: int
    eligible_users: tuple[str, ...]
    utility_report_users: tuple[str, ...]
    checkpoint_path: Path | None = None
    checkpoint_identity: str | None = None
    model_variant: str | None = None


@dataclass(frozen=True)
class PrivacyInputIdentity:
    """Authenticated identity safe to copy into a privacy report."""

    schema_version: str
    name: str
    kind: str
    selection_role: str
    export_path: str
    export_bytes: int
    export_sha256: str
    export_schema: str
    model_variant: str
    component_order: tuple[str, ...]
    component_dimensions: tuple[int, ...]
    export_keys_sha256: str
    cutoffs: tuple[str, ...]
    checkpoint_identity: str
    preparation_metadata_sha256: str
    preparation_definition_sha256: str
    dataset_contract: str
    categorical_fields: tuple[str, ...]
    continuous_fields: tuple[str, ...]
    observed_source_hashes: tuple[tuple[str, str], ...]
    parameter_count: int
    eligible_users_sha256: str
    utility_report_users_sha256: str


@dataclass(frozen=True)
class AuthenticatedPrivacyInputs:
    """Evidence decision and the fully authenticated control identities."""

    evidence_index_sha256: str
    evidence_index_schema: str
    evidence_task_id: str
    t2_7_decision: str
    inputs: tuple[PrivacyInputIdentity, ...]
    runtime_metadata: RuntimeMetadata | None = None


@dataclass(frozen=True)
class PrivacyPopulationRecord:
    """One attack example: all of a user's vectors, masks, and public strata."""

    user_id: str
    membership: bool
    split: Literal["train", "validation", "test"]
    vector_features: tuple[float, ...]
    missing_cutoff_mask: tuple[int, ...]
    missing_component_mask: tuple[int, ...]
    provenance_covariates: tuple[float, ...]
    matching_stratum: tuple[int, ...]


@dataclass(frozen=True)
class PrivacyPopulation:
    """Deterministic user population, or a fail-closed unavailable decision."""

    schema_version: str
    status: Literal["available", "unavailable"]
    reason: str | None
    target_model_lineage: str
    cutoff_order: tuple[str, ...]
    component_order: tuple[str, ...]
    vector_feature_names: tuple[str, ...]
    provenance_feature_names: tuple[str, ...]
    records: tuple[PrivacyPopulationRecord, ...]
    excluded_users: tuple[str, ...]
    user_set_hashes: tuple[tuple[str, str], ...]

    def user_set_hash(self, name: str) -> str:
        try:
            return dict(self.user_set_hashes)[name]
        except KeyError as exc:
            raise KeyError(f"Unknown privacy population set {name!r}") from exc


def _defined(value: float | None, reason: str | None = None) -> dict[str, Any]:
    """JSON-safe metric value which never disguises an undefined result as zero."""
    return {"value": None if value is None else float(value), "undefined_reason": reason}


def _binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    support = int(y.size); positives = int(y.sum()); negatives = support - positives
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = positives - tp; tn = negatives - fp
    reason = None if positives and negatives else "test_split_lacks_both_classes"
    if reason:
        auc = ap = None
    else:
        order = np.argsort(score, kind="stable")
        ranks = np.empty(support, dtype=float); ranks[order] = np.arange(1, support + 1)
        # Average tied ranks, which makes constant-score controls exactly 0.5.
        for value in np.unique(score):
            mask = score == value; ranks[mask] = ranks[mask].mean()
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        ap = 0.0
        previous_recall = 0.0
        for cutoff in np.sort(np.unique(score))[::-1]:
            selected = score >= cutoff
            selected_tp = int(np.sum((y == 1) & selected))
            recall_at_cutoff = selected_tp / positives
            precision_at_cutoff = selected_tp / int(selected.sum())
            ap += (recall_at_cutoff - previous_recall) * precision_at_cutoff
            previous_recall = recall_at_cutoff
    recall = None if not positives else tp / positives
    precision = None if not (tp + fp) else tp / (tp + fp)
    tpr = None if not positives else tp / positives; tnr = None if not negatives else tn / negatives
    balanced = None if tpr is None or tnr is None else (tpr + tnr) / 2
    return {"roc_auc": _defined(auc, reason), "average_precision": _defined(ap, reason),
            "balanced_accuracy": _defined(balanced, reason),
            "precision": _defined(precision, "no_predicted_positive" if precision is None else None),
            "recall": _defined(recall, "no_positive_support" if recall is None else None),
            "threshold": float(threshold), "threshold_rule": "score_greater_than_or_equal_to_threshold",
            "natural_prevalence": positives / support if support else None,
            "support": {"total": support, "positive": positives, "negative": negatives}}


def _multiclass_metrics(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    pred = classes[np.argmax(probabilities, axis=1)]
    recalls: list[float] = []; f1s: list[float] = []; aucs: list[float] = []
    per_class: dict[str, int] = {}
    for column, label in enumerate(classes):
        actual = y == label; guessed = pred == label; count = int(actual.sum()); per_class[str(label)] = count
        tp = int(np.sum(actual & guessed)); fp = int(np.sum(~actual & guessed)); fn = count - tp
        if count: recalls.append(tp / count)
        denominator = 2 * tp + fp + fn
        if denominator: f1s.append(2 * tp / denominator)
        one = _binary_metrics(actual.astype(int), probabilities[:, column], 0.5)["roc_auc"]["value"]
        if one is not None: aucs.append(one)
    absent = "test_split_has_fewer_than_two_supported_classes"
    result = {"macro_f1": _defined(float(np.mean(f1s)) if len(f1s) == len(classes) else None, absent),
              "balanced_accuracy": _defined(float(np.mean(recalls)) if len(recalls) == len(classes) else None, absent),
              "one_vs_rest_macro_auc": _defined(float(np.mean(aucs)) if len(aucs) == len(classes) else None, absent),
              "per_class_support": per_class, "support": int(y.size)}
    if len(classes) == 2:
        result.update({k: v for k, v in _binary_metrics((y == classes[1]).astype(int), probabilities[:, 1], 0.5).items()
                       if k in {"roc_auc", "average_precision"}})
    return result


def seeded_stratified_user_bootstrap(
    user_ids: Sequence[str], labels: Sequence[Any], values: np.ndarray, *,
    strata: Sequence[Any] | None = None, metric: Any, replicates: int,
    seed: int, confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a user-level metric while keeping each user's row intact.

    ``values`` may contain a score, a probability vector, or an entire feature
    row.  The metric callable receives ``(sampled_labels, sampled_values)``.
    Sampling is independently performed within the supplied strata; class is
    always included in the effective stratum so the natural test-set class
    counts are preserved.  Canonical user ordering makes the result independent
    of the caller's row ordering.
    """
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("bootstrap confidence_level must be between zero and one")
    users = np.asarray(list(map(str, user_ids)), dtype=str)
    y = np.asarray(labels)
    value = np.asarray(values)
    if len(users) != len(y) or len(users) != len(value):
        raise ValueError("bootstrap users, labels, and values must have equal length")
    if len(set(users.tolist())) != len(users):
        raise ValueError("bootstrap requires exactly one joined row per user")
    supplied = np.asarray(["all"] * len(users), dtype=object) if strata is None else np.asarray(strata, dtype=object)
    if len(supplied) != len(users):
        raise ValueError("bootstrap strata must have one value per user")
    order = np.argsort(users, kind="stable")
    users, y, value, supplied = users[order], y[order], value[order], supplied[order]
    # repr handles tuple-valued matching strata without relying on NumPy's
    # sometimes surprising multidimensional coercion of a list of tuples.
    effective = np.asarray([_canonical_hash([str(label), repr(stratum)])
                            for label, stratum in zip(y, supplied)], dtype=str)
    groups = [np.flatnonzero(effective == name) for name in sorted(set(effective.tolist()))]
    expected_classes = set(y.tolist())
    try:
        point_estimate = float(metric(y, value)) if len(expected_classes) >= 2 else None
        if point_estimate is not None and not np.isfinite(point_estimate):
            point_estimate = None
    except (ArithmeticError, ValueError, IndexError, TypeError):
        point_estimate = None
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    degenerate = excluded = 0
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        if set(y[sampled].tolist()) != expected_classes or len(expected_classes) < 2:
            degenerate += 1
            continue
        try:
            estimate = float(metric(y[sampled], value[sampled]))
        except (ArithmeticError, ValueError, IndexError, TypeError):
            excluded += 1
            continue
        if not np.isfinite(estimate):
            excluded += 1
            continue
        estimates.append(estimate)
    alpha = (1.0 - confidence_level) / 2.0
    interval = ([float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))]
                if estimates else None)
    return {
        "method": "stratified_user_percentile", "seed": int(seed),
        "confidence_level": float(confidence_level), "replicate_count": int(replicates),
        "requested_replicates": int(replicates), "successful_replicates": len(estimates),
        "degenerate_replicates": degenerate, "excluded_replicates": excluded,
        "point_estimate": point_estimate, "interval": interval,
        "undefined_reason": None if interval is not None else "no_successful_bootstrap_replicates",
    }


def seeded_matched_representation_delta_bootstrap(
    user_ids: Sequence[str], labels: Sequence[Any], control_values: Mapping[str, np.ndarray], *,
    reference: str, strata: Sequence[Any] | None = None, metric: Any,
    replicates: int, seed: int, confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute paired control-minus-reference intervals from identical users."""
    if reference not in control_values:
        raise ValueError(f"Unknown reference control {reference!r}")
    lengths = {name: len(np.asarray(values)) for name, values in control_values.items()}
    if any(length != len(user_ids) for length in lengths.values()):
        raise ValueError("matched controls must contain the same ordered users")
    stacked = np.stack([np.asarray(control_values[name]) for name in control_values], axis=1)
    names = tuple(control_values)
    result: dict[str, Any] = {}
    for name in names:
        if name == reference:
            continue
        left, right = names.index(name), names.index(reference)
        def delta(sample_y: np.ndarray, sample_values: np.ndarray, a: int = left, b: int = right) -> float:
            return float(metric(sample_y, sample_values[:, a]) - metric(sample_y, sample_values[:, b]))
        result[name] = seeded_stratified_user_bootstrap(
            user_ids, labels, stacked, strata=strata, metric=delta, replicates=replicates,
            seed=seed, confidence_level=confidence_level,
        )
    return {"reference": reference, "deltas": result}


def _fit_preprocessor(x: np.ndarray, pca_components: int | None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    median = np.nanmedian(x, axis=0); median = np.where(np.isfinite(median), median, 0.0)
    clean = np.where(np.isfinite(x), x, median)
    mean = clean.mean(axis=0); scale = clean.std(axis=0); scale[scale == 0] = 1
    standardized = (clean - mean) / scale
    basis = np.empty((standardized.shape[1], 0))
    if pca_components is not None and 0 < pca_components < standardized.shape[1]:
        _, _, vt = np.linalg.svd(standardized, full_matrices=False); basis = vt[:pca_components].T
        standardized = standardized @ basis
    return {"imputation": median, "mean": mean, "scale": scale, "pca_basis": basis}, standardized


def _transform(x: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    clean = np.where(np.isfinite(x), x, state["imputation"])
    value = (clean - state["mean"]) / state["scale"]
    return value @ state["pca_basis"] if state["pca_basis"].shape[1] else value


def _fit_softmax(x: np.ndarray, y: np.ndarray, classes: np.ndarray, c: float,
                 weights: np.ndarray, seed: int, epochs: int = 500) -> tuple[np.ndarray, np.ndarray]:
    del seed  # initialization is deliberately deterministic and data-independent
    targets = (y[:, None] == classes[None, :]).astype(float)
    coef = np.zeros((x.shape[1], len(classes))); intercept = np.zeros(len(classes))
    rate = 0.2 / max(1.0, np.sqrt(x.shape[1]))
    for step in range(epochs):
        logits = x @ coef + intercept; logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
        error = (probability - targets) * weights[:, None]
        coef -= rate / np.sqrt(step + 1) * (x.T @ error / weights.sum() + coef / (c * len(y)))
        intercept -= rate / np.sqrt(step + 1) * error.sum(axis=0) / weights.sum()
    return coef, intercept


def _predict_softmax(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    logits = x @ model[0] + model[1]; logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits); return result / result.sum(axis=1, keepdims=True)


def run_privacy_attacks(
    population: PrivacyPopulation, labels_by_user: Mapping[str, Any] | None, *,
    inverse_regularization_strengths: Sequence[float], hidden_units: Sequence[int],
    nonlinear_epochs: int, nonlinear_tuning_budget: int, maximum_parameters: int,
    seed: int, task: Literal["membership", "sensitive_attribute"] = "membership",
    pca_components: int | None = None, bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen attack families with train/validation/test isolation.

    All preprocessing and class weights are estimated from attack-train.  The
    validation split alone chooses regularization, architecture, and (for
    membership) the decision threshold; test is transformed and scored once.
    """
    if population.status != "available": return {"status": "unavailable", "reason": population.reason, "attacks": {}}
    records = list(population.records)
    labels = np.asarray([int(r.membership) if labels_by_user is None else labels_by_user[r.user_id] for r in records])
    indices = {name: np.asarray([i for i, r in enumerate(records) if r.split == name]) for name in ("train", "validation", "test")}
    classes = np.unique(labels[indices["train"]]) if len(indices["train"]) else np.asarray([])
    if (any(not len(value) for value in indices.values()) or len(classes) < 2
            or not set(np.unique(labels[indices["validation"]])).issubset(set(classes))
            or not set(np.unique(labels[indices["test"]])).issubset(set(classes))):
        return {"status": "unavailable", "reason": "attack_split_support_inadequate", "attacks": {}}
    vector = np.asarray([r.vector_features + r.missing_cutoff_mask + r.missing_component_mask for r in records], float)
    provenance = np.asarray([r.provenance_covariates for r in records], float)
    features = {"provenance_logistic": provenance, "vector_logistic": vector,
                "vector_plus_provenance_logistic": np.column_stack((vector, provenance)),
                "bounded_nonlinear": np.column_stack((vector, provenance))}
    train_counts = {label: int(np.sum(labels[indices["train"]] == label)) for label in classes}
    row_weights = np.asarray([len(indices["train"]) / (len(classes) * train_counts[label]) for label in labels[indices["train"]]])

    def quality(y: np.ndarray, probability: np.ndarray) -> float:
        if task == "membership":
            return _binary_metrics(y.astype(int), probability[:, 1], 0.5)["roc_auc"]["value"] or -1.0
        value = _multiclass_metrics(y, probability, classes)["macro_f1"]["value"]; return value if value is not None else -1.0

    output: dict[str, Any] = {}
    test = indices["test"]; validation = indices["validation"]
    test_users = [records[i].user_id for i in test]
    test_strata = [records[i].matching_stratum for i in test]

    def add_uncertainty(metrics: dict[str, Any], probability: np.ndarray) -> None:
        if bootstrap is None:
            return
        if task == "membership":
            test_labels = (labels[test] == classes[-1]).astype(int)
            primary_name = "roc_auc"
            primary_metric = lambda yy, vv: _binary_metrics(yy.astype(int), vv, 0.5)["roc_auc"]["value"]
            primary_values = probability[:, 1] if probability.ndim == 2 else probability
        else:
            test_labels = labels[test]
            primary_name = "macro_f1"
            primary_metric = lambda yy, vv: _multiclass_metrics(yy, vv, classes)["macro_f1"]["value"]
            primary_values = probability
        metrics[primary_name]["bootstrap"] = seeded_stratified_user_bootstrap(
            test_users, test_labels, primary_values, strata=test_strata, metric=primary_metric,
            replicates=int(bootstrap["replicates"]), seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )
    prevalence = float(np.mean(labels[indices["train"]] == classes[-1]))
    random_scores = np.asarray([int(_seeded_user_hash(seed, population.target_model_lineage, records[i].user_id, "random-control"), 16) / 2**256 for i in test])
    if task == "membership":
        output["deterministic_random"] = _binary_metrics((labels[test] == classes[-1]).astype(int), random_scores, 0.5)
        output["majority_base_rate"] = _binary_metrics((labels[test] == classes[-1]).astype(int), np.full(len(test), prevalence), 0.5)
        add_uncertainty(output["deterministic_random"], random_scores)
        add_uncertainty(output["majority_base_rate"], np.full(len(test), prevalence))
    else:
        priors = np.asarray([train_counts[c] for c in classes], float); priors /= priors.sum()
        output["majority_prior"] = _multiclass_metrics(labels[test], np.tile(priors, (len(test), 1)), classes)
        random_probability = np.asarray([[int(_seeded_user_hash(seed + column, population.target_model_lineage,
                                                                  records[i].user_id, "random-control"), 16) / 2**256
                                          for column in range(len(classes))] for i in test])
        random_probability /= random_probability.sum(axis=1, keepdims=True)
        output["deterministic_random"] = _multiclass_metrics(labels[test], random_probability, classes)
        add_uncertainty(output["majority_prior"], np.tile(priors, (len(test), 1)))
        add_uncertainty(output["deterministic_random"], random_probability)
    for name, raw in features.items():
        state, train_x = _fit_preprocessor(raw[indices["train"]], pca_components)
        val_x = _transform(raw[validation], state); test_x = _transform(raw[test], state)
        candidates: list[tuple[float, dict[str, Any], np.ndarray, Any]] = []
        if name != "bounded_nonlinear":
            for c in inverse_regularization_strengths:
                model = _fit_softmax(train_x, labels[indices["train"]], classes, float(c), row_weights, seed)
                val_probability = _predict_softmax(val_x, model)
                candidates.append((quality(labels[validation], val_probability), {"C": float(c)}, val_probability, model))
        else:
            budget = 0
            for hidden in hidden_units:
                for c in inverse_regularization_strengths:
                    if budget >= nonlinear_tuning_budget: break
                    parameters = (train_x.shape[1] + 1) * hidden + (hidden + 1) * len(classes)
                    if parameters > maximum_parameters: continue
                    torch.manual_seed(seed + budget); model = torch.nn.Sequential(torch.nn.Linear(train_x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, len(classes)))
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1 / float(c))
                    tx = torch.tensor(train_x, dtype=torch.float32); ty = torch.tensor(np.searchsorted(classes, labels[indices["train"]]), dtype=torch.long)
                    tw = torch.tensor(row_weights, dtype=torch.float32)
                    for _ in range(nonlinear_epochs):
                        optimizer.zero_grad(); losses = torch.nn.functional.cross_entropy(model(tx), ty, reduction="none"); (losses * tw).mean().backward(); optimizer.step()
                    with torch.no_grad():
                        vp = torch.softmax(model(torch.tensor(val_x, dtype=torch.float32)), 1).numpy()
                    candidates.append((quality(labels[validation], vp), {"hidden_units": hidden, "C": float(c), "parameter_count": parameters}, vp, model)); budget += 1
                if budget >= nonlinear_tuning_budget: break
        if not candidates:
            output[name] = {"status": "unavailable", "reason": "no_model_within_frozen_parameter_budget"}; continue
        _, selected, validation_probability, selected_model = max(candidates, key=lambda item: (item[0], -list(item[1].values())[0]))
        if name == "bounded_nonlinear":
            with torch.no_grad():
                probability = torch.softmax(selected_model(torch.tensor(test_x, dtype=torch.float32)), 1).numpy()
        else:
            probability = _predict_softmax(test_x, selected_model)
        if task == "membership":
            # Threshold is selected solely from validation predictions of the frozen selected configuration.
            # Refit its validation predictions without ever consulting test outcomes.
            score = probability[:, 1]
            validation_scores = validation_probability[:, 1]
            thresholds = np.unique(np.r_[0.0, 0.5, 1.0, validation_scores])
            threshold = max(thresholds, key=lambda value: (
                _binary_metrics((labels[validation] == classes[-1]).astype(int), validation_scores, float(value))["balanced_accuracy"]["value"] or -1.0,
                -abs(float(value) - 0.5), -float(value)))
            output[name] = {**_binary_metrics((labels[test] == classes[-1]).astype(int), score, threshold), "selected_hyperparameters": selected}
        else:
            output[name] = {**_multiclass_metrics(labels[test], probability, classes), "selected_hyperparameters": selected}
        add_uncertainty(output[name], probability)
        output[name]["preprocessing_fit_split"] = "train"; output[name]["selection_split"] = "validation"
        output[name]["test_scoring_passes"] = 1; output[name]["pca_components"] = pca_components
    return {"status": "available", "task": task, "class_order": classes.tolist(), "attacks": output}


def run_privacy_attacks_from_config(
    population: PrivacyPopulation, labels_by_user: Mapping[str, Any] | None,
    config: Any, *, task: Literal["membership", "sensitive_attribute"] = "membership",
) -> dict[str, Any]:
    """Execute attacks using only the architecture and search budget frozen in YAML."""
    linear = config.attacks["linear"]; nonlinear = config.attacks["nonlinear"]
    reduction = config.attacks.get("dimensionality_reduction", {})
    return run_privacy_attacks(
        population, labels_by_user,
        inverse_regularization_strengths=linear["inverse_regularization_strengths"],
        hidden_units=nonlinear["hidden_units"], nonlinear_epochs=nonlinear["epochs"],
        nonlinear_tuning_budget=nonlinear["tuning_budget"],
        maximum_parameters=nonlinear["maximum_parameters"], seed=config.audit_seed,
        task=task, pca_components=reduction.get("components") if reduction.get("enabled", False) else None,
        bootstrap=config.bootstrap,
    )


def _user_set_hash(users: Sequence[str]) -> str:
    """Hash a set rather than its caller-provided iteration order."""
    values = sorted(map(str, users))
    if len(values) != len(set(values)):
        raise ValueError("User sets must not contain duplicate identities")
    return _canonical_hash(values)


def _seeded_user_hash(seed: int, lineage: str, user_id: str, purpose: str) -> str:
    return _canonical_hash({"version": "privacy-user-hash/1.0", "seed": seed,
                            "target_model_lineage": lineage, "user_id": user_id,
                            "purpose": purpose})


def _bin(value: float, boundaries: Sequence[float], name: str) -> int:
    edges = tuple(float(item) for item in boundaries)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"Matching boundaries for {name!r} are not strictly increasing")
    if not np.isfinite(value):
        raise ValueError(f"Matching covariate {name!r} is not finite")
    # Frozen strata are half-open, except that the final right edge is inclusive.
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        if left <= value < right or (index == len(edges) - 2 and value == right):
            return index
    raise ValueError(f"Matching covariate {name!r}={value} is outside frozen boundaries")


def construct_privacy_population(
    export: LoadedEmbeddingExport | str | Path,
    *,
    target_model_lineage: str,
    membership_by_user: Mapping[str, bool],
    provenance_by_user: Mapping[str, Mapping[str, float]],
    cutoff_order: Sequence[str],
    component_order: Sequence[str],
    matching_boundaries: Mapping[str, Sequence[float]],
    audit_seed: int,
    split_seed: int,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    minimum_total: int = 1,
    minimum_per_class: int = 1,
    minimum_per_stratum: int = 1,
    expected_user_set_hashes: Mapping[str, str] | None = None,
) -> PrivacyPopulation:
    """Build immutable user-level membership examples from a frozen export.

    Membership is supplied once per user for this target lineage.  In
    particular, export cutoffs never create membership labels: a post-training
    event belonging to a training user remains part of that member's one record.
    Matching and splitting use only the explicitly declared public provenance.
    """
    if not target_model_lineage.strip():
        raise ValueError("target_model_lineage must be non-empty")
    loaded = load_embedding_export(export) if isinstance(export, (str, Path)) else export
    cutoffs = tuple(map(str, cutoff_order)); components = tuple(map(str, component_order))
    if not cutoffs or len(cutoffs) != len(set(cutoffs)):
        raise ValueError("cutoff_order must be non-empty and unique")
    if not components or len(components) != len(set(components)):
        raise ValueError("component_order must be non-empty and unique")
    if any(name not in loaded.components for name in components):
        raise ValueError("component_order contains a component absent from the export")
    provenance_names = tuple(matching_boundaries)
    if not provenance_names:
        raise ValueError("At least one frozen public matching covariate is required")
    if set(membership_by_user) != set(provenance_by_user):
        raise ValueError("Membership and frozen-provenance user sets differ")
    if any(not isinstance(value, (bool, np.bool_)) for value in membership_by_user.values()):
        raise ValueError("Membership must be one boolean per (target_model_lineage, user_id)")
    fractions = tuple(float(value) for value in split_fractions)
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must be three positive values summing to one")

    export_users = loaded.arrays["user_id"].astype(str)
    export_cutoffs = loaded.arrays["cutoff"].astype(str)
    row_by_key: dict[tuple[str, str], int] = {}
    for index, key in enumerate(zip(export_users.tolist(), export_cutoffs.tolist())):
        if key in row_by_key:
            raise ValueError(f"Duplicate export user/cutoff key: {key!r}")
        if key[1] not in cutoffs:
            raise ValueError(f"Post-hoc cutoff {key[1]!r} is absent from the declared cutoff order")
        row_by_key[key] = index
    unknown_export_users = set(export_users) - set(membership_by_user)
    if unknown_export_users:
        raise ValueError(f"Export contains users without frozen membership: {sorted(unknown_export_users)}")

    eligible = sorted(map(str, membership_by_user))
    excluded: list[str] = []
    candidates: dict[str, dict[str, Any]] = {}
    feature_names: list[str] = []
    for cutoff in cutoffs:
        for component in components:
            feature_names.extend(f"vector:{cutoff}:{component}:{i}" for i in range(loaded.components[component].shape[1]))
    for user in eligible:
        provenance = provenance_by_user[user]
        if set(provenance) != set(provenance_names):
            raise ValueError(f"Public provenance schema mismatch for user {user!r}")
        try:
            provenance_values = tuple(float(provenance[name]) for name in provenance_names)
            stratum = tuple(_bin(value, matching_boundaries[name], name)
                            for name, value in zip(provenance_names, provenance_values))
        except ValueError:
            excluded.append(user)
            continue
        values: list[float] = []; cutoff_mask: list[int] = []; component_mask: list[int] = []
        for cutoff in cutoffs:
            row = row_by_key.get((user, cutoff)); cutoff_mask.append(int(row is None))
            for component in components:
                missing = row is None
                component_mask.append(int(missing))
                dimension = loaded.components[component].shape[1]
                values.extend(([0.0] * dimension) if missing else loaded.components[component][row].astype(float).tolist())
        candidates[user] = {"membership": bool(membership_by_user[user]), "vector": tuple(values),
                            "cutoff_mask": tuple(cutoff_mask), "component_mask": tuple(component_mask),
                            "provenance": provenance_values, "stratum": stratum}

    # Pair within common-support cells; deterministic order gives matching
    # without replacement and never consults vectors or protected outcomes.
    cells: dict[tuple[int, ...], dict[bool, list[str]]] = {}
    for user, value in candidates.items():
        cells.setdefault(value["stratum"], {False: [], True: []})[value["membership"]].append(user)
    matched: list[str] = []
    for stratum in sorted(cells):
        cell = cells[stratum]
        members = sorted(cell[True], key=lambda u: _seeded_user_hash(audit_seed, target_model_lineage, u, "match"))
        nonmembers = sorted(cell[False], key=lambda u: _seeded_user_hash(audit_seed, target_model_lineage, u, "match"))
        count = min(len(members), len(nonmembers))
        if count < minimum_per_stratum:
            excluded.extend(members); excluded.extend(nonmembers)
        else:
            matched.extend(members[:count]); matched.extend(nonmembers[:count])
            excluded.extend(members[count:]); excluded.extend(nonmembers[count:])
    matched = sorted(matched)
    class_counts = {value: sum(candidates[user]["membership"] == value for user in matched) for value in (False, True)}

    thresholds = (fractions[0], fractions[0] + fractions[1])
    split_users: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    records: list[PrivacyPopulationRecord] = []
    for user in matched:
        # Hashing the complete canonical identity makes assignment independent
        # of export row order and keeps the whole user in exactly one split.
        unit = int(_seeded_user_hash(split_seed, target_model_lineage, user, "attack-split"), 16) / 2**256
        split = "train" if unit < thresholds[0] else "validation" if unit < thresholds[1] else "test"
        split_users[split].append(user)
        value = candidates[user]
        records.append(PrivacyPopulationRecord(user, value["membership"], split, value["vector"],
                                               value["cutoff_mask"], value["component_mask"],
                                               value["provenance"], value["stratum"]))
    sets = {"eligible": eligible, "matched": matched, "excluded": sorted(set(excluded)), **split_users}
    hashes = tuple((name, _user_set_hash(users)) for name, users in sets.items())
    if set(split_users["train"]) & set(split_users["validation"]) or set(split_users["train"]) & set(split_users["test"]) or set(split_users["validation"]) & set(split_users["test"]):
        raise ValueError("Attack user splits overlap")
    if set().union(*map(set, split_users.values())) != set(matched):
        raise ValueError("Attack split assignment changed the matched user set")
    if expected_user_set_hashes is not None:
        actual = dict(hashes)
        for name, expected in expected_user_set_hashes.items():
            if name not in actual or actual[name] != expected:
                raise ValueError(f"Post-hoc {name!r} user-set change detected")
    reason = None
    if not class_counts[False] or not class_counts[True]: reason = "membership_classes_inadequate"
    elif len(matched) < minimum_total or min(class_counts.values()) < minimum_per_class: reason = "common_support_inadequate"
    status: Literal["available", "unavailable"] = "unavailable" if reason else "available"
    return PrivacyPopulation(PRIVACY_POPULATION_SCHEMA_VERSION, status, reason, target_model_lineage,
                             cutoffs, components, tuple(feature_names), provenance_names,
                             tuple(records), tuple(sorted(set(excluded))), hashes)


# Public spelling retained for callers that describe this phase as population building.
build_privacy_population = construct_privacy_population


def _indexed_artifact(index: dict[str, Any], path: Path) -> dict[str, Any]:
    artifacts = index.get("artifacts")
    if isinstance(artifacts, dict):
        candidates = (str(path), str(path.resolve()))
        for candidate in candidates:
            if candidate in artifacts:
                return artifacts[candidate]
    # The general evidence index stores artifacts in named lists.
    required = index.get("required_artifacts", {})
    for entries in required.values() if isinstance(required, dict) else ():
        for artifact in entries:
            identifier = artifact.get("identifier")
            if identifier in {str(path), str(path.resolve())}:
                return artifact
    raise ValueError(f"Export is absent from evidence index: {path}")


def _require_indexed_bytes(index: dict[str, Any], path: Path) -> None:
    artifact = _indexed_artifact(index, path)
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if int(artifact.get("bytes", actual_bytes)) != actual_bytes:
        raise ValueError(f"Indexed byte count mismatch for {path}")
    if artifact.get("sha256") != actual_hash:
        raise ValueError(f"Indexed SHA-256 mismatch for {path}")


def _source_hashes(metadata: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    sources = metadata.get("source_files")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Preparation metadata has no observed-source hashes")
    result = tuple(sorted((str(name), str(digest)) for name, digest in sources.items()))
    if any(len(digest) != 64 for _, digest in result):
        raise ValueError("Preparation metadata contains a malformed observed-source hash")
    return result


def _as_scalar(arrays: dict[str, np.ndarray], name: str) -> str:
    if name not in arrays or np.asarray(arrays[name]).shape != ():
        raise ValueError(f"Export metadata {name!r} is missing or non-scalar")
    return str(np.asarray(arrays[name]).item())


def _authenticate_one(spec: PrivacyInput, index: dict[str, Any], matched: dict[str, Any]) -> PrivacyInputIdentity:
    if spec.selection_role != SELECTION_ROLE:
        raise ValueError(f"{spec.name} selection_role must be immutable {SELECTION_ROLE!r}")
    if isinstance(spec.parameter_count, bool) or not isinstance(spec.parameter_count, int) or spec.parameter_count < 0:
        raise ValueError(f"{spec.name} parameter_count must be a non-negative integer")
    for path in (spec.export_path, spec.prepared_metadata_path, spec.utility_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _require_indexed_bytes(index, spec.export_path)
    _require_indexed_bytes(index, spec.prepared_metadata_path)
    _require_indexed_bytes(index, spec.utility_report_path)

    metadata = read_json(spec.prepared_metadata_path)
    metadata_hash = sha256_file(spec.prepared_metadata_path)
    definition = {key: metadata.get(key) for key in ("train_end", "validation_end", "categorical_fields", "continuous_fields")}
    definition_hash = _canonical_hash(definition)
    categorical = tuple(metadata.get("categorical_fields", ()))
    continuous = tuple(metadata.get("continuous_fields", ()))
    if not categorical or not continuous or len(set(categorical)) != len(categorical) or len(set(continuous)) != len(continuous):
        raise ValueError(f"{spec.name} preparation field order is missing or ambiguous")
    sources = _source_hashes(metadata)
    if metadata.get("dataset_contract") is None:
        raise ValueError(f"{spec.name} preparation metadata lacks dataset contract")

    export = load_embedding_export(spec.export_path)
    arrays = export.arrays
    if export.schema_version != EXPORT_SCHEMA_VERSION:
        raise ValueError("Privacy audits require the versioned component export schema")
    order = tuple(str(value) for value in arrays["component_names"])
    dimensions = tuple(int(value) for value in arrays["component_dimensions"])
    if order != COMPONENT_NAMES:
        raise ValueError("Export component order does not match the canonical schema")
    if tuple(str(value) for value in arrays["categorical_fields"]) != categorical or tuple(str(value) for value in arrays["continuous_fields"]) != continuous:
        raise ValueError("Export/preparation ordered field mismatch")
    if _as_scalar(arrays, "preparation_hash") != metadata_hash:
        raise ValueError("Export preparation-metadata hash mismatch")
    export_sources = tuple(sorted(zip(arrays["source_file_names"].astype(str), arrays["source_hashes"].astype(str))))
    if export_sources != sources:
        raise ValueError("Export/preparation observed-source hash mismatch")
    if _as_scalar(arrays, "train_end") != str(metadata["train_end"]) or _as_scalar(arrays, "validation_end") != str(metadata["validation_end"]):
        raise ValueError("Export/preparation cutoff-definition mismatch")
    users = arrays["user_id"].astype(str)
    cutoffs = arrays["cutoff"].astype(str)
    keys = sorted(zip(users.tolist(), cutoffs.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("Export contains duplicate user/cutoff keys")
    eligible = tuple(sorted(spec.eligible_users))
    utility_users = tuple(sorted(spec.utility_report_users))
    if len(eligible) != len(set(eligible)) or len(utility_users) != len(set(utility_users)):
        raise ValueError("Population identities contain duplicate users")
    if sorted(set(users)) != list(eligible):
        raise ValueError("Exact export user eligibility identity mismatch")

    model_variant = _as_scalar(arrays, "model_variant")
    if spec.kind == "learned":
        if spec.checkpoint_path is None or not spec.checkpoint_path.is_file():
            raise FileNotFoundError(spec.checkpoint_path or "missing learned checkpoint")
        _require_indexed_bytes(index, spec.checkpoint_path)
        checkpoint_hash = sha256_file(spec.checkpoint_path)
        if spec.checkpoint_identity != checkpoint_hash:
            raise ValueError("Learned checkpoint SHA-256 identity mismatch")
        checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_variant = checkpoint.get("model_variant") or checkpoint.get("component_schema", {}).get("model_variant")
        if checkpoint_variant != model_variant or spec.model_variant != model_variant:
            raise ValueError("Learned model variant mismatch")
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("Learned checkpoint lacks model_state for parameter authentication")
        actual_parameter_count = sum(int(value.numel()) for value in model_state.values())
        if spec.parameter_count != actual_parameter_count:
            raise ValueError("Learned parameter count mismatch")
        checkpoint_identity = checkpoint_hash
    else:
        if spec.checkpoint_path is not None or spec.checkpoint_identity != BASELINE_CHECKPOINT_IDENTITY:
            raise ValueError("Statistical baseline checkpoint identity must be explicit not_applicable")
        if model_variant != "statistical_baseline" or spec.model_variant != model_variant:
            raise ValueError("Statistical baseline model variant mismatch")
        if spec.parameter_count != 0:
            raise ValueError("Statistical baseline parameter count must be zero")
        checkpoint_identity = BASELINE_CHECKPOINT_IDENTITY

    matched_sources = tuple(sorted((str(k), str(v)) for k, v in matched.get("source_files", {}).items()))
    if matched_sources and matched_sources != sources:
        raise ValueError("Evidence-index observed-source identity mismatch")
    if matched.get("preparation_definition") not in (None, definition):
        raise ValueError("Evidence-index preparation definition mismatch")
    if matched.get("export_keys_sha256") not in (None, _canonical_hash(keys)):
        raise ValueError("Evidence-index export key identity mismatch")
    if matched.get("user_mask_sha256") not in (None, _canonical_hash(list(eligible))):
        raise ValueError("Evidence-index eligible-user identity mismatch")
    if matched.get("cutoffs") not in (None, sorted(set(cutoffs))):
        raise ValueError("Evidence-index cutoff identity mismatch")

    utility = read_json(spec.utility_report_path)
    declared_utility_users = utility.get("population_identity", {}).get("users")
    declared_utility_hash = utility.get("population_identity", {}).get("user_set_sha256")
    expected_utility_hash = _canonical_hash(list(utility_users))
    if declared_utility_users is not None and tuple(sorted(map(str, declared_utility_users))) != utility_users:
        raise ValueError("Utility-report population identity mismatch")
    if declared_utility_hash != expected_utility_hash:
        raise ValueError("Utility-report population hash mismatch")

    return PrivacyInputIdentity(
        PRIVACY_INPUT_SCHEMA_VERSION, spec.name, spec.kind, spec.selection_role,
        str(spec.export_path), spec.export_path.stat().st_size, sha256_file(spec.export_path),
        export.schema_version, model_variant, order, dimensions, _canonical_hash(keys),
        tuple(sorted(set(cutoffs))), checkpoint_identity, metadata_hash, definition_hash,
        str(metadata["dataset_contract"]), categorical, continuous, sources,
        spec.parameter_count, _canonical_hash(list(eligible)), expected_utility_hash,
    )


def authenticate_privacy_inputs(
    evidence_index_path: str | Path,
    inputs: tuple[PrivacyInput, ...] | list[PrivacyInput],
    *,
    runtime_metadata: RuntimeMetadata | None = None,
) -> AuthenticatedPrivacyInputs:
    """Authenticate every public input before a caller opens protected labels.

    No output path or protected-label path is accepted by this function.  Thus a
    failure cannot partially create either privacy output or accidentally open
    truth merely while attempting authentication.
    """
    path = Path(evidence_index_path)
    index = read_json(path)
    schema = index.get("schema_version")
    if schema not in {FACTORIZATION_INDEX_SCHEMA_VERSION, EVIDENCE_INDEX_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported evidence-index schema: {schema!r}")
    if index.get("task_id") not in {"T2.7", "T2.4-T2.7"}:
        raise ValueError("Privacy inputs require the T2.7 evidence identity")
    if index.get("decision") != "do not advance":
        raise ValueError("T2.7 decision mismatch; expected 'do not advance'")
    matched = index.get("matched_identity")
    if not isinstance(matched, dict):
        raise ValueError("Evidence index lacks the matched T2.7 identity")
    if not inputs:
        raise ValueError("At least one privacy input is required")
    names = [item.name for item in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Privacy input names must be unique")
    identities = tuple(_authenticate_one(item, index, matched) for item in inputs)
    first = identities[0]
    for identity in identities[1:]:
        for field in ("preparation_metadata_sha256", "preparation_definition_sha256", "dataset_contract", "categorical_fields", "continuous_fields", "observed_source_hashes", "export_keys_sha256", "cutoffs", "eligible_users_sha256", "utility_report_users_sha256"):
            if getattr(identity, field) != getattr(first, field):
                raise ValueError(f"Privacy controls have mismatched {field}")
    return AuthenticatedPrivacyInputs(
        sha256_file(path), str(schema), str(index["task_id"]), str(index["decision"]),
        identities, runtime_metadata,
    )


def _excluded_summary(
    declaration: ProtectedAttributeDeclaration,
    reason: str,
    *,
    eligible_count: int = 0,
    missing_count: int = 0,
    unsupported_count: int = 0,
) -> ProtectedAttributeSummary:
    return ProtectedAttributeSummary(
        declaration.name, "excluded", reason, declaration.privacy_rationale,
        declaration.visibility, declaration.derivation_version, declaration.derivation_method,
        None, declaration.labels, (), (), eligible_count, missing_count, unsupported_count,
    )


def load_protected_labels(
    authenticated: AuthenticatedPrivacyInputs,
    user_latents_truth_path: str | Path,
    *,
    split_by_user: Mapping[str, Literal["train", "validation", "test"]],
    attributes: Sequence[str] = ("price_sensitivity_group", "family_orientation_group"),
    minimum_total: int = 100,
    minimum_per_class: int = 20,
    minimum_cell_support: int = 20,
) -> ProtectedLabelBundle:
    """Load and derive a bounded set of protected simulator labels.

    Authentication and all request validation happen before the truth file is
    opened. Continuous bin edges are fitted on probe-train values only. Missing
    values are excluded rather than imputed, and unsupported/public attributes
    receive aggregate machine-readable exclusions. ``minimum_cell_support`` is
    enforced for every class-by-probe-split cell, preventing reconstructable
    small cells from reaching an attack or report.
    """
    if not isinstance(authenticated, AuthenticatedPrivacyInputs) or not authenticated.inputs:
        raise TypeError("representation authentication must succeed before protected labels are loaded")
    if authenticated.evidence_index_schema not in {
        FACTORIZATION_INDEX_SCHEMA_VERSION, EVIDENCE_INDEX_SCHEMA_VERSION
    } or authenticated.t2_7_decision != "do not advance":
        raise ValueError("authenticated representation evidence is not eligible for privacy evaluation")
    if not attributes or len(attributes) != len(set(attributes)):
        raise ValueError("protected attributes must be a non-empty, unique predeclared list")
    for value in (minimum_total, minimum_per_class, minimum_cell_support):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("privacy label support thresholds must be positive integers")
    invalid_splits = {str(value) for value in split_by_user.values()} - {"train", "validation", "test"}
    if invalid_splits:
        raise ValueError(f"unsupported probe splits: {sorted(invalid_splits)}")

    declarations: list[ProtectedAttributeDeclaration] = []
    for name in attributes:
        lower = str(name).lower()
        if any(token in lower for token in PROHIBITED_PROTECTED_ATTRIBUTE_TOKENS):
            raise ValueError(f"prohibited protected attribute request: {name!r}")
        declaration = SUPPORTED_PROTECTED_ATTRIBUTES.get(str(name))
        if declaration is None:
            raise ValueError(f"undeclared protected attribute request: {name!r}")
        declarations.append(declaration)

    path = Path(user_latents_truth_path).expanduser().resolve()
    if path.name != TRUTH_FILES["user_latents"] or path.parent.name != "truth":
        raise ValueError("protected labels require the evaluator-resolved canonical truth/user_latents path")
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical protected-label source: {path}")

    # Only after every public/authentication/request check has passed may truth open.
    frame = pd.read_csv(path, dtype={"user_id": str})
    if "user_id" not in frame or frame["user_id"].isna().any() or frame["user_id"].duplicated().any():
        raise ValueError("canonical user_latents truth must contain unique, nonmissing user_id values")
    frame["user_id"] = frame["user_id"].astype(str)
    summaries: list[ProtectedAttributeSummary] = []
    protected: dict[str, dict[str, str]] = {}
    total_rows = int(len(frame))

    for declaration in declarations:
        if declaration.visibility == "public_or_observed":
            summaries.append(_excluded_summary(
                declaration, "public_or_observed_not_a_protected_leakage_target",
                unsupported_count=total_rows,
            ))
            continue
        if declaration.source_field not in frame:
            summaries.append(_excluded_summary(
                declaration, "canonical_truth_field_missing", unsupported_count=total_rows,
            ))
            continue
        numeric = pd.to_numeric(frame[declaration.source_field], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        split_known = frame["user_id"].isin(split_by_user)
        eligible_mask = finite & split_known.to_numpy()
        missing_count = int((~finite).sum())
        unsupported_count = int((finite & ~split_known.to_numpy()).sum())
        train_mask = eligible_mask & frame["user_id"].map(split_by_user).eq("train").to_numpy()
        train_values = numeric.to_numpy(dtype=float)[train_mask]
        if len(train_values) < len(declaration.labels):
            summaries.append(_excluded_summary(
                declaration, "probe_train_support_inadequate_for_bin_fit",
                eligible_count=int(eligible_mask.sum()), missing_count=missing_count,
                unsupported_count=unsupported_count,
            ))
            continue
        quantiles = np.linspace(0.0, 1.0, len(declaration.labels) + 1)[1:-1]
        interior = np.quantile(train_values, quantiles)
        if not np.all(np.isfinite(interior)) or len(np.unique(interior)) != len(interior):
            summaries.append(_excluded_summary(
                declaration, "probe_train_bin_boundaries_not_distinct",
                eligible_count=int(eligible_mask.sum()), missing_count=missing_count,
                unsupported_count=unsupported_count,
            ))
            continue
        values = numeric.to_numpy(dtype=float)[eligible_mask]
        indices = np.searchsorted(interior, values, side="right")
        labels = np.asarray(declaration.labels, dtype=object)[indices]
        users = frame.loc[eligible_mask, "user_id"].to_numpy(dtype=str)
        derived = dict(zip(users.tolist(), labels.tolist(), strict=True))
        counts = {label: int(np.sum(labels == label)) for label in declaration.labels}
        cell_counts = {
            (split, label): sum(1 for user, value in derived.items() if split_by_user[user] == split and value == label)
            for split in ("train", "validation", "test") for label in declaration.labels
        }
        reason = None
        if len(derived) < minimum_total:
            reason = "minimum_total_support_not_met"
        elif any(count < minimum_per_class for count in counts.values()):
            reason = "minimum_class_support_not_met"
        elif any(count < minimum_cell_support for count in cell_counts.values()):
            reason = "minimum_class_by_split_cell_support_not_met"
        if reason:
            # Do not expose the small per-cell counts which caused exclusion.
            summaries.append(_excluded_summary(
                declaration, reason, eligible_count=len(derived), missing_count=missing_count,
                unsupported_count=unsupported_count,
            ))
            continue
        summaries.append(ProtectedAttributeSummary(
            declaration.name, "available", None, declaration.privacy_rationale,
            declaration.visibility, declaration.derivation_version, declaration.derivation_method,
            "train", declaration.labels, tuple(float(value) for value in interior),
            tuple((label, counts[label]) for label in declaration.labels), len(derived),
            missing_count, unsupported_count,
        ))
        protected[declaration.name] = derived
    return ProtectedLabelBundle(summaries, protected)


def run_protected_attribute_attacks(
    population: PrivacyPopulation,
    bundle: ProtectedLabelBundle,
    attribute: str,
    config: Any,
) -> dict[str, Any]:
    """Consume opaque labels in-process without returning or logging them."""
    summary = next((item for item in bundle.summaries if item.name == attribute), None)
    if summary is None:
        return {"status": "unavailable", "reason": "attribute_not_requested", "attacks": {}}
    if summary.status != "available":
        return {"status": "unavailable", "reason": summary.reason, "attacks": {}}
    labels = bundle._labels[attribute]
    if any(record.user_id not in labels for record in population.records):
        return {"status": "unavailable", "reason": "privacy_population_has_missing_labels", "attacks": {}}
    return run_privacy_attacks_from_config(population, labels, config, task="sensitive_attribute")
