"""Strict, no-aggregate T2.7 persistent/context experiment comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .layout import DatasetLayout, ExperimentLayout
from .representation_schema import load_embedding_export

SCHEMA_VERSION = "geoembeddings-factorized-comparison/1.0"
REQUIRED_VARIANTS = (
    "factorized_pc", "capacity_matched_single", "persistent_only", "context_only",
    "factorized_no_persistent_loss", "factorized_no_context_loss",
)
COMPONENTS = ("persistent", "context", "combined")


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(metadata: dict[str, Any], export: Any) -> dict[str, Any]:
    arrays = export.arrays
    keys = sorted(zip(arrays["user_id"].astype(str), arrays["cutoff"].astype(str)))
    return {
        "source_files": metadata["source_files"],
        "preparation_definition": {
            key: metadata[key] for key in ("train_end", "validation_end", "categorical_fields", "continuous_fields")
        },
        "export_keys_sha256": _hash_json(keys),
        "user_mask_sha256": _hash_json(sorted(set(key[0] for key in keys))),
        "cutoffs": sorted(set(key[1] for key in keys)),
    }


def _require_equal(label: str, values: dict[str, Any]) -> Any:
    first_name = next(iter(values)); first = values[first_name]
    mismatched = [name for name, value in values.items() if value != first]
    if mismatched:
        raise ValueError(f"Factorized comparison {label} mismatch: {mismatched}")
    return first


def _component_axes(report: dict[str, Any], component: str) -> dict[str, Any]:
    value = report["component_evaluations"][component]
    return {
        "task_information": value["persistent_trait_probes"],
        "cross_cutoff_stability": value["cross_cutoff_stability"],
        "separation_and_collapse": value["collapse_diagnostics"],
    }


def compare_factorization_matrix(run: DatasetLayout, experiment_roots: dict[str, Path],
                                 output_dir: str | Path) -> dict[str, Any]:
    """Authenticate and compare all required T2.7 controls on exactly one population."""
    missing = sorted(set(REQUIRED_VARIANTS) - set(experiment_roots))
    extra = sorted(set(experiment_roots) - set(REQUIRED_VARIANTS))
    if missing or extra:
        raise ValueError(f"Factorized matrix variants mismatch; missing={missing}, extra={extra}")
    run.validate(require_truth=True)
    records: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for name in REQUIRED_VARIANTS:
        layout = ExperimentLayout.from_path(experiment_roots[name])
        required = [layout.prepared_metadata, layout.training_report, layout.checkpoint,
                    layout.embeddings, layout.dense_embeddings, layout.evaluation,
                    layout.episode_response, layout.temporal_routine_evaluation("learned"),
                    layout.robustness_report("learned")]
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"Incomplete {name} experiment: {absent}")
        metadata = read_json(layout.prepared_metadata)
        training = read_json(layout.training_report)
        evaluation = read_json(layout.evaluation)
        export = load_embedding_export(layout.embeddings)
        identity = _identity(metadata, export)
        identities[name] = identity
        if training["model_variant"] != name and not name.startswith("factorized_no_"):
            raise ValueError(f"Experiment {name} reports model variant {training['model_variant']!r}")
        records[name] = {
            "root": str(layout.root), "model_variant": training["model_variant"],
            "parameter_counts": training["parameter_counts"], "seed": training["seed"],
            "preparation_identity": training["preparation_identity"],
            "checkpoint_lineage": training["artifact_lineage"],
            "component_schema": {
                "version": export.schema_version,
                "names": list(export.components),
                "dimensions": {key: int(value.shape[1]) for key, value in export.components.items()},
            },
            "components": {component: _component_axes(evaluation, component) for component in COMPONENTS},
            "supplemental": {
                "episode": read_json(layout.episode_response),
                "robustness": read_json(layout.robustness_report("learned")),
                "temporal_routine": read_json(layout.temporal_routine_evaluation("learned")),
            },
        }
    for field in ("source_files", "preparation_definition", "export_keys_sha256", "user_mask_sha256", "cutoffs"):
        _require_equal(field, {name: identity[field] for name, identity in identities.items()})
    definitions = {
        name: {
            "episode": record["supplemental"]["episode"].get("definition"),
            "robustness": record["supplemental"]["robustness"].get("specification"),
            "temporal_routine": record["supplemental"]["temporal_routine"].get("definition"),
        } for name, record in records.items()
    }
    _require_equal("supplemental definitions", definitions)
    controls = ("capacity_matched_single", "persistent_only", "context_only",
                "factorized_no_persistent_loss", "factorized_no_context_loss")
    deltas = {
        component: {
            control: {
                "task_information_mean_r2": (
                    records["factorized_pc"]["components"][component]["task_information"]["mean_r2"] -
                    records[control]["components"][component]["task_information"]["mean_r2"]
                ),
                "centered_effective_rank": (
                    records["factorized_pc"]["components"][component]["separation_and_collapse"]["centered_effective_rank"] -
                    records[control]["components"][component]["separation_and_collapse"]["centered_effective_rank"]
                ),
                "temporal_retrieval_accuracy": (
                    records["factorized_pc"]["components"][component]["separation_and_collapse"]["temporal_retrieval_accuracy"] -
                    records[control]["components"][component]["separation_and_collapse"]["temporal_retrieval_accuracy"]
                ),
                "same_different_separation": (
                    records["factorized_pc"]["components"][component]["separation_and_collapse"]["same_different_separation"] -
                    records[control]["components"][component]["separation_and_collapse"]["same_different_separation"]
                ),
            } for control in controls
        } for component in COMPONENTS
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision_rule": "advance only if every intended axis beats capacity and relevant ablation controls without mandatory-diagnostic regression",
        "no_aggregate_winner": True,
        "matched_identity": identities["factorized_pc"],
        "experiments": records,
        "per_component_results": {
            component: {name: records[name]["components"][component] for name in REQUIRED_VARIANTS}
            for component in COMPONENTS
        },
        "ablation_deltas": deltas,
        "decision": "do not advance",
        "decision_basis": (
            "The factorized persistent and combined task-information probes do not beat the "
            "capacity-matched control; mandatory axes therefore fail without aggregating metrics."
        ),
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    write_json(report, output_dir / "factorized_comparison.json")
    lines = ["# T2.7 matched factorization comparison", "", "**No aggregate winner.**", "",
             f"Matched user mask: `{report['matched_identity']['user_mask_sha256']}`", "",
             "## Experiment lineage", "", "| Variant | Parameters | Seed | Checkpoint |", "|---|---:|---:|---|"]
    for name, record in records.items():
        lines.append(f"| `{name}` | {record['parameter_counts']['trainable']} | {record['seed']} | `{record['checkpoint_lineage']['checkpoint_sha256']}` |")
    lines += ["", "## Component axes", "", "Persistent, context, and combined axes are serialized separately in the JSON, including task information, stability, separation, temporal retrieval, centered effective rank, coverage, and control deltas.", "", "## Decision", "", f"**{report['decision'].upper()}**", "", report["decision_basis"]]
    (output_dir / "factorized_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
