"""Protected matched-run representation evaluation for R5 and R7."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .comparison import PERSISTENT_TRAITS, PREFERENCE_TRAITS, _load_embeddings
from .contract import COUNTERFACTUAL_COMPARISON_SCHEMA, PairManifest
from .io import read_json, sha256_file, write_json
from .layout import ExperimentLayout, PairLayout
from .pair_integrity import require_passing_pair_integrity
from .runtime_metadata import collect_runtime_metadata


def _stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) / float(16**16)


def _source_contract(metadata: dict[str, Any], side: Any, label: str) -> None:
    expected = {Path(name).name: digest for name, digest in side.source_hashes.items()
                if name.startswith("observed/")}
    actual = metadata.get("source_files")
    if actual != expected:
        raise ValueError(f"{label} preparation source lineage does not match the pair manifest")


def _load_side(side: str, identity: Any, baseline_dir: Path, learned_dir: Path) -> dict[str, Any]:
    layouts = {"baseline": ExperimentLayout.from_path(baseline_dir),
               "learned": ExperimentLayout.from_path(learned_dir)}
    metadata: dict[str, dict[str, Any]] = {}
    exports: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    paths = {"baseline": layouts["baseline"].baseline_embeddings,
             "learned": layouts["learned"].embeddings}
    if paths["baseline"].name != "statistical_baseline.npz" or paths["learned"].name != "embeddings.npz":
        raise ValueError(f"{side} representation kinds must use their canonical export artifacts")
    for kind in ("baseline", "learned"):
        meta_path = layouts[kind].prepared_metadata
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {side} {kind} preparation metadata: {meta_path}")
        metadata[kind] = read_json(meta_path)
        _source_contract(metadata[kind], identity, f"{side} {kind}")
        exports[kind] = _load_embeddings(paths[kind], f"{side} {kind}")
    contract_fields = ("source_files", "train_end", "validation_end", "categorical_fields", "continuous_fields")
    mismatch = [field for field in contract_fields
                if metadata["baseline"].get(field) != metadata["learned"].get(field)]
    if mismatch:
        raise ValueError(f"{side} baseline/learned preparation contracts mismatch: {mismatch}")
    if set(exports["baseline"]) != set(exports["learned"]):
        raise ValueError(f"{side} baseline and learned exports have mismatching keys")
    return {"metadata": metadata["learned"], "exports": exports, "paths": paths,
            "metadata_paths": {k: layouts[k].prepared_metadata for k in layouts}}


def match_pair_keys(reference: dict[tuple[str, str], np.ndarray],
                    intervention: dict[tuple[str, str], np.ndarray]) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Return deterministic common keys and explicit symmetric coverage."""
    left, right = set(reference), set(intervention)
    matched = sorted(left & right)
    return matched, {"reference_rows": len(left), "intervention_rows": len(right),
        "matched_rows": len(matched), "reference_only_rows": len(left-right),
        "intervention_only_rows": len(right-left),
        "match_fraction_reference": len(matched)/len(left) if left else 0.0,
        "match_fraction_intervention": len(matched)/len(right) if right else 0.0,
        "reference_only_samples": [list(x) for x in sorted(left-right)[:10]],
        "intervention_only_samples": [list(x) for x in sorted(right-left)[:10]]}


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12)


def _effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    values = np.linalg.svd(centered, compute_uv=False) ** 2
    if not values.sum(): return 0.0
    p = values / values.sum()
    return float(np.exp(-np.sum(p*np.log(np.maximum(p, 1e-15)))))


def _probe_change(ref: np.ndarray, intervention: np.ndarray, labels: pd.DataFrame,
                  users: list[str], targets: list[str], train_fraction: float, alpha: float) -> dict[str, Any]:
    available = [x for x in targets if x in labels.columns]
    mask = np.asarray([_stable_fraction(user) < train_fraction for user in users])
    if mask.sum() < 2 or (~mask).sum() < 2 or not available:
        return {"status": "insufficient_held_out_users", "targets": available,
                "train_users": int(mask.sum()), "test_users": int((~mask).sum())}
    y = labels.loc[users, available].to_numpy(float)
    model = Ridge(alpha=alpha).fit(ref[mask], y[mask])
    ref_r2 = float(r2_score(y[~mask], model.predict(ref[~mask]), multioutput="variance_weighted"))
    int_r2 = float(r2_score(y[~mask], model.predict(intervention[~mask]), multioutput="variance_weighted"))
    return {"status": "measured", "targets": available, "train_users": int(mask.sum()),
            "test_users": int((~mask).sum()), "reference_r2": ref_r2,
            "intervention_r2": int_r2, "intervention_minus_reference_r2": int_r2-ref_r2,
            "frozen_probe": True}


def representation_metrics(reference: dict[tuple[str, str], np.ndarray],
                           intervention: dict[tuple[str, str], np.ndarray], labels: pd.DataFrame,
                           *, train_fraction: float, alpha: float) -> dict[str, Any]:
    keys, coverage = match_pair_keys(reference, intervention)
    test_keys = [key for key in keys if key[1] == "test" and key[0] in labels.index]
    users = [key[0] for key in test_keys]
    if len(test_keys) < 2:
        raise ValueError("Paired evaluation requires at least two matched test-cutoff users")
    a, b = (np.stack([source[key] for key in test_keys]) for source in (reference, intervention))
    if a.shape[1] != b.shape[1]:
        raise ValueError("Reference and intervention representation dimensions mismatch")
    cosine = _cosine_rows(a, b)
    normalized_a = a/np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    normalized_b = b/np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    similarities = normalized_a @ normalized_b.T
    retrieval = float(np.mean(np.argmax(similarities, axis=1) == np.arange(len(users))))
    different = similarities[~np.eye(len(users), dtype=bool)]
    return {"coverage": {**coverage, "matched_test_users": len(users),
            "excluded_matched_non_test_rows": len(keys)-len(test_keys)},
        "embedding_drift": {"mean_cosine_distance": float(np.mean(1-cosine)),
            "median_cosine_distance": float(np.median(1-cosine))},
        "persistent_trait_probe_change": _probe_change(a,b,labels,users,PERSISTENT_TRAITS,train_fraction,alpha),
        "frozen_downstream_degradation": _probe_change(a,b,labels,users,PREFERENCE_TRAITS,train_fraction,alpha),
        "separation": {"matched_same_user_cosine": float(np.mean(cosine)),
            "different_user_cross_run_cosine": float(np.mean(different))},
        "retrieval": {"cross_run_user_top1": retrieval, "users": len(users)},
        "effective_rank": {"reference": _effective_rank(a), "intervention": _effective_rank(b)},
        "task_information": {"persistent_traits": "frozen reference-fit ridge probe",
            "category_preferences": "frozen reference-fit ridge probe"}}


def evaluate_pair(pair_manifest_path: str | Path, baseline_experiment_dirs: list[Path],
                  learned_experiment_dirs: list[Path], config: dict[str, Any], *, overwrite: bool=False) -> dict[str, Any]:
    """Evaluate baseline and learned exports after authenticating the protected pair."""
    started = time.perf_counter()
    if len(baseline_experiment_dirs) != 2 or len(learned_experiment_dirs) != 2:
        raise ValueError("Exactly reference and intervention experiment directories are required for each kind")
    integrity = require_passing_pair_integrity(pair_manifest_path)
    layout = PairLayout.from_manifest_path(pair_manifest_path)
    outputs = (layout.counterfactual_comparison_json, layout.counterfactual_comparison_markdown)
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError("Counterfactual comparison already exists; use --overwrite to replace it")
    pair = PairManifest.from_dict(read_json(layout.manifest))
    ref = _load_side("reference", pair.reference, baseline_experiment_dirs[0], learned_experiment_dirs[0])
    inter = _load_side("intervention", pair.intervention, baseline_experiment_dirs[1], learned_experiment_dirs[1])
    compatible = ("train_end", "validation_end", "categorical_fields", "continuous_fields")
    mismatch = [x for x in compatible if ref["metadata"].get(x) != inter["metadata"].get(x)]
    if mismatch: raise ValueError(f"Paired-run preparation contracts mismatch: {mismatch}")
    truth_paths = [Path(pair.reference.run_dir)/"truth"/"user_latents.csv.gz",
                   Path(pair.intervention.run_dir)/"truth"/"user_latents.csv.gz"]
    left, right = (pd.read_csv(path).set_index("user_id") for path in truth_paths)
    common_truth = sorted(set(left.index) & set(right.index))
    trait_fields = [x for x in PERSISTENT_TRAITS+PREFERENCE_TRAITS if x in left and x in right]
    if not left.loc[common_truth, trait_fields].equals(right.loc[common_truth, trait_fields]):
        raise ValueError("Protected invariant trait labels differ across paired runs")
    settings=config["evaluation"]; metrics={}
    for kind in ("baseline","learned"):
        metrics[kind]=representation_metrics(ref["exports"][kind],inter["exports"][kind],left,
            train_fraction=float(settings["probe_train_fraction"]),alpha=float(settings["ridge_alpha"]))
        if pair.intervention_type == "schedule-shift":
            metrics[kind]["schedule_shift_response"] = {
                "mean_matched_cosine_distance": metrics[kind]["embedding_drift"]["mean_cosine_distance"],
                "cross_run_periodic_retrieval_top1": metrics[kind]["retrieval"]["cross_run_user_top1"],
                "interpretation": "Response to recurring-clock displacement; persistent probes and geometry are mandatory controls.",
            }
    report={"schema_version":COUNTERFACTUAL_COMPARISON_SCHEMA,
        "runtime_metadata":collect_runtime_metadata(duration_seconds=time.perf_counter()-started,
            seed=int(config.get("seed",0)),device=None).to_dict(),
        "intervention": {"type":pair.intervention_type,"parameters":pair.intervention_parameters,
            "seed_lineage":pair.stream_lineage},
        "integrity": {"status":integrity["status"],"report_sha256":sha256_file(layout.integrity_report),
            "pair_manifest_sha256":sha256_file(layout.manifest)},
        "contracts": {"field_order":{"categorical":ref["metadata"]["categorical_fields"],
            "continuous":ref["metadata"]["continuous_fields"]},"cutoffs":{
            "train_end":ref["metadata"]["train_end"],"validation_end":ref["metadata"]["validation_end"]},
            "prepared_metadata_sha256":{"reference":{k:sha256_file(v) for k,v in ref["metadata_paths"].items()},
            "intervention":{k:sha256_file(v) for k,v in inter["metadata_paths"].items()}},
            "export_sha256":{"reference":{k:sha256_file(v) for k,v in ref["paths"].items()},
            "intervention":{k:sha256_file(v) for k,v in inter["paths"].items()}}},
        "results":metrics,"requirements":{"R3":"executable" if pair.intervention_type == "schedule-shift" else "not_targeted",
            "R4":"executable" if pair.intervention_type == "schedule-shift" else "not_targeted", "R5":"executable","R7":"executable"},
        "information_boundary":"Protected invariant and intervention labels are joined only inside this evaluator.",
        "interpretation":"Interventions and representations are reported independently; no aggregate winner is calculated.",
        "limitations":["Controlled evidence is valid only within the simulator, not externally causal.",
            "GPS/missingness sensitivity is an observation intervention and is not exposure/opportunity invariance."]}
    write_json(report,outputs[0]); outputs[1].write_text(_markdown(report),encoding="utf-8")
    return report


def _markdown(report: dict[str,Any])->str:
    lines=["# Counterfactual representation comparison","",f"Schema: `{report['schema_version']}`",
        f"Intervention: **{report['intervention']['type']}**","",
        "> No aggregate winner is calculated. GPS/missingness sensitivity remains distinct from controlled exposure/opportunity invariance.",""]
    for kind, result in report["results"].items():
        lines += [f"## {kind.title()}","",f"- Matched rows: {result['coverage']['matched_rows']}",
            f"- Matched test users: {result['coverage']['matched_test_users']}",
            f"- Mean cosine drift: {result['embedding_drift']['mean_cosine_distance']:.6f}",
            f"- Cross-run retrieval top-1: {result['retrieval']['cross_run_user_top1']:.6f}",
            f"- Effective rank (reference/intervention): {result['effective_rank']['reference']:.4f} / {result['effective_rank']['intervention']:.4f}",""]
    lines += ["## Provenance and limitations","",f"- Pair manifest SHA-256: `{report['integrity']['pair_manifest_sha256']}`",
        f"- Integrity report SHA-256: `{report['integrity']['report_sha256']}`",f"- Seed: `{report['runtime_metadata']['seed']}`"]
    lines += [f"- {x}" for x in report["limitations"]]
    return "\n".join(lines)+"\n"
