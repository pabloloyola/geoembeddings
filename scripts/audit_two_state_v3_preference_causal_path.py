#!/usr/bin/env python3
"""Audit the v3 persistent-preference causal path from truth to observed data.

This is an evaluator-only diagnostic.  It reads existing development runs,
truth-side candidate/utility tables, and observed events; it never simulates,
trains, changes a configuration, or opens a held-out run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import recoverability_gate as gate


MIN_LOG_ODDS_EFFECT = 0.25
MIN_RISK_DIFFERENCE = 0.10
BOOTSTRAP_REPLICATES = 1000
FACTOR = "pref_cafe"


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_uniform(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -30.0, 30.0))
    return weights / weights.sum()


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
                 / max(1, positives * negatives))


def _bootstrap_difference(high: np.ndarray, low: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    point = float(np.mean(high) - np.mean(low))
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(high), size=len(high))
        samples.append(float(np.mean(high[indices]) - np.mean(low[indices])))
    interval = np.quantile(samples, [0.025, 0.975])
    return {
        "risk_difference": point,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
        "bootstrap_ci_95": [float(interval[0]), float(interval[1])],
    }


def _odds_ratio(high: np.ndarray, low: np.ndarray) -> float:
    epsilon = 0.5 / max(1, len(high))
    high_rate = float(np.mean(high))
    low_rate = float(np.mean(low))
    return float(((high_rate + epsilon) / (1.0 - high_rate + epsilon))
                 / ((low_rate + epsilon) / (1.0 - low_rate + epsilon)))


def _event_key(user_id: str, timestamp: Any, object_id: str) -> tuple[str, int, str]:
    parsed = pd.Timestamp(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return str(user_id), int(parsed.value), str(object_id)


def _candidate_probability_map(rows: list[dict[str, str]]) -> dict[str, float]:
    utilities = np.asarray([float(row["utility_total"]) for row in rows], dtype=float)
    probabilities = _softmax(utilities)
    return {row["candidate_poi_id"]: float(probability)
            for row, probability in zip(rows, probabilities)}


def _counterfactual_pair(
    rows: list[dict[str, str]],
    latent: dict[str, str],
    preferred_category: str,
    group_low: dict[str, float],
    group_high: dict[str, float],
    preference_weight: float,
) -> dict[str, Any]:
    """Use logged non-preference utility and a deterministic common random number."""
    actual_pref = {
        row["candidate_poi_id"]: preference_weight * float(latent[f"pref_{row['candidate_category']}"])
        for row in rows
    }
    base = np.asarray([
        float(row["utility_total"]) - actual_pref[row["candidate_poi_id"]]
        for row in rows
    ], dtype=float)
    high_utilities = base + np.asarray([
        preference_weight * (group_high[row["candidate_category"]]
                             if row["candidate_category"] == preferred_category
                             else group_low[row["candidate_category"]])
        for row in rows
    ])
    low_utilities = base + np.asarray([
        preference_weight * (group_low[row["candidate_category"]]
                             if row["candidate_category"] == preferred_category
                             else group_high[row["candidate_category"]])
        for row in rows
    ])
    high_probabilities = _softmax(high_utilities)
    low_probabilities = _softmax(low_utilities)
    preferred_positions = np.asarray([
        row["candidate_category"] == preferred_category for row in rows
    ])
    high_rate = float(high_probabilities[preferred_positions].sum())
    low_rate = float(low_probabilities[preferred_positions].sum())
    preferred_position = int(np.flatnonzero(preferred_positions)[0])
    alternative_position = int(np.flatnonzero(~preferred_positions)[0])
    high_log_odds = float(np.log(high_probabilities[preferred_position]
                                / high_probabilities[alternative_position]))
    low_log_odds = float(np.log(low_probabilities[preferred_position]
                               / low_probabilities[alternative_position]))
    common_random_number = _stable_uniform(rows[0]["decision_id"] + ":v3-causal-audit")
    high_selected = bool(common_random_number <= np.cumsum(high_probabilities)[-1])
    high_selected_position = int(np.searchsorted(np.cumsum(high_probabilities), common_random_number, side="right"))
    low_selected_position = int(np.searchsorted(np.cumsum(low_probabilities), common_random_number, side="right"))
    return {
        "preferred_choice_probability_high": high_rate,
        "preferred_choice_probability_low": low_rate,
        "preferred_vs_alternative_log_odds_high": high_log_odds,
        "preferred_vs_alternative_log_odds_low": low_log_odds,
        "log_odds_effect": high_log_odds - low_log_odds,
        "common_random_number": common_random_number,
        "high_selected_preferred": bool(preferred_positions[high_selected_position]),
        "low_selected_preferred": bool(preferred_positions[low_selected_position]),
        "high_selected_candidate_id": rows[high_selected_position]["candidate_poi_id"],
        "low_selected_candidate_id": rows[low_selected_position]["candidate_poi_id"],
    }


def audit_run(run_dir: Path, seed: int, trace_path: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    choice_config = config["choice"]
    preference_weight = float(choice_config["preference_weight"])
    cutoff = gate._history_cutoff(run_dir)
    latent_rows = _rows(run_dir / "truth/user_latents.csv.gz")
    latent = {str(row["user_id"]): row for row in latent_rows}
    latent_frame = pd.DataFrame(latent_rows).set_index("user_id")
    labels, label_definition = gate._extreme_binary_labels(latent_frame[FACTOR])
    preferred = {
        user_id: max(("grocery", "restaurant", "cafe"),
                     key=lambda category: float(row[f"pref_{category}"]))
        for user_id, row in latent.items()
    }
    group_low = {category: float(pd.to_numeric(latent_frame[f"pref_{category}"], errors="coerce").quantile(0.20))
                 for category in ("grocery", "restaurant", "cafe")}
    group_high = {category: float(pd.to_numeric(latent_frame[f"pref_{category}"], errors="coerce").quantile(0.80))
                  for category in ("grocery", "restaurant", "cafe")}

    candidates = _rows(run_dir / "truth/candidate_sets.csv.gz")
    choices = _rows(run_dir / "truth/choices_truth.csv.gz")
    candidates_by_decision: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        candidates_by_decision.setdefault(row["decision_id"], []).append(row)
    choices_by_decision = {row["decision_id"]: row for row in choices}

    events = _rows(run_dir / "observed/observed_events.csv.gz")
    event_map: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    for event in events:
        if event["service_id"] != "local_commerce":
            continue
        event_map.setdefault(_event_key(event["user_id"], event["timestamp"], event["object_id"]), []).append(event)
    delay = float(config["events"]["local_commerce"]["event_delay_hours"])

    records: list[dict[str, Any]] = []
    response_high: list[float] = []
    response_low: list[float] = []
    response_high_selected: list[bool] = []
    response_low_selected: list[bool] = []
    effects: list[float] = []
    utility_formula_ok = True
    observation_checks: list[bool] = []
    visible_count = 0
    emitted_count = 0
    balanced_count = 0
    for decision_id, rows in candidates_by_decision.items():
        categories = {row["candidate_category"] for row in rows}
        if len(categories) < 2:
            continue
        choice = choices_by_decision.get(decision_id)
        if choice is None:
            continue
        user_id = str(choice["user_id"])
        user_latent = latent[user_id]
        preferred_category = preferred[user_id]
        preferred_rows = [row for row in rows if row["candidate_category"] == preferred_category]
        alternative_rows = [row for row in rows if row["candidate_category"] != preferred_category]
        if not preferred_rows or not alternative_rows:
            continue
        distance_gap = abs(
            np.mean([float(row["distance_km"]) for row in preferred_rows])
            - np.mean([float(row["distance_km"]) for row in alternative_rows])
        )
        price_gap = abs(
            np.mean([float(row["price"]) for row in preferred_rows])
            - np.mean([float(row["price"]) for row in alternative_rows])
        )
        balanced = bool(distance_gap <= 2.0 and price_gap <= 0.35)
        if not balanced:
            continue
        balanced_count += 1
        choice_time = pd.Timestamp(choice["timestamp"])
        if choice_time.tzinfo is None:
            choice_time = choice_time.tz_localize("Asia/Tokyo")
        event_time = choice_time + pd.to_timedelta(float(delay), unit="h")
        event_rows = event_map.get(_event_key(user_id, event_time, choice["chosen_poi_id"]), [])
        event_row = event_rows[0] if event_rows else None
        event_visible = bool(event_row and pd.Timestamp(event_row["timestamp"]) < cutoff)
        opportunity_visible = bool(choice_time.tz_convert("UTC") < cutoff)
        visible_count += int(opportunity_visible)
        emitted_count += int(bool(event_row))
        if event_row:
            observation_checks.append(
                event_row["object_category"] == choice["chosen_category"]
                and event_row["service_id"] == "local_commerce"
            )
        actual_probabilities = _candidate_probability_map(rows)
        counterfactual = _counterfactual_pair(
            rows, user_latent, preferred_category, group_low, group_high, preference_weight,
        )
        effects.append(counterfactual["log_odds_effect"])
        response_high.append(counterfactual["preferred_choice_probability_high"])
        response_low.append(counterfactual["preferred_choice_probability_low"])
        response_high_selected.append(counterfactual["high_selected_preferred"])
        response_low_selected.append(counterfactual["low_selected_preferred"])
        candidate_records = []
        for row in rows:
            category = row["candidate_category"]
            contribution = preference_weight * float(user_latent[f"pref_{category}"])
            candidate_records.append({
                "candidate_poi_id": row["candidate_poi_id"],
                "candidate_category": category,
                "distance_km": float(row["distance_km"]),
                "price": float(row["price"]),
                "quality": float(row["quality"]),
                "exposed": int(row["exposed"]),
                "utility_decomposition": {
                    "preference_specific": contribution,
                    "utility_preference_logged": float(row["utility_preference"]),
                    "utility_preference_other": float(row["utility_preference"]) - contribution,
                    "price_penalty": float(row["utility_price_penalty"]),
                    "distance_penalty": float(row["utility_distance_penalty"]),
                    "exposure_utility": float(row["utility_exposure"]),
                    "utility_total_logged": float(row["utility_total"]),
                    "noise_residual_from_logged_terms": float(row["utility_total"]) - (
                        float(row["utility_preference"]) - float(row["utility_price_penalty"])
                        - float(row["utility_distance_penalty"]) + float(row["utility_exposure"])
                    ),
                },
                "preference_parameter": f"pref_{category}",
                "preference_parameter_value": float(user_latent[f"pref_{category}"]),
                "preference_specific_utility_contribution": contribution,
                "resulting_choice_probability": actual_probabilities[row["candidate_poi_id"]],
                "selected": bool(int(row["is_chosen"])),
            })
            utility_formula_ok = bool(utility_formula_ok and bool(np.isfinite(contribution)))
        records.append({
            "decision_id": decision_id,
            "user_id": user_id,
            "timestamp": choice["timestamp"],
            "evaluation_cutoff": cutoff.isoformat(),
            "opportunity_inside_model_visible_history": opportunity_visible,
            "event_inside_model_visible_history": event_visible,
            "recoverability_label": int(labels.get(user_id, -1)),
            "recoverability_label_parameter": FACTOR,
            "recoverability_label_parameter_value": float(user_latent[FACTOR]),
            "true_preferred_category": preferred_category,
            "candidate_categories": sorted(categories),
            "distance_gap_km": distance_gap,
            "price_gap": price_gap,
            "balanced_matched_opportunity": balanced,
            "candidates": candidate_records,
            "selected_candidate_id": choice["chosen_poi_id"],
            "selected_candidate_category": choice["chosen_category"],
            "emitted_observed_event": event_row is not None,
            "emitted_observed_event_category": event_row["object_category"] if event_row else None,
            "emitted_observed_service_token": event_row["service_id"] if event_row else None,
            "counterfactual_common_random_number": counterfactual,
        })

    trace_path = Path(trace_path) if trace_path else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    labels = labels.loc[labels.index.intersection(sorted({record["user_id"] for record in records}))]
    per_user = pd.DataFrame([
        {
            "user_id": record["user_id"],
            "label": record["recoverability_label"],
            "preferred_category": record["true_preferred_category"],
            "emitted": int(record["emitted_observed_event"]),
            "preferred_emitted": int(record["emitted_observed_event"] and
                                      record["emitted_observed_event_category"] == record["true_preferred_category"]),
            "truth_preferred": int(record["selected_candidate_category"] == record["true_preferred_category"]),
        }
        for record in records
    ]).groupby("user_id").agg(
        label=("label", "first"), preferred_category=("preferred_category", "first"),
        emitted=("emitted", "sum"), preferred_emitted=("preferred_emitted", "sum"),
        truth_preferred=("truth_preferred", "sum"), opportunities=("emitted", "size"),
    )
    per_user["observed_preferred_choice_rate"] = per_user["preferred_emitted"] / per_user["emitted"].replace(0, np.nan)
    per_user["truth_preferred_choice_rate"] = per_user["truth_preferred"] / per_user["opportunities"]
    per_user = per_user.loc[per_user["label"].isin([0, 1])]
    per_user = per_user.dropna(subset=["observed_preferred_choice_rate"])
    baseline_labels = per_user["label"].to_numpy(dtype=int)
    baseline_scores = per_user["observed_preferred_choice_rate"].to_numpy(dtype=float)
    baseline_prediction = baseline_scores >= 0.5
    baseline_ba = float((baseline_prediction[baseline_labels == 1].mean()
                         + (~baseline_prediction[baseline_labels == 0]).mean()) / 2.0)
    baseline_auroc = _auc(baseline_labels, baseline_scores)
    high_mask = np.asarray(response_high_selected, dtype=bool)
    low_mask = np.asarray(response_low_selected, dtype=bool)
    response = _bootstrap_difference(high_mask.astype(float), low_mask.astype(float), seed + 7001)
    response["odds_ratio"] = _odds_ratio(high_mask, low_mask)
    response["high_preferred_choice_rate"] = float(high_mask.mean())
    response["low_preferred_choice_rate"] = float(low_mask.mean())
    response["risk_difference_pass"] = response["risk_difference"] >= MIN_RISK_DIFFERENCE
    report = {
        "schema_version": "geoembeddings-v3-preference-causal-path-audit/1.0",
        "seed": seed,
        "source_run": str(run_dir),
        "information_boundary": {
            "truth_used_only_for_evaluator_artifacts": True,
            "observed_event_path_checked": True,
            "model_training_performed": False,
            "heldout_seed_opened": False,
        },
        "cutoff": {"timestamp": cutoff.isoformat(), "injected_opportunities": len(records),
                   "visible_before_cutoff": visible_count, "emitted_local_events": emitted_count},
        "label_alignment": {
            "passed": bool(label_definition["label_policy"] == "low <= p20; high >= p80"),
            "label_parameter": FACTOR,
            "label_definition": label_definition,
            "utility_formula": "preference_weight * pref_<candidate_category> + quality/family/episode terms",
            "preference_weight": preference_weight,
            "all_candidate_preference_contributions_finite": utility_formula_ok,
            "interpretation": "The evaluator label is exactly derived from pref_cafe; each candidate utility uses the category-specific latent preference parameter.",
        },
        "utility_effect": {
            "declared_minimum_log_odds_effect": MIN_LOG_ODDS_EFFECT,
            "minimum_observed_log_odds_effect": float(np.min(effects)),
            "median_log_odds_effect": float(np.median(effects)),
            "maximum_observed_log_odds_effect": float(np.max(effects)),
            "passed": bool(np.min(effects) >= MIN_LOG_ODDS_EFFECT),
            "toggle": "only category preference contributions were replaced by p20/p80 category values; logged quality, price, distance, exposure, and noise terms were held fixed",
        },
        "choice_response": response,
        "observation_path": {
            "emitted_event_category_matches_selected_candidate": bool(observation_checks) and all(observation_checks),
            "checked_emitted_events": len(observation_checks),
            "service_token": "local_commerce",
        },
        "cutoff_path": {
            "all_injected_opportunities_before_cutoff": visible_count == len(records),
            "visible_opportunities": visible_count,
            "total_injected_opportunities": len(records),
            "visible_emitted_events": sum(int(record["event_inside_model_visible_history"]) for record in records),
        },
        "trivial_observable_baseline": {
            "feature": "per-user observed preferred-category event rate over injected opportunities",
            "threshold": 0.5,
            "evaluated_users": int(len(per_user)),
            "balanced_accuracy": baseline_ba,
            "auroc": baseline_auroc,
            "ba_gate_pass": baseline_ba >= 0.70,
            "auroc_gate_pass": baseline_auroc >= 0.70,
            "passed": baseline_ba >= 0.70 and baseline_auroc >= 0.70,
            "preference_category_rate_feature": per_user.reset_index().to_dict(orient="records"),
        },
        "assertions": {
            "label_alignment": bool(label_definition["label_policy"] == "low <= p20; high >= p80" and utility_formula_ok),
            "utility_effect": bool(np.min(effects) >= MIN_LOG_ODDS_EFFECT),
            "choice_response": bool(response["risk_difference_pass"]),
            "observation_path": bool(observation_checks) and all(observation_checks),
            "cutoff_path": visible_count == len(records),
            "baseline_sanity": bool(baseline_ba >= 0.70 and baseline_auroc >= 0.70),
        },
        "trace_artifact": str(trace_path.resolve()) if trace_path else None,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("experiments/multihorizon-profile-s20260817/recoverable_two_state_benchmark_v3/calibration"),
    )
    args = parser.parse_args()
    roots = {
        20260821: args.input_root / "seed20260821/reference",
        20260822: args.input_root / "seed20260822/reference",
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed, root in roots.items():
        report = audit_run(root, seed, args.output_root / f"seed{seed}_preference_causal_trace.jsonl.gz")
        (args.output_root / f"seed{seed}_preference_causal_audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(report)
    result = {
        "schema_version": "geoembeddings-v3-preference-causal-path-audit/1.0",
        "status": "passed" if all(report["assertions"]["baseline_sanity"] for report in reports) else "generator_label_design_contradiction",
        "seeds": reports,
        "heldout_seed_opened": False,
        "model_training_performed": False,
    }
    (args.output_root / "two_state_v3_preference_causal_path_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "seeds": [
        {"seed": report["seed"], "assertions": report["assertions"],
         "baseline": {key: report["trivial_observable_baseline"][key]
                      for key in ("evaluated_users", "balanced_accuracy", "auroc", "passed")}}
        for report in reports
    ]}, indent=2))


if __name__ == "__main__":
    main()
