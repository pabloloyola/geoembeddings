"""Isolated observed-only recoverable two-state benchmark protocol.

This module resolves a benchmark overlay onto the legacy simulator configuration,
generates an immutable matched clean/reference schedule pair, and audits the
evaluator-only coverage evidence.  It deliberately does not alter the legacy
configuration or any model/training path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml
import numpy as np

from . import simulator
from .simulate_pair import simulate_pair


BENCHMARK_SCHEMA = "geoembeddings-recoverable-two-state-benchmark/1.0"
BENCHMARK_V2_SCHEMA = "geoembeddings-recoverable-two-state-benchmark/2.0"
BENCHMARK_V3_SCHEMA = "geoembeddings-recoverable-two-state-benchmark/3.0"
BENCHMARK_V4_SCHEMA = "geoembeddings-recoverable-two-state-benchmark/4.0"
REGISTRY_SCHEMA = "geoembeddings-recoverability-factor-registry/1.0"
REGISTRY_V2_SCHEMA = "geoembeddings-recoverability-factor-registry/2.0"
REGISTRY_V3_SCHEMA = "geoembeddings-recoverability-factor-registry/3.0"
REGISTRY_V4_SCHEMA = "geoembeddings-recoverability-factor-registry/4.0"
_OBSERVED_FORBIDDEN = ("latent", "utility", "true_", "chosen", "episode", "change_")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_spec(spec_path: str | Path) -> dict[str, Any]:
    path = Path(spec_path).resolve()
    spec = _read_yaml(path)
    schema_version = spec.get("schema_version")
    if schema_version not in {BENCHMARK_SCHEMA, BENCHMARK_V2_SCHEMA, BENCHMARK_V3_SCHEMA, BENCHMARK_V4_SCHEMA}:
        raise ValueError("unsupported two-state benchmark schema")
    required = {"benchmark_id", "base_config", "base_config_sha256", "protocol", "generator"}
    if not required <= set(spec):
        raise ValueError(f"benchmark spec is missing {sorted(required - set(spec))}")
    base_path = (path.parent / spec["base_config"]).resolve()
    if not base_path.is_file():
        base_path = (Path.cwd() / spec["base_config"]).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"benchmark base configuration not found: {spec['base_config']}")
    if _sha256(base_path) != str(spec["base_config_sha256"]):
        raise ValueError("benchmark base configuration hash does not match")
    base = simulator.load_config(base_path)
    protocol = spec["protocol"]
    categories = list(protocol["persistent_categories"])
    if schema_version == BENCHMARK_V4_SCHEMA:
        pairs = protocol.get("affinity_pairs", [])
        pair_categories = [category for pair in pairs for category in pair.get("categories", [])]
        if len(pairs) != 2 or len(pair_categories) != 4 or len(set(pair_categories)) != 4 or set(categories) != set(pair_categories):
            raise ValueError("v4 requires two fixed pairs over four distinct categories")
    elif len(categories) != 3 or len(set(categories)) != len(categories):
        raise ValueError("the benchmark requires exactly three declared categories")
    if any(category not in base["world"]["poi_categories"] for category in categories):
        raise ValueError("benchmark category is not supported by the base simulator")
    if protocol["scenario"] != "clean" or protocol["timezone"] != "Asia/Tokyo":
        raise ValueError("the benchmark contract requires clean and Asia/Tokyo")
    if int(protocol["users"]) != 2048 or int(protocol["days"]) != 84:
        raise ValueError("the benchmark contract requires 2048 users and 84 days")
    if not {"development_seeds", "heldout_seed"} <= set(protocol):
        raise ValueError("calibration and held-out seeds must be declared")
    seeds = [int(seed) for seed in protocol["development_seeds"]]
    if not seeds or int(protocol["heldout_seed"]) in seeds:
        raise ValueError("held-out seed must be distinct from all calibration seeds")
    schedule = protocol["schedule_duration_days"]
    if not 7 <= int(schedule) <= 14:
        raise ValueError("temporary schedule duration must be 7-14 days")
    if schema_version in {BENCHMARK_V2_SCHEMA, BENCHMARK_V3_SCHEMA, BENCHMARK_V4_SCHEMA}:
        amendment = (
            "recoverable_two_state_benchmark_v4" if schema_version == BENCHMARK_V4_SCHEMA
            else "recoverable_two_state_benchmark_v3" if schema_version == BENCHMARK_V3_SCHEMA
            else "recoverable_two_state_benchmark_v2"
        )
        if protocol.get("amendment_id") != amendment:
            raise ValueError(f"{amendment} benchmark must declare its amendment id")
        if protocol.get("gates") != {
            "held_out_balanced_accuracy_min": 0.70,
            "held_out_auroc_min": 0.70,
            "bootstrap_lower_ci_above_stratified_permutation_null": True,
            "required_event_opportunity_coverage": True,
            "both_factors_required": True,
        }:
            raise ValueError("v2 benchmark gates must match the amended feasibility protocol")
        expected_mutable = ([] if schema_version == BENCHMARK_V4_SCHEMA else
                            ["generator.preference_discriminating_opportunities.minimum_per_user"]
                            if schema_version == BENCHMARK_V3_SCHEMA else [
                                "generator.category_preference_scale",
                                "protocol.min_matched_opportunities_per_user",
                            ])
        if protocol.get("calibration", {}).get("mutable_parameters") != expected_mutable:
            raise ValueError("calibration must declare only the versioned preference mechanism")
        shift = spec["generator"]["schedule_shift"]
        if (int(protocol["schedule_start_day_offset"]), int(protocol["schedule_duration_days"]),
                float(shift["weekday_hours"]), float(shift["weekend_hours"])) != (35, 14, 5.0, -4.0):
            raise ValueError("the amended benchmark must preserve the passing temporary-schedule parameters")
    resolved = copy.deepcopy(base)
    resolved["run"].update(users=2048, days=84, start_date=str(protocol["start_date"]),
                             scenario="clean", requested_scenario="clean", resolved_scenario="clean")
    resolved["scenario_resolution"] = {
        "declaration_version": "geoembeddings-scenario-resolution/1.0",
        "mode": "explicit", "overrides": {},
    }
    resolved["population"]["preference_categories"] = categories
    resolved["population"]["latent_sd"] = float(spec["generator"]["latent_sd"])
    resolved["population"]["preference_mean"] = float(spec["generator"]["preference_mean"])
    resolved["choice"]["category_preference_scale"] = float(spec["generator"]["category_preference_scale"])
    resolved["choice"]["preference_weight"] = float(spec["generator"]["preference_weight"])
    if schema_version == BENCHMARK_V3_SCHEMA:
        resolved["choice"]["preference_discriminating_opportunities"] = copy.deepcopy(
            spec["generator"]["preference_discriminating_opportunities"]
        )
    if schema_version == BENCHMARK_V4_SCHEMA:
        resolved["choice"]["stable_category_affinity"] = copy.deepcopy(
            spec["generator"]["stable_category_affinity"]
        )
    weights = {
        "routine": {"grocery": 3.0, "restaurant": 2.0, "cafe": 2.0},
        "shopping": {"grocery": 4.0, "restaurant": 2.0, "cafe": 1.0},
        "leisure": {"restaurant": 4.0, "cafe": 3.0, "grocery": 1.0},
        "family_outing": {"restaurant": 4.0, "cafe": 1.0, "grocery": 1.0},
        "travel": {"restaurant": 4.0, "cafe": 1.0, "grocery": 1.0},
    }
    resolved["episodes"]["category_weights"] = weights
    shift = spec["generator"]["schedule_shift"]
    definition = resolved["interventions"]["temporary_schedule_shift_v1"]
    definition["selected_user_fraction"] = float(protocol["selected_user_fraction"])
    definition["eligible_time_flexibility_min"] = float(shift["eligible_time_flexibility_min"])
    definition["schedule_shift"].update(
        start_day_offset=int(protocol["schedule_start_day_offset"]),
        duration_days=int(protocol["schedule_duration_days"]),
        weekday_hours=float(shift["weekday_hours"]), weekend_hours=float(shift["weekend_hours"]),
    )
    resolved["run"]["benchmark"] = {
        "benchmark_id": spec["benchmark_id"], "benchmark_schema": schema_version,
        "factor_registry": spec.get("factor_registry", "recoverable_two_state_benchmark_factor_registry.json"),
        "timezone": "Asia/Tokyo", "training_boundary": "observed_only",
    }
    return {"spec": spec, "base_path": base_path, "config": resolved,
            "categories": categories, "seeds": seeds, "heldout_seed": int(protocol["heldout_seed"])}


def load_factor_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema_version") not in {REGISTRY_SCHEMA, REGISTRY_V2_SCHEMA, REGISTRY_V3_SCHEMA, REGISTRY_V4_SCHEMA}:
        raise ValueError("unsupported factor registry schema")
    factors = registry.get("factors")
    if not isinstance(factors, list) or len(factors) != 2:
        raise ValueError("the two-state registry must contain exactly two factors")
    expected_factors = {"stable_category_affinity", "temporary_schedule_state"} if registry.get("schema_version") == REGISTRY_V4_SCHEMA else {"persistent_category_preference", "temporary_schedule_state"}
    if {factor.get("factor_id") for factor in factors} != expected_factors:
        raise ValueError("the two-state registry must contain only the declared factors")
    expected_amendment = {REGISTRY_V2_SCHEMA: "recoverable_two_state_benchmark_v2",
                          REGISTRY_V3_SCHEMA: "recoverable_two_state_benchmark_v3",
                          REGISTRY_V4_SCHEMA: "recoverable_two_state_benchmark_v4"}.get(registry.get("schema_version"))
    if expected_amendment and registry.get("protocol_amendment") != expected_amendment:
        raise ValueError("factor registry protocol amendment does not match its schema")
    return registry


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _category_coverage(run_dir: Path, categories: list[str], minimum: int) -> dict[str, Any]:
    latents = _rows(run_dir / "truth" / "user_latents.csv.gz")
    choices = _rows(run_dir / "truth" / "choices_truth.csv.gz")
    user_pref = {
        row["user_id"]: max(categories, key=lambda category: float(row[f"pref_{category}"]))
        for row in latents
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in choices:
        if int(row.get("candidate_count", 0)) < 2:
            continue
        user = row["user_id"]
        category = row["chosen_category"]
        counts[user]["preferred" if category == user_pref[user] else "nonpreferred"] += 1
    per_user = {
        user: {"preferred": count["preferred"], "nonpreferred": count["nonpreferred"],
               "matched": min(count["preferred"], count["nonpreferred"]),
               "preferred_category": user_pref[user]}
        for user, count in counts.items()
    }
    for user in user_pref:
        per_user.setdefault(user, {"preferred": 0, "nonpreferred": 0, "matched": 0,
                                   "preferred_category": user_pref[user]})
    eligible = [user for user, count in per_user.items()
                if count["preferred"] >= minimum and count["nonpreferred"] >= minimum]
    category_groups = Counter(count["preferred_category"] for count in per_user.values())
    return {
        "users": len(per_user), "eligible_users": len(eligible), "eligible_user_ids": sorted(eligible),
        "minimum_per_side": minimum, "category_group_counts": dict(sorted(category_groups.items())),
        "per_user": per_user,
        "total_choice_rows": len(choices),
    }


def _schedule_coverage(reference: Path, intervention: Path, minimum_events: int) -> dict[str, Any]:
    records = _rows(intervention / "truth" / "temporary_schedule_shift_truth.csv.gz")
    if not records:
        raise ValueError("temporary schedule intervention records are missing")
    intervals = {(row["change_start_time"], row["change_end_time"]) for row in records}
    if len(intervals) != 1:
        raise ValueError("temporary schedule records have conflicting intervals")
    start_s, end_s = next(iter(intervals))
    start, end = _iso(start_s), _iso(end_s)
    duration = (end - start).total_seconds() / 86400.0
    if not 7 <= duration <= 14:
        raise ValueError("temporary schedule interval is not 7-14 days")
    selected = {row["user_id"] for row in records if row["selected"] == "1" and row["applied"] == "1"}
    affected = _rows(intervention / "truth" / "temporary_schedule_shift_events.csv.gz")
    by_user = Counter(row["user_id"] for row in affected)
    insufficient = sorted(user for user in selected if by_user[user] < minimum_events)
    ref_events = _rows(reference / "observed" / "observed_events.csv.gz")
    int_events = _rows(intervention / "observed" / "observed_events.csv.gz")

    def outside(rows: Iterable[dict[str, str]]) -> Counter[str]:
        return Counter(json.dumps(row, sort_keys=True) for row in rows
                       if not (start_s <= row["timestamp"] < end_s))

    post_equal = outside(ref_events) == outside(int_events)
    return {
        "selected_users": len(selected), "selected_user_ids": sorted(selected),
        "affected_events": len(affected), "affected_events_by_user": dict(sorted(by_user.items())),
        "eligible_users_with_minimum_events": len(selected) - len(insufficient),
        "insufficient_users": insufficient, "start": start_s, "end": end_s,
        "duration_days": duration, "post_interval_observed_events_equal": post_equal,
        "matched_control_window_basis": "same selected users and same local dates in reference run",
    }


def _preference_discriminating_coverage(reference: Path, categories: list[str], minimum: int,
                                        max_distance_gap: float = 2.0, max_price_gap: float = 0.35) -> dict[str, Any]:
    """Audit genuine mixed-category opportunities, not event volume."""
    latents = _rows(reference / "truth" / "user_latents.csv.gz")
    choices = _rows(reference / "truth" / "choices_truth.csv.gz")
    candidates = _rows(reference / "truth" / "candidate_sets.csv.gz")
    preferred = {row["user_id"]: max(categories, key=lambda category: float(row[f"pref_{category}"])) for row in latents}
    by_decision: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_decision[row["decision_id"]].append(row)
    counts: Counter[str] = Counter()
    balanced_decisions: set[str] = set()
    for choice in choices:
        rows = by_decision.get(choice["decision_id"], [])
        if not rows:
            continue
        user_preferred = preferred[choice["user_id"]]
        option_categories = {row["candidate_category"] for row in rows}
        if user_preferred not in option_categories or len(option_categories - {user_preferred}) == 0:
            continue
        alternatives = [row for row in rows if row["candidate_category"] != user_preferred]
        preferred_rows = [row for row in rows if row["candidate_category"] == user_preferred]
        mean_distance_gap = abs(sum(float(row["distance_km"]) for row in preferred_rows) / len(preferred_rows)
                                - sum(float(row["distance_km"]) for row in alternatives) / len(alternatives))
        mean_price_gap = abs(sum(float(row["price"]) for row in preferred_rows) / len(preferred_rows)
                             - sum(float(row["price"]) for row in alternatives) / len(alternatives))
        if mean_distance_gap <= max_distance_gap and mean_price_gap <= max_price_gap:
            balanced_decisions.add(choice["decision_id"])
            counts[choice["user_id"]] += 1
    per_user = {row["user_id"]: int(counts[row["user_id"]]) for row in latents}
    eligible = sorted(user for user, count in per_user.items() if count >= minimum)
    return {
        "minimum_per_user": minimum, "eligible_users": len(eligible),
        "eligible_user_ids": eligible, "per_user_counts": per_user,
        "balanced_discriminating_decisions": len(balanced_decisions),
        "definition": f"candidate set contains latent-preferred and non-preferred categories with mean distance gap <={max_distance_gap}km and mean price gap <={max_price_gap}; same decision supplies time/service matching",
    }


def _stable_affinity_coverage(reference: Path, pairs: list[dict[str, Any]], minimum: int,
                              max_distance_gap: float = 2.0, max_price_gap: float = 0.35) -> dict[str, Any]:
    """Audit v4 opportunities from evaluator-only provenance and candidate truth."""
    latents = {row["user_id"]: row for row in _rows(reference / "truth" / "user_latents.csv.gz")}
    choices = {row["decision_id"]: row for row in _rows(reference / "truth" / "choices_truth.csv.gz")}
    provenance = _rows(reference / "truth" / "stable_category_affinity_opportunities_truth.csv.gz")
    candidates = _rows(reference / "truth" / "candidate_sets.csv.gz")
    by_decision: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_decision[row["decision_id"]].append(row)
    pair_map = {str(pair["pair_id"]): list(pair["categories"]) for pair in pairs}
    counts: Counter[str] = Counter()
    balanced_decisions: set[str] = set()
    pair_counts: Counter[str] = Counter()
    label_counts: Counter[tuple[str, int]] = Counter()
    bad_provenance: list[str] = []
    for item in provenance:
        user = latents.get(item["user_id"])
        choice = choices.get(item["decision_id"])
        rows = by_decision.get(item["decision_id"], [])
        categories = pair_map.get(item["affinity_pair_id"], [])
        if user is None or choice is None or len(rows) == 0 or int(item["affinity_label"]) != int(user["stable_affinity_label"]) or item["affinity_category"] != user["stable_affinity_category"] or set(categories) != {
            item["affinity_category"], item["alternative_category"]
        } or set(row["candidate_category"] for row in rows) != set(categories):
            bad_provenance.append(item["decision_id"])
            continue
        by_category = {category: [row for row in rows if row["candidate_category"] == category] for category in categories}
        distance_gap = abs(np.mean([float(row["distance_km"]) for row in by_category[categories[0]]])
                           - np.mean([float(row["distance_km"]) for row in by_category[categories[1]]]))
        price_gap = abs(np.mean([float(row["price"]) for row in by_category[categories[0]]])
                        - np.mean([float(row["price"]) for row in by_category[categories[1]]]))
        if distance_gap <= max_distance_gap and price_gap <= max_price_gap:
            counts[item["user_id"]] += 1
            balanced_decisions.add(item["decision_id"])
        pair_counts[item["affinity_pair_id"]] += 1
        label_counts[(item["affinity_pair_id"], int(item["affinity_label"]))] += 1
    per_user = {user_id: int(counts[user_id]) for user_id in latents}
    eligible = sorted(user_id for user_id, count in per_user.items() if count >= minimum)
    return {
        "minimum_per_user": minimum,
        "eligible_users": len(eligible),
        "eligible_user_ids": eligible,
        "per_user_counts": per_user,
        "balanced_discriminating_decisions": len(balanced_decisions),
        "pair_counts": dict(sorted(pair_counts.items())),
        "pair_label_counts": {f"{pair}:{label}": count for (pair, label), count in sorted(label_counts.items())},
        "bad_provenance": sorted(bad_provenance),
        "definition": f"fixed declared category pair with mean distance gap <={max_distance_gap}km and mean price gap <={max_price_gap}; same decision supplies timing and local_commerce service matching",
    }


def audit_pair(pair_root: str | Path, resolved: dict[str, Any]) -> dict[str, Any]:
    pair = Path(pair_root).resolve()
    reference = pair.parent / "reference"
    intervention = pair.parent / "intervention"
    protocol = resolved["spec"]["protocol"]
    is_v4 = resolved["spec"].get("schema_version") == BENCHMARK_V4_SCHEMA
    category = (_category_coverage(reference, resolved["categories"], int(protocol.get("min_matched_opportunities_per_user", 0)))
                if not is_v4 else None)
    schedule = _schedule_coverage(reference, intervention, int(protocol["min_schedule_events_per_selected_user"]))
    opportunity_spec = resolved["spec"].get("generator", {}).get("preference_discriminating_opportunities", {})
    matching_spec = opportunity_spec.get("matching", {})
    preference = (_preference_discriminating_coverage(
        reference, resolved["categories"], int(opportunity_spec.get("minimum_per_user", 0)),
        float(matching_spec.get("max_mean_distance_gap_km", 2.0)),
        float(matching_spec.get("max_mean_price_gap", 0.35)),
    ) if resolved["spec"].get("schema_version") == BENCHMARK_V3_SCHEMA else None)
    affinity = (_stable_affinity_coverage(
        reference, resolved["spec"]["protocol"]["affinity_pairs"],
        int(protocol["min_affinity_opportunities_per_user"]),
    ) if is_v4 else None)
    observed_fields = set()
    for path in sorted((reference / "observed").glob("*.csv.gz")):
        observed_fields.update(_rows(path)[0].keys() if _rows(path) else ())
    leaked = sorted(field for field in observed_fields if any(fragment in field.lower() for fragment in _OBSERVED_FORBIDDEN))
    minimum_category = int(protocol.get("min_matched_opportunities_per_user", 0))
    status = "passed" if (
        ((affinity["eligible_users"] >= int(protocol.get("min_affinity_eligible_users", 0)) and not affinity["bad_provenance"])
         if is_v4 else category["eligible_users"] >= int(protocol.get("min_persistent_eligible_users", 0)))
        and schedule["eligible_users_with_minimum_events"] >= int(protocol.get("min_schedule_eligible_users", 0))
        and (preference is None or preference["eligible_users"] >= int(protocol.get("min_preference_discriminating_users", 0)))
        and schedule["post_interval_observed_events_equal"]
        and not leaked
    ) else "failed"
    return {
        "schema_version": (
            "geoembeddings-two-state-coverage/4.0"
            if resolved["spec"].get("schema_version") == BENCHMARK_V4_SCHEMA
            else
            "geoembeddings-two-state-coverage/3.0"
            if resolved["spec"].get("schema_version") == BENCHMARK_V3_SCHEMA
            else "geoembeddings-two-state-coverage/2.0"
            if resolved["spec"].get("schema_version") == BENCHMARK_V2_SCHEMA
            else "geoembeddings-two-state-coverage/1.0"
        ),
        "benchmark_id": resolved["spec"]["benchmark_id"],
        "status": status,
        "source": {"pair_root": str(pair), "reference_run": str(reference), "intervention_run": str(intervention),
                    "seed": resolved["config"]["run"].get("seed"), "scenario": "clean",
                    "requested_scenario": "clean", "resolved_scenario": "clean", "timezone": "Asia/Tokyo"},
        "persistent_category_preference": category,
        "preference_discriminating_opportunities": preference,
        "stable_category_affinity": affinity,
        "temporary_schedule_state": schedule,
        "observed_contract": {"forbidden_fields": _OBSERVED_FORBIDDEN, "leaked_fields": leaked,
                              "truth_not_used_as_training_input": not leaked},
        "minimums": {"matched_opportunities_per_user": minimum_category,
                     "schedule_events_per_selected_user": int(protocol["min_schedule_events_per_selected_user"])},
    }


def simulate_benchmark_pair(spec_path: str | Path, output_root: str | Path, *, seed: int) -> dict[str, Any]:
    resolved = load_benchmark_spec(spec_path)
    registry_path = Path(resolved["spec"].get("factor_registry", "recoverable_two_state_benchmark_factor_registry.json"))
    if not registry_path.is_absolute():
        registry_path = (Path(spec_path).resolve().parents[1] / "recoverability" / registry_path.name).resolve()
    load_factor_registry(registry_path)
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"immutable benchmark output already exists: {root}")
    root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="two-state-config-") as directory:
        config_path = Path(directory) / "resolved.yaml"
        config_path.write_text(yaml.safe_dump(resolved["config"], sort_keys=False), encoding="utf-8")
        result = simulate_pair(config_path, root / "reference", root / "intervention", root / "pair",
                               intervention="temporary_schedule_shift_v1", users=2048, days=84,
                               seed=int(seed), scenario="clean")
    resolved["config"]["run"]["seed"] = int(seed)
    coverage = audit_pair(root / "pair", resolved)
    coverage["source"]["seed"] = int(seed)
    _write_json(root / "two_state_opportunity_coverage.json", coverage)
    if coverage["status"] != "passed":
        raise RuntimeError(f"two-state benchmark coverage failed; inspect {root / 'two_state_opportunity_coverage.json'}")
    return {**result, "coverage_report": str(root / "two_state_opportunity_coverage.json"), "status": "passed"}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/simulation/recoverable_two_state_benchmark_v1.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = simulate_benchmark_pair(args.spec, args.output_root, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
