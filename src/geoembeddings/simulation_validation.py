#!/usr/bin/env python3
"""Deep validation for GeoEmbeddings Kanto simulator datasets.

The simulator's built-in validator protects the table contract while this
script checks three additional layers:

1. Structural integrity and the observed/truth information boundary.
2. Whether observed events remain coherent with the latent world.
3. Whether the intended behavioral mechanisms are visible in aggregate.

It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .contract import OBSERVED_FILES, TRUTH_FILES, validate_identity_manifest
from .simulator import RANDOM_STREAM_NAMES, identity_set_hash


TABLES = {
    "events": f"observed/{OBSERVED_FILES['events']}",
    "users": f"observed/{OBSERVED_FILES['users']}",
    "poi_catalog": f"observed/{OBSERVED_FILES['poi_catalog']}",
    "recommendation_requests": f"observed/{OBSERVED_FILES['recommendation_requests']}",
    "impressions": f"observed/{OBSERVED_FILES['impressions']}",
    "interactions": f"observed/{OBSERVED_FILES['interactions']}",
    "latents": f"truth/{TRUTH_FILES['user_latents']}",
    "episodes": f"truth/{TRUTH_FILES['episodes']}",
    "candidates": f"truth/{TRUTH_FILES['candidate_sets']}",
    "choices": f"truth/{TRUTH_FILES['choices']}",
    "trajectories": f"truth/{TRUTH_FILES['trajectories']}",
    "observation": f"truth/{TRUTH_FILES['observation_process']}",
}

OBSERVED_FORBIDDEN = (
    "latent",
    "utility",
    "episode_id",
    "true_",
    "price_sensitivity",
    "distance_sensitivity",
    "digital_engagement",
    "travel_propensity",
)


def read_rows(root: Path, relative: str) -> list[dict[str, str]]:
    path = root / relative
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dataset(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return manifest, {name: read_rows(root, relative) for name, relative in TABLES.items()}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def check(
    name: str,
    passed: bool,
    value: Any,
    expectation: str,
    layer: str,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "expectation": expectation,
        "layer": layer,
        "severity": severity,
    }


def duplicate_count(rows: list[dict[str, str]], key: str) -> int:
    counts = Counter(row[key] for row in rows)
    return sum(value - 1 for value in counts.values() if value > 1)


def validate(root: Path) -> dict[str, Any]:
    manifest, data = load_dataset(root)
    events = data["events"]
    users = data["users"]
    latents = data["latents"]
    episodes = data["episodes"]
    candidates = data["candidates"]
    choices = data["choices"]
    trajectories = data["trajectories"]
    observation = data["observation"]
    checks: list[dict[str, Any]] = []

    identity_error = None
    try:
        validate_identity_manifest(manifest.get("identity"), stream_names=RANDOM_STREAM_NAMES)
    except ValueError as exc:
        identity_error = str(exc)
    checks.append(check("Identity manifest schema", identity_error is None, identity_error or "complete", "Versioned identity and stream provenance is complete", "integrity"))

    # Contract and key integrity.
    row_counts = {relative: len(data[name]) for name, relative in TABLES.items()}
    expected_counts = manifest.get("table_rows", {})
    mismatched_counts = {
        path: {"manifest": expected_counts.get(path), "actual": actual}
        for path, actual in row_counts.items()
        if expected_counts.get(path) != actual
    }
    checks.append(check("Manifest row counts", not mismatched_counts, mismatched_counts or "all match", "Every table count matches manifest.json", "integrity"))

    unique_specs = [
        ("Observed users", users, "user_id"),
        ("Latent users", latents, "user_id"),
        ("Episodes", episodes, "episode_id"),
        ("Choices", choices, "decision_id"),
        ("Trajectories", trajectories, "trajectory_id"),
    ]
    duplicate_summary = {label: duplicate_count(rows, key) for label, rows, key in unique_specs}
    checks.append(check("Primary-key uniqueness", not any(duplicate_summary.values()), duplicate_summary, "Zero duplicate primary keys", "integrity"))

    durable_pattern = re.compile(r"^(user|episode|decision|trajectory)_[0-9a-f]{24}$")
    durable_values = [
        *(row["user_id"] for row in users), *(row["episode_id"] for row in episodes),
        *(row["decision_id"] for row in choices), *(row["trajectory_id"] for row in trajectories),
    ]
    malformed = sum(durable_pattern.fullmatch(value) is None for value in durable_values)
    checks.append(check("Durable identity format", malformed == 0, malformed, "Generated identities use the versioned semantic-key format", "integrity"))

    if identity_error is None:
        declarations = manifest["identity"]["entities"]
        actual_identity_sets = {
            "users": [row["user_id"] for row in users],
            "episodes": [row["episode_id"] for row in episodes],
            "choices": [row["decision_id"] for row in choices],
            "trajectories": [row["trajectory_id"] for row in trajectories],
        }
        inconsistent = {
            name: {"declared": declaration, "actual_count": len(values), "actual_sha256": identity_set_hash(values)}
            for name, values in actual_identity_sets.items()
            if (declaration := declarations[name])["count"] != len(values)
            or declaration["identity_sha256"] != identity_set_hash(values)
        }
        top_streams = manifest.get("random_streams", {})
        nested_streams = manifest["identity"]["random_streams"]
        if top_streams != nested_streams:
            inconsistent["random_streams"] = {"top_level": top_streams, "identity": nested_streams}
        referenced_pois = {row["candidate_poi_id"] for row in candidates}
        if declarations["pois"]["count"] < len(referenced_pois):
            inconsistent["pois"] = {"declared_count": declarations["pois"]["count"], "referenced_count": len(referenced_pois)}
        referenced_regions = {row["region_id"] for row in events} | {row["home_region_id"] for row in users}
        if declarations["regions"]["count"] < len(referenced_regions):
            inconsistent["regions"] = {"declared_count": declarations["regions"]["count"], "referenced_count": len(referenced_regions)}
        checks.append(check("Identity manifest consistency", not inconsistent, inconsistent or "all match", "Entity hashes/counts and stream declarations agree with tables", "integrity"))

    user_ids = {row["user_id"] for row in users}
    foreign_failures = {
        name: sum(row["user_id"] not in user_ids for row in rows)
        for name, rows in (("events", events), ("latents", latents), ("episodes", episodes), ("trajectories", trajectories), ("observation", observation))
    }
    checks.append(check("User foreign keys", not any(foreign_failures.values()), foreign_failures, "Every user_id resolves to observed/users", "integrity"))

    decision_candidates: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        decision_candidates[row["decision_id"]].append(row)
    choice_by_decision = {row["decision_id"]: row for row in choices}
    candidate_problems = 0
    for decision_id, rows in decision_candidates.items():
        selected = [row for row in rows if row["is_chosen"] == "1"]
        if len(selected) != 1 or decision_id not in choice_by_decision:
            candidate_problems += 1
            continue
        choice = choice_by_decision[decision_id]
        if selected[0]["candidate_poi_id"] != choice["chosen_poi_id"] or len(rows) != int(choice["candidate_count"]):
            candidate_problems += 1
    candidate_problems += len(set(choice_by_decision) - set(decision_candidates))
    checks.append(check("Choice/candidate consistency", candidate_problems == 0, candidate_problems, "One matching chosen candidate and declared candidate count per decision", "integrity"))

    observed_columns = set(events[0]) | set(users[0]) if events and users else set()
    leaked = sorted(column for column in observed_columns if any(fragment in column for fragment in OBSERVED_FORBIDDEN))
    checks.append(check("Observed/truth boundary", not leaked, leaked or "no forbidden columns", "No latent, utility, episode, or true-coordinate columns in observed tables", "integrity"))

    event_times = [parse_time(row["timestamp"]) for row in events]
    start = datetime.fromisoformat(manifest["start_date"] + "T00:00:00+09:00")
    end = start + timedelta(days=int(manifest["days"]))
    outside_window = sum(not (start <= timestamp < end + timedelta(hours=6)) for timestamp in event_times)
    checks.append(check("Event time window", outside_window == 0, outside_window, "All observed events fall inside the simulated interval", "integrity"))

    invalid_coordinates = sum(
        not (34.0 <= float(row["latitude"]) <= 38.0 and 138.0 <= float(row["longitude"]) <= 142.0 and float(row["location_accuracy_m"]) > 0)
        for row in events
    )
    checks.append(check("Coordinate plausibility", invalid_coordinates == 0, invalid_coordinates, "All observed points fall in a broad Kanto bounding box with positive accuracy", "integrity"))

    # Continuous user geography should not collapse onto a handful of hub
    # centroids. Neighboring hub catchments are intentionally allowed to
    # overlap, so some users assigned different home regions should be nearby.
    home_coordinate_columns = {"home_latitude", "home_longitude", "home_region_id"}
    if latents and home_coordinate_columns <= set(latents[0]):
        unique_home_points = {
            (round(float(row["home_latitude"]), 5), round(float(row["home_longitude"]), 5))
            for row in latents
        }
        unique_share = len(unique_home_points) / len(latents)
        checks.append(
            check(
                "Continuous home geography",
                unique_share >= 0.99,
                round(unique_share, 4),
                "At least 99% of users have distinct home coordinates",
                "coherence",
            )
        )
        overlap_sample = sorted(latents, key=lambda row: row["user_id"])
        if len(overlap_sample) > 2000:
            step = len(overlap_sample) / 2000
            overlap_sample = [overlap_sample[int(index * step)] for index in range(2000)]
        nearest_cross_region: list[float] = []
        for row in overlap_sample:
            distances = [
                haversine_km(
                    float(row["home_latitude"]),
                    float(row["home_longitude"]),
                    float(other["home_latitude"]),
                    float(other["home_longitude"]),
                )
                for other in overlap_sample
                if other["home_region_id"] != row["home_region_id"]
            ]
            if distances:
                nearest_cross_region.append(min(distances))
        overlap_share = mean(distance <= 5.0 for distance in nearest_cross_region)
        checks.append(
            check(
                "Cross-region spatial overlap",
                overlap_share >= 0.10,
                {
                    "share_with_cross_region_neighbor_within_5km": round(overlap_share, 3),
                    "median_nearest_cross_region_km": round(percentile(nearest_cross_region, 0.50), 3),
                },
                "At least 10% of sampled users are within 5 km of a user assigned to another hub catchment",
                "coherence",
            )
        )
    else:
        checks.append(
            check(
                "Continuous home geography",
                False,
                "home coordinate columns absent",
                "New v0.2 datasets include continuous home/work coordinates in evaluator truth",
                "coherence",
                severity="warning",
            )
        )

    # Latent-to-observed coherence. Passive points should match a true stop for
    # the same user and approximately the same time, despite timestamp jitter.
    trajectory_index: defaultdict[str, list[tuple[datetime, dict[str, str]]]] = defaultdict(list)
    for row in trajectories:
        trajectory_index[row["user_id"]].append((parse_time(row["timestamp"]), row))
    passive = [row for row in events if row["observation_mode"] == "passive"]
    passive_errors: list[float] = []
    unmatched_passive = 0
    region_mismatches = 0
    for row in passive:
        timestamp = parse_time(row["timestamp"])
        temporal_candidates = [
            item
            for item in trajectory_index[row["user_id"]]
            if abs((item[0] - timestamp).total_seconds()) <= 15 * 60
        ]
        if not temporal_candidates:
            unmatched_passive += 1
            continue
        # Two truth stops can legitimately be close in time (for example a
        # midday work stop and a nearby POI visit). Region is an observed
        # coarse label, so use it to disambiguate before selecting by time.
        same_region = [item for item in temporal_candidates if item[1]["true_region_id"] == row["region_id"]]
        nearby = min(same_region or temporal_candidates, key=lambda item: abs((item[0] - timestamp).total_seconds()))
        truth = nearby[1]
        if row["region_id"] != truth["true_region_id"]:
            region_mismatches += 1
        passive_errors.append(
            haversine_km(float(row["latitude"]), float(row["longitude"]), float(truth["true_latitude"]), float(truth["true_longitude"])) * 1000
        )
    match_rate = 1.0 - unmatched_passive / max(1, len(passive))
    checks.append(check("Passive-to-trajectory matching", match_rate >= 0.99, round(match_rate, 5), "At least 99% of passive events match a true stop within 15 minutes", "coherence"))
    checks.append(check("Passive region consistency", region_mismatches == 0, region_mismatches, "Matched passive events retain the true region label", "coherence"))
    median_gps_error = percentile(passive_errors, 0.50)
    p95_gps_error = percentile(passive_errors, 0.95)
    checks.append(check("GPS noise is present", 5.0 <= median_gps_error <= 250.0, {"median_m": round(median_gps_error, 2), "p95_m": round(p95_gps_error, 2)}, "Median passive displacement is nonzero and plausible for configured synthetic noise", "coherence"))

    # Mechanism diagnostics. These establish that the generated data contains
    # the intended signals; they are not embedding-model results.
    chosen = [row for row in candidates if row["is_chosen"] == "1"]
    alternatives = [row for row in candidates if row["is_chosen"] == "0"]
    chosen_distance = mean(float(row["distance_km"]) for row in chosen)
    candidate_distance = mean(float(row["distance_km"]) for row in candidates)
    checks.append(check("Choice responds to distance", chosen_distance < candidate_distance, {"chosen_mean_km": round(chosen_distance, 3), "candidate_mean_km": round(candidate_distance, 3)}, "Chosen POIs are nearer on average than the candidate pool", "mechanism"))

    chosen_exposure = mean(float(row["exposed"]) for row in chosen)
    candidate_exposure = mean(float(row["exposed"]) for row in candidates)
    checks.append(check("Choice responds to exposure", chosen_exposure > candidate_exposure, {"chosen_share": round(chosen_exposure, 3), "candidate_share": round(candidate_exposure, 3)}, "Chosen POIs are more often exposed than candidates overall", "mechanism"))

    chosen_utility = mean(float(row["utility_total"]) for row in chosen)
    alternative_utility = mean(float(row["utility_total"]) for row in alternatives)
    checks.append(check("Choice responds to utility", chosen_utility > alternative_utility, {"chosen_mean": round(chosen_utility, 3), "alternative_mean": round(alternative_utility, 3)}, "Chosen alternatives have higher mean total utility", "mechanism"))

    negative_residuals = 0
    for row in candidates:
        deterministic = (
            float(row["utility_preference"])
            - float(row["utility_price_penalty"])
            - float(row["utility_distance_penalty"])
            + float(row["utility_exposure"])
        )
        if float(row["utility_total"]) + 2e-5 < deterministic:
            negative_residuals += 1
    checks.append(check("Utility decomposition", negative_residuals == 0, negative_residuals, "Recorded stochastic residual is non-negative for every candidate", "mechanism"))

    events_by_user = Counter(row["user_id"] for row in events)
    engagement_by_user = {row["user_id"]: float(row["digital_engagement"]) for row in latents}
    engagement_correlation = pearson(
        [engagement_by_user[user_id] for user_id in sorted(user_ids)],
        [events_by_user[user_id] for user_id in sorted(user_ids)],
    )
    checks.append(check("Non-random observability", engagement_correlation > 0.15, round(engagement_correlation, 3), "Digital engagement positively correlates with recorded event volume", "mechanism"))

    intent_counts = Counter(row["primary_intent"] for row in episodes)
    required_intents = {"routine", "shopping", "leisure", "family_outing", "travel"}
    checks.append(check("Episode coverage", required_intents <= set(intent_counts), dict(intent_counts), "All five intended episode types appear", "coverage"))

    services_per_user: defaultdict[str, set[str]] = defaultdict(set)
    for row in events:
        services_per_user[row["user_id"]].add(row["service_id"])
    multiservice_share = mean(len(services_per_user[user_id]) >= 2 for user_id in user_ids)
    checks.append(check("Cross-service cohort", multiservice_share >= 0.50, round(multiservice_share, 3), "At least half of users have observations in two or more services", "coverage"))

    service_counts = Counter(row["service_id"] for row in events)
    travel_share = service_counts["travel"] / max(1, len(events))
    checks.append(check("Travel signal volume", service_counts["travel"] >= 100, {"events": service_counts["travel"], "share": round(travel_share, 4)}, "At least 100 travel events in the pilot; still treat this as a sparse probe", "coverage", "warning"))

    error_failures = [item for item in checks if not item["passed"] and item["severity"] == "error"]
    warning_failures = [item for item in checks if not item["passed"] and item["severity"] == "warning"]
    return {
        "status": "passed" if not error_failures else "failed",
        "dataset": str(root.resolve()),
        "scenario": manifest["scenario"],
        "simulator_version": manifest["simulator_version"],
        "summary": {
            "checks_passed": sum(item["passed"] for item in checks),
            "checks_total": len(checks),
            "errors": len(error_failures),
            "warnings": len(warning_failures),
            "users": len(users),
            "observed_events": len(events),
            "choices": len(choices),
            "passive_trace_match_rate": round(match_rate, 5),
            "median_gps_error_m": round(median_gps_error, 2),
        },
        "checks": checks,
        "distributions": {
            "services": dict(service_counts),
            "modes": dict(Counter(row["observation_mode"] for row in events)),
            "intents": dict(intent_counts),
        },
    }


def compare_scenarios(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def value(report: dict[str, Any], check_name: str, field: str) -> float:
        item = next(item for item in report["checks"] if item["name"] == check_name)
        return float(item["value"][field])

    comparisons: list[dict[str, Any]] = []
    if {"mixed", "opportunity_confounded"} <= reports.keys():
        mixed = value(reports["mixed"], "Choice responds to distance", "chosen_mean_km")
        stressed = value(reports["opportunity_confounded"], "Choice responds to distance", "chosen_mean_km")
        comparisons.append(check("Opportunity scenario shortens chosen distance", stressed < mixed, {"mixed_km": mixed, "opportunity_km": stressed}, "Opportunity-confounded mean chosen distance is below mixed", "scenario"))
    if {"mixed", "exposure_confounded"} <= reports.keys():
        mixed = value(reports["mixed"], "Choice responds to exposure", "chosen_share")
        stressed = value(reports["exposure_confounded"], "Choice responds to exposure", "chosen_share")
        comparisons.append(check("Exposure scenario raises exposed choices", stressed > mixed, {"mixed_share": mixed, "exposure_share": stressed}, "Exposure-confounded chosen exposure share is above mixed", "scenario"))
    if {"mixed", "observation_biased"} <= reports.keys():
        mixed_count = reports["mixed"]["summary"]["observed_events"]
        biased_count = reports["observation_biased"]["summary"]["observed_events"]
        mixed_corr = float(next(item for item in reports["mixed"]["checks"] if item["name"] == "Non-random observability")["value"])
        biased_corr = float(next(item for item in reports["observation_biased"]["checks"] if item["name"] == "Non-random observability")["value"])
        comparisons.append(check("Observation-bias scenario changes observability", biased_count < mixed_count and biased_corr > mixed_corr, {"mixed_events": mixed_count, "biased_events": biased_count, "mixed_correlation": mixed_corr, "biased_correlation": biased_corr}, "Observation-biased world records fewer events with stronger engagement correlation", "scenario"))
    return {
        "status": "passed" if comparisons and all(item["passed"] for item in comparisons) else "failed",
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Simulator output directory")
    parser.add_argument("--compare", action="append", default=[], metavar="NAME=PATH", help="Add scenario dataset to aggregate comparison")
    parser.add_argument("--output", type=Path, help="Write the JSON report here")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate(args.dataset.resolve())
    if args.compare:
        reports = {report["scenario"]: report}
        for item in args.compare:
            if "=" not in item:
                raise SystemExit(f"Invalid --compare value: {item}; use NAME=PATH")
            name, raw_path = item.split("=", 1)
            reports[name] = validate(Path(raw_path).resolve())
        report["scenario_validation"] = compare_scenarios(reports)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "passed" or report.get("scenario_validation", {}).get("status", "passed") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
