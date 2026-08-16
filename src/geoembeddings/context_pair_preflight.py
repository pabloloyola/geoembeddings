"""Observed-only context-session pair construction and preflight authentication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from . import __version__
from .config import load_config, load_mapping_config
from .contract import CONTEXT_PAIR_MANIFEST_SCHEMA, CONTEXT_PAIR_PREFLIGHT_SCHEMA
from .io import sha256_file
from .layout import DatasetLayout, ExperimentLayout
from .prepare import _temporal_boundaries
from .schema import EVENT_FILE, USER_FILE, load_observed
from .user_roles import authenticate_roles, protocol_config


RELATIONS = ("positive", "negative")
PAIR_FIELDS = (
    "pair_id", "relation", "user_id", "anchor_group_id", "paired_group_id",
    "anchor_timestamp", "paired_timestamp", "anchor_session_id",
    "paired_session_id", "anchor_history_event_count",
    "paired_history_event_count", "gap_hours", "intervening_group_count",
)


def _canonical_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("context preflight requires timezone-aware timestamps")
    return timestamp.tz_convert("UTC").isoformat()


def _group_id(user_id: str, timestamp: str) -> str:
    return hashlib.sha256(f"context-group\0{user_id}\0{timestamp}".encode()).hexdigest()


def _session_id(user_id: str, ordinal: int) -> str:
    return hashlib.sha256(f"observed-session\0{user_id}\0{ordinal}".encode()).hexdigest()


def _pair_key(pair: Mapping[str, Any]) -> tuple[str, str, str]:
    groups = tuple(sorted((str(pair["anchor_group_id"]), str(pair["paired_group_id"]))))
    return str(pair["user_id"]), groups[0], groups[1]


def _pair_id(relation: str, key: tuple[str, str, str]) -> str:
    material = f"{CONTEXT_PAIR_MANIFEST_SCHEMA}\0{relation}\0" + "\0".join(key)
    return hashlib.sha256(material.encode()).hexdigest()


def _selection_key(seed: int, relation: str, user_id: str, left: str, right: str) -> str:
    return hashlib.sha256(f"{seed}\0{relation}\0{user_id}\0{left}\0{right}".encode()).hexdigest()


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    series = pd.Series(values, dtype=float)
    return {
        "count": int(len(values)),
        "min": float(series.min()),
        "p25": float(series.quantile(.25)),
        "median": float(series.quantile(.50)),
        "p75": float(series.quantile(.75)),
        "p95": float(series.quantile(.95)),
        "max": float(series.max()),
    }


def _local_calendar_day(value: Any, timezone_name: str) -> Any:
    try:
        timezone_value = ZoneInfo(str(timezone_name))
    except Exception as exc:
        raise ValueError(f"unknown local-day timezone: {timezone_name!r}") from exc
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("local-day evaluation requires timezone-aware timestamps")
    return timestamp.tz_convert(timezone_value).date()


def _validate_pair_identity(pair: Mapping[str, Any]) -> None:
    missing = set(PAIR_FIELDS) - set(pair)
    if missing:
        raise ValueError(f"context pair is missing fields: {sorted(missing)}")
    if pair["relation"] not in RELATIONS:
        raise ValueError(f"unsupported context pair relation: {pair['relation']!r}")
    if not str(pair["user_id"]):
        raise ValueError("context pair user identity is empty")
    if pair["anchor_group_id"] == pair["paired_group_id"]:
        raise ValueError("context pair identity reuses one timestamp group")
    anchor = pd.Timestamp(pair["anchor_timestamp"])
    paired = pd.Timestamp(pair["paired_timestamp"])
    if anchor.tzinfo is None or paired.tzinfo is None or anchor >= paired:
        raise ValueError("context pair violates strict timestamp ordering")
    if float(pair["gap_hours"]) <= 0:
        raise ValueError("context pair has a non-positive timestamp gap")
    if int(pair["anchor_history_event_count"]) < 0 or int(pair["paired_history_event_count"]) < 0:
        raise ValueError("context pair has a negative history count")
    if pair["relation"] == "positive":
        if pair["anchor_session_id"] != pair["paired_session_id"]:
            raise ValueError("positive context pair crosses observed sessions")
        if int(pair["intervening_group_count"]) < 1:
            raise ValueError("positive context pair has no intervening timestamp group")
    else:
        if pair["anchor_session_id"] == pair["paired_session_id"]:
            raise ValueError("negative context pair stays within one observed session")
    key = _pair_key(pair)
    if str(pair["pair_id"]) != _pair_id(str(pair["relation"]), key):
        raise ValueError("context pair identity hash is inconsistent")


def validate_context_pair_manifest(value: Mapping[str, Any]) -> None:
    """Validate pair identities without opening a dataset or protected truth."""
    if not isinstance(value, Mapping) or value.get("schema_version") != CONTEXT_PAIR_MANIFEST_SCHEMA:
        raise ValueError("unsupported context pair manifest schema")
    for section in ("source_authentication", "preparation_authentication", "pair_configuration"):
        if not isinstance(value.get(section), Mapping):
            raise ValueError(f"context pair manifest is missing {section}")
    auth = value["source_authentication"]
    if auth.get("truth_files_opened") is not False:
        raise ValueError("context pair manifest does not authenticate the observed-only boundary")
    pairs = value.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("context pair manifest pairs must be a list")
    pair_ids: set[str] = set()
    identities: dict[tuple[str, str, str], str] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("context pair manifest contains a malformed pair")
        _validate_pair_identity(pair)
        pair_id = str(pair["pair_id"])
        key = _pair_key(pair)
        if pair_id in pair_ids:
            raise ValueError("duplicate context pair identity")
        if key in identities:
            raise ValueError("conflicting context pair identity")
        pair_ids.add(pair_id)
        identities[key] = str(pair["relation"])


def _pair_record(relation: str, anchor: Mapping[str, Any], paired: Mapping[str, Any]) -> dict[str, Any]:
    key = (str(anchor["user_id"]), *sorted((str(anchor["group_id"]), str(paired["group_id"]))))
    anchor_timestamp = pd.Timestamp(anchor["timestamp"])
    paired_timestamp = pd.Timestamp(paired["timestamp"])
    return {
        "pair_id": _pair_id(relation, key),
        "relation": relation,
        "user_id": str(anchor["user_id"]),
        "anchor_group_id": str(anchor["group_id"]),
        "paired_group_id": str(paired["group_id"]),
        "anchor_timestamp": _canonical_timestamp(anchor_timestamp),
        "paired_timestamp": _canonical_timestamp(paired_timestamp),
        "anchor_session_id": str(anchor["session_id"]),
        "paired_session_id": str(paired["session_id"]),
        "anchor_history_event_count": int(anchor["history_event_count"]),
        "paired_history_event_count": int(paired["history_event_count"]),
        "gap_hours": float((paired_timestamp - anchor_timestamp).total_seconds() / 3600.0),
        "intervening_group_count": int(paired["position"] - anchor["position"] - 1),
    }


def build_context_pairs(
    events: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    min_history_events: int,
    session_gap_hours: float,
    min_intervening_groups_for_positive: int,
    positive_pairs_per_anchor: int,
    negative_pairs_per_anchor: int,
    seed: int,
    positive_local_day_timezone: str | None = None,
    positive_same_local_day: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic pairs from observed events at or before train_end."""
    if min_history_events < 1 or session_gap_hours <= 0:
        raise ValueError("invalid context pair history/session configuration")
    if min_intervening_groups_for_positive < 1:
        raise ValueError("positive context pairs require at least one intervening group")
    if positive_pairs_per_anchor < 1 or negative_pairs_per_anchor < 1:
        raise ValueError("positive and negative pair caps must be positive")
    cutoff = pd.Timestamp(train_end)
    if cutoff.tzinfo is None:
        raise ValueError("training cutoff must be timezone-aware")
    frame = events.copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].isna().any():
        raise ValueError("observed events contain invalid timestamps")
    if frame.duplicated(keep=False).any():
        raise ValueError("duplicate observed event records have conflicting identities")
    cross_cutoff_count = int((frame["timestamp"] > cutoff).sum())
    frame = frame.loc[frame["timestamp"] <= cutoff].copy()
    frame = frame.sort_values(["user_id", "timestamp"], kind="stable")

    groups: dict[str, list[dict[str, Any]]] = {}
    excluded = {
        "insufficient_history": 0,
        "no_same_session_positive": 0,
        "no_different_session_negative": 0,
        "cross_cutoff": cross_cutoff_count,
    }
    for user_id, user_frame in frame.groupby("user_id", sort=True):
        user_groups: list[dict[str, Any]] = []
        history_count = 0
        session_ordinal = -1
        previous_timestamp: pd.Timestamp | None = None
        for timestamp, group in user_frame.groupby("timestamp", sort=True):
            timestamp = pd.Timestamp(timestamp)
            if previous_timestamp is None or (timestamp - previous_timestamp).total_seconds() / 3600.0 > session_gap_hours:
                session_ordinal += 1
            timestamp_text = _canonical_timestamp(timestamp)
            user_groups.append({
                "user_id": str(user_id),
                "timestamp": timestamp,
                "timestamp_text": timestamp_text,
                "group_id": _group_id(str(user_id), timestamp_text),
                "event_count": int(len(group)),
                "history_event_count": history_count,
                "session_id": _session_id(str(user_id), session_ordinal),
                "position": len(user_groups),
            })
            history_count += int(len(group))
            previous_timestamp = timestamp
        groups[str(user_id)] = user_groups

    pairs: list[dict[str, Any]] = []
    valid_anchor_count = 0
    positive_anchor_count = 0
    negative_anchor_count = 0
    user_positive: set[str] = set()
    user_negative: set[str] = set()
    for user_id in sorted(groups):
        user_groups = groups[user_id]
        valid = [group for group in user_groups if group["history_event_count"] >= min_history_events]
        excluded["insufficient_history"] += len(user_groups) - len(valid)
        valid_anchor_count += len(valid)
        for anchor in valid:
            later = [candidate for candidate in valid if candidate["position"] > anchor["position"]]
            positive_candidates = [candidate for candidate in later
                                   if candidate["session_id"] == anchor["session_id"]
                                   and candidate["position"] - anchor["position"] - 1 >= min_intervening_groups_for_positive]
            if positive_same_local_day:
                if not positive_local_day_timezone:
                    raise ValueError("positive_same_local_day requires positive_local_day_timezone")
                positive_candidates = [
                    candidate for candidate in positive_candidates
                    if _local_calendar_day(anchor["timestamp"], positive_local_day_timezone)
                    == _local_calendar_day(candidate["timestamp"], positive_local_day_timezone)
                ]
            negative_candidates = [candidate for candidate in later
                                   if candidate["session_id"] != anchor["session_id"]]
            if not positive_candidates:
                excluded["no_same_session_positive"] += 1
            else:
                positive_anchor_count += 1
                user_positive.add(user_id)
                selected = sorted(
                    positive_candidates,
                    key=lambda candidate: _selection_key(seed, "positive", user_id, anchor["group_id"], candidate["group_id"]),
                )[:positive_pairs_per_anchor]
                pairs.extend(_pair_record("positive", anchor, candidate) for candidate in selected)
            if not negative_candidates:
                excluded["no_different_session_negative"] += 1
            else:
                negative_anchor_count += 1
                user_negative.add(user_id)
                selected = sorted(
                    negative_candidates,
                    key=lambda candidate: _selection_key(seed, "negative", user_id, anchor["group_id"], candidate["group_id"]),
                )[:negative_pairs_per_anchor]
                pairs.extend(_pair_record("negative", anchor, candidate) for candidate in selected)

    pairs.sort(key=lambda pair: (pair["relation"], pair["user_id"], pair["anchor_timestamp"], pair["paired_timestamp"]))
    manifest_identity = {
        "duplicate_pair_count": 0,
        "reversed_duplicate_pair_count": 0,
        "same_anchor_positive_negative_conflicts": 0,
        "same_pair_positive_negative_conflicts": 0,
        "same_timestamp_pair_count": 0,
        "identical_session_pair_count": 0,
    }
    coverage = {
        "users_seen": int(frame["user_id"].nunique()),
        "users_with_training_events": int(frame["user_id"].nunique()),
        "users_with_valid_anchors": int(sum(1 for user in groups.values() if any(g["history_event_count"] >= min_history_events for g in user))),
        "users_with_positive_pairs": len(user_positive),
        "users_with_negative_pairs": len(user_negative),
        "valid_anchor_count": valid_anchor_count,
        "positive_anchor_count": positive_anchor_count,
        "negative_anchor_count": negative_anchor_count,
        "positive_pair_count": sum(pair["relation"] == "positive" for pair in pairs),
        "negative_pair_count": sum(pair["relation"] == "negative" for pair in pairs),
        "positive_anchor_coverage": positive_anchor_count / max(1, valid_anchor_count),
        "negative_anchor_coverage": negative_anchor_count / max(1, valid_anchor_count),
        "session_count": len({group["session_id"] for user in groups.values() for group in user}),
        "timestamp_group_count": sum(len(user) for user in groups.values()),
    }
    gap_values = {
        relation: [float(pair["gap_hours"]) for pair in pairs if pair["relation"] == relation]
        for relation in RELATIONS
    }
    diagnostics = {
        "coverage": coverage,
        "exclusions": {"total": int(sum(excluded.values())), "by_reason": excluded},
        "gap_distributions_hours": {
            "positive": _distribution(gap_values["positive"]),
            "negative": _distribution(gap_values["negative"]),
            "positive_intervening_group_count": _distribution([
                float(pair["intervening_group_count"]) for pair in pairs if pair["relation"] == "positive"
            ]),
            "sessions_per_user": _distribution([
                float(len({group["session_id"] for group in user})) for user in groups.values()
            ]),
        },
        "duplicate_and_conflict_checks": manifest_identity,
        "matching_and_fallback": {
            "negative_rule": "same_user_different_observed_session",
            "same_user_negative_match_attempts": valid_anchor_count,
            "same_user_negative_matches": negative_anchor_count,
            "same_user_negative_match_rate": negative_anchor_count / max(1, valid_anchor_count),
            "fallback_attempts": 0,
            "fallback_used": 0,
            "fallback_rate": 0.0,
            "cross_user_matching": "not_applicable",
        },
    }
    if not coverage["positive_pair_count"] or not coverage["negative_pair_count"]:
        raise ValueError("context pair preflight found zero usable positive or negative coverage")
    return pairs, diagnostics


def _sensitivity_groups(
    events: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    session_gap_hours: float,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, int]:
    """Create atomic observed timestamp groups for one session definition."""
    if session_gap_hours <= 0:
        raise ValueError("session gap must be positive")
    cutoff = pd.Timestamp(train_end)
    if cutoff.tzinfo is None:
        raise ValueError("training cutoff must be timezone-aware")
    frame = events.copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].isna().any():
        raise ValueError("observed events contain invalid timestamps")
    if frame.duplicated(keep=False).any():
        raise ValueError("duplicate observed event records have conflicting identities")
    cross_cutoff_count = int((frame["timestamp"] > cutoff).sum())
    frame = frame.loc[frame["timestamp"] <= cutoff].copy()
    frame = frame.sort_values(["user_id", "timestamp"], kind="stable")

    groups: dict[str, list[dict[str, Any]]] = {}
    for user_id, user_frame in frame.groupby("user_id", sort=True):
        user_groups: list[dict[str, Any]] = []
        history_count = 0
        session_ordinal = -1
        previous_timestamp: pd.Timestamp | None = None
        for timestamp, group in user_frame.groupby("timestamp", sort=True):
            timestamp = pd.Timestamp(timestamp)
            gap_hours = None if previous_timestamp is None else (
                (timestamp - previous_timestamp).total_seconds() / 3600.0
            )
            if gap_hours is None or gap_hours > session_gap_hours:
                session_ordinal += 1
            timestamp_text = _canonical_timestamp(timestamp)
            services = []
            if "service_id" in group:
                services = sorted({str(value) for value in group["service_id"].dropna()})
            user_groups.append({
                "user_id": str(user_id),
                "timestamp": timestamp,
                "timestamp_text": timestamp_text,
                "group_id": _group_id(str(user_id), timestamp_text),
                "event_count": int(len(group)),
                "history_event_count": history_count,
                "services": services,
                "previous_gap_hours": gap_hours,
                "session_id": _session_id(str(user_id), session_ordinal),
                "position": len(user_groups),
            })
            history_count += int(len(group))
            previous_timestamp = timestamp
        groups[str(user_id)] = user_groups
    return groups, frame, cross_cutoff_count


def _decile_coverage(
    training_user_ids: set[str],
    training_event_counts: Mapping[str, int],
    positive_pairs_by_user: Counter[str],
) -> list[dict[str, Any]]:
    ordered = sorted(
        ((user_id, int(training_event_counts.get(user_id, 0))) for user_id in training_user_ids),
        key=lambda item: (item[1], item[0]),
    )
    if not ordered:
        return []
    rows: list[dict[str, Any]] = []
    for decile in range(1, 11):
        members = [
            (user_id, count) for rank, (user_id, count) in enumerate(ordered)
            if min(10, math.floor(rank * 10 / len(ordered)) + 1) == decile
        ]
        if not members:
            continue
        pair_count = sum(positive_pairs_by_user[user_id] for user_id, _ in members)
        covered_users = sum(positive_pairs_by_user[user_id] > 0 for user_id, _ in members)
        rows.append({
            "decile": decile,
            "user_count": len(members),
            "training_event_count_min": min(count for _, count in members),
            "training_event_count_max": max(count for _, count in members),
            "users_with_positive": covered_users,
            "positive_pair_count": pair_count,
            "positive_user_coverage": covered_users / len(members),
        })
    return rows


def build_session_definition_sensitivity(
    events: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    min_history_events: int,
    training_user_ids: set[str],
    session_gap_hours: list[float],
    min_intervening_groups: list[int],
    positive_pairs_per_anchor: int,
    seed: int,
    daily_gap_flag_hours: float,
    adjacent_prefix_flag_share: float,
    high_overlap_ratio: float,
    high_overlap_flag_share: float,
    daily_span_flag_share: float,
) -> list[dict[str, Any]]:
    """Report deterministic positive-pair support across observed session definitions."""
    if min_history_events < 1 or positive_pairs_per_anchor < 1:
        raise ValueError("invalid sensitivity history or pair cap")
    if not training_user_ids:
        raise ValueError("sensitivity requires authenticated training users")
    training_user_ids = {str(user_id) for user_id in training_user_ids}
    training_event_counts = (
        events.assign(user_id=events["user_id"].astype(str))
        .loc[lambda frame: frame["user_id"].isin(training_user_ids)]
        .groupby("user_id").size().to_dict()
    )
    results: list[dict[str, Any]] = []
    for gap in session_gap_hours:
        for min_intervening in min_intervening_groups:
            if float(gap) <= 0 or int(min_intervening) < 0:
                raise ValueError("invalid sensitivity session definition")
            groups, frame, cross_cutoff_count = _sensitivity_groups(
                events,
                train_end=train_end,
                session_gap_hours=float(gap),
            )
            pairs: list[dict[str, Any]] = []
            positive_pairs_by_user: Counter[str] = Counter()
            valid_anchor_count = 0
            positive_anchor_count = 0
            excluded_insufficient = 0
            excluded_no_positive = 0
            group_service_signatures: set[tuple[str, ...]] = set()
            for user_groups in groups.values():
                for group in user_groups:
                    group_service_signatures.add(tuple(group["services"]))
                valid = [
                    group for group in user_groups
                    if group["history_event_count"] >= min_history_events
                ]
                excluded_insufficient += len(user_groups) - len(valid)
                valid_anchor_count += len(valid)
                for anchor in valid:
                    candidates = [
                        candidate for candidate in valid
                        if candidate["position"] > anchor["position"]
                        and candidate["session_id"] == anchor["session_id"]
                        and candidate["position"] - anchor["position"] - 1 >= int(min_intervening)
                    ]
                    if not candidates:
                        excluded_no_positive += 1
                        continue
                    positive_anchor_count += 1
                    user_id = str(anchor["user_id"])
                    selected = sorted(
                        candidates,
                        key=lambda candidate: _selection_key(
                            seed,
                            f"sensitivity-positive-{float(gap):g}-{int(min_intervening)}",
                            user_id,
                            anchor["group_id"],
                            candidate["group_id"],
                        ),
                    )[:positive_pairs_per_anchor]
                    for candidate in selected:
                        between = [
                            group["previous_gap_hours"] for group in user_groups
                            if anchor["position"] < group["position"] <= candidate["position"]
                            and group["previous_gap_hours"] is not None
                        ]
                        anchor_history = int(anchor["history_event_count"])
                        paired_history = int(candidate["history_event_count"])
                        overlap_ratio = anchor_history / max(1, paired_history)
                        record = {
                            "user_id": user_id,
                            "anchor_group_id": str(anchor["group_id"]),
                            "paired_group_id": str(candidate["group_id"]),
                            "anchor_timestamp": _canonical_timestamp(anchor["timestamp"]),
                            "paired_timestamp": _canonical_timestamp(candidate["timestamp"]),
                            "anchor_history_event_count": anchor_history,
                            "paired_history_event_count": paired_history,
                            "gap_hours": float((candidate["timestamp"] - anchor["timestamp"]).total_seconds() / 3600.0),
                            "intervening_group_count": int(candidate["position"] - anchor["position"] - 1),
                            "anchor_services": list(anchor["services"]),
                            "paired_services": list(candidate["services"]),
                            "overlap_ratio": float(overlap_ratio),
                            "max_intervening_gap_hours": float(max(between, default=0.0)),
                            "crosses_calendar_day": anchor["timestamp"].date() != candidate["timestamp"].date(),
                        }
                        pairs.append(record)
                        positive_pairs_by_user[user_id] += 1
            pairs.sort(key=lambda pair: (pair["user_id"], pair["anchor_timestamp"], pair["paired_timestamp"]))
            pair_count = len(pairs)
            positive_users = sum(count > 0 for count in positive_pairs_by_user.values())
            pair_counts = sorted(
                ((user_id, positive_pairs_by_user[user_id]) for user_id in training_user_ids),
                key=lambda item: (-item[1], item[0]),
            )
            top_user_count = max(1, math.ceil(len(training_user_ids) * 0.10))
            top_user_pairs = sum(count for _, count in pair_counts[:top_user_count])
            adjacent_share = sum(pair["intervening_group_count"] == 0 for pair in pairs) / max(1, pair_count)
            overlap_share = sum(pair["overlap_ratio"] >= high_overlap_ratio for pair in pairs) / max(1, pair_count)
            daily_span_share = sum(
                pair["max_intervening_gap_hours"] >= daily_gap_flag_hours for pair in pairs
            ) / max(1, pair_count)
            day_cross_count = sum(pair["crosses_calendar_day"] for pair in pairs)
            service_combinations = Counter(
                (tuple(pair["anchor_services"]), tuple(pair["paired_services"])) for pair in pairs
            )
            combination_rows = [
                {
                    "anchor_services": list(key[0]),
                    "paired_services": list(key[1]),
                    "positive_pair_count": count,
                }
                for key, count in sorted(service_combinations.items())
            ]
            flags = []
            if adjacent_share >= adjacent_prefix_flag_share:
                flags.append("adjacent_prefix_dominant")
            if overlap_share >= high_overlap_flag_share:
                flags.append("highly_overlapping_pairs")
            if daily_span_share >= daily_span_flag_share:
                flags.append("merges_clearly_distinct_daily_activity")
            results.append({
                "setting": {
                    "session_gap_hours": float(gap),
                    "min_intervening_timestamp_groups": int(min_intervening),
                },
                "sessions": int(len({group["session_id"] for user in groups.values() for group in user})),
                "timestamp_groups": int(sum(len(user) for user in groups.values())),
                "positive_pair_count": pair_count,
                "positive_anchor_count": positive_anchor_count,
                "valid_anchor_count": valid_anchor_count,
                "positive_anchor_coverage": positive_anchor_count / max(1, valid_anchor_count),
                "users_with_at_least_one_positive": positive_users,
                "positive_time_gap_distribution_hours": _distribution([pair["gap_hours"] for pair in pairs]),
                "training_event_count_decile_coverage": _decile_coverage(
                    training_user_ids, training_event_counts, positive_pairs_by_user,
                ),
                "service_combination_coverage": {
                    "definition": "ordered anchor timestamp-group service set x paired timestamp-group service set",
                    "observed_group_service_set_count": len(group_service_signatures),
                    "unique_service_combinations_covered": len(service_combinations),
                    "possible_service_combinations_denominator": len(group_service_signatures) ** 2,
                    "coverage_fraction": len(service_combinations) / max(1, len(group_service_signatures) ** 2),
                    "combinations": combination_rows,
                },
                "calendar_day_boundary": {
                    "pairs_crossing_calendar_day": day_cross_count,
                    "pair_share": day_cross_count / max(1, pair_count),
                    "any_pairs_crossing": bool(day_cross_count),
                },
                "pair_concentration": {
                    "top_user_fraction": 0.10,
                    "top_user_count": top_user_count,
                    "top_user_positive_pairs": top_user_pairs,
                    "top_user_positive_share": top_user_pairs / max(1, pair_count),
                },
                "exclusions": {
                    "cross_cutoff_events": cross_cutoff_count,
                    "insufficient_history_timestamp_groups": excluded_insufficient,
                    "valid_anchors_without_positive": excluded_no_positive,
                },
                "diagnostic_flags": {
                    "adjacent_prefix_share": adjacent_share,
                    "high_overlap_share": overlap_share,
                    "long_within_session_gap_share": daily_span_share,
                    "thresholds": {
                        "adjacent_prefix_flag_share": adjacent_prefix_flag_share,
                        "high_overlap_ratio": high_overlap_ratio,
                        "high_overlap_flag_share": high_overlap_flag_share,
                        "daily_gap_flag_hours": daily_gap_flag_hours,
                        "daily_span_flag_share": daily_span_flag_share,
                    },
                    "flags": flags,
                },
            })
    return results


def _auth_metadata(
    run_dir: str | Path,
    experiment_dir: str | Path,
    embedding_config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], set[str]]:
    run = DatasetLayout.from_path(run_dir)
    manifest = run.validate(require_truth=False)
    experiment = ExperimentLayout.from_path(experiment_dir)
    if not experiment.prepared_metadata.is_file() or not experiment.resolved_config.is_file():
        raise FileNotFoundError("authenticated prepared metadata/config is missing")
    metadata = json.loads(experiment.prepared_metadata.read_text(encoding="utf-8"))
    prepared_config = load_config(experiment.resolved_config)
    embedding_config = load_config(Path(embedding_config_path).expanduser().resolve())
    users, events = load_observed(run.observed)
    if metadata.get("run_dir") and Path(metadata["run_dir"]).resolve() != run.root:
        raise ValueError("prepared metadata run identity does not match run-dir")
    if metadata.get("observed_dir") and Path(metadata["observed_dir"]).resolve() != run.observed:
        raise ValueError("prepared metadata observed identity does not match run-dir")
    if metadata.get("dataset_contract") != manifest.get("dataset_contract"):
        raise ValueError("prepared metadata dataset contract does not match run manifest")
    declared_sources = metadata.get("source_files")
    if not isinstance(declared_sources, Mapping) or not declared_sources:
        raise ValueError("prepared metadata has no authenticated observed source hashes")
    if set(declared_sources) != {USER_FILE, EVENT_FILE}:
        raise ValueError("prepared metadata observed source identity is incomplete")
    actual_sources = {}
    for name, expected in declared_sources.items():
        path = run.observed / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError("prepared metadata source authentication failed")
        actual_sources[str(name)] = sha256_file(path)
    if set(actual_sources) != set(declared_sources):
        raise ValueError("prepared metadata source identity is incomplete")
    for field in ("categorical_fields", "continuous_fields", "include_object_id", "max_sequence_length", "min_history_events", "train_fraction", "validation_fraction", "train_end", "validation_end"):
        prepared_value = prepared_config.get("data", {}).get(field)
        supplied_value = embedding_config.get("data", {}).get(field)
        if prepared_value != supplied_value:
            raise ValueError(f"prepared/configuration authentication failed for data.{field}")
    prepared_protocol = protocol_config(prepared_config)
    supplied_protocol = protocol_config(embedding_config)
    if (prepared_protocol is None) != (supplied_protocol is None) or (
        prepared_protocol is not None and dict(prepared_protocol) != dict(supplied_protocol)
    ):
        raise ValueError("prepared/configuration user-role protocol authentication failed")
    assignments = authenticate_roles(metadata, embedding_config, users["user_id"].astype(str))
    train_end = pd.Timestamp(metadata.get("train_end"))
    validation_end = pd.Timestamp(metadata.get("validation_end"))
    if train_end.tzinfo is None or validation_end.tzinfo is None or train_end >= validation_end:
        raise ValueError("prepared metadata has invalid authenticated cutoffs")
    expected_train_end, expected_validation_end = _temporal_boundaries(events, embedding_config["data"])
    if train_end != expected_train_end or validation_end != expected_validation_end:
        raise ValueError("prepared metadata cutoffs do not match the authenticated observed split")
    if assignments is None:
        train_users = set(users["user_id"].astype(str))
    else:
        train_users = {user for user, role in assignments.items() if role == "target_train"}
    train_events = events.loc[
        events["user_id"].astype(str).isin(train_users) & (events["timestamp"] <= train_end)
    ].copy()
    expected_train_events = metadata.get("target_events_by_split", {}).get("train")
    if expected_train_events is not None and int(expected_train_events) != len(train_events):
        raise ValueError("prepared metadata training cutoff/user-role identity does not match observed events")
    auth = {
        "source_authentication": {
            "run_dir": str(run.root),
            "run_manifest_sha256": sha256_file(run.manifest_path),
            "simulation_config_sha256": manifest.get("config_sha256"),
            "dataset_contract_version": manifest.get("dataset_contract", {}).get("version"),
            "simulator_version": manifest.get("simulator_version"),
            "simulator_seed": manifest.get("seed"),
            "user_count": int(len(users)),
            "duration_days": manifest.get("days"),
            "observed_file_hashes": actual_sources,
            "truth_files_opened": False,
        },
        "preparation_authentication": {
            "prepared_metadata_path": str(experiment.prepared_metadata),
            "prepared_metadata_sha256": sha256_file(experiment.prepared_metadata),
            "prepared_config_sha256": sha256_file(experiment.resolved_config),
            "embedding_config_sha256": sha256_file(embedding_config_path),
            "observed_source_hashes": actual_sources,
            "train_cutoff": _canonical_timestamp(train_end),
            "validation_cutoff": _canonical_timestamp(validation_end),
            "training_user_count": len(train_users),
            "training_event_count": len(train_events),
            "min_history_events": int(embedding_config["data"]["min_history_events"]),
            "user_role_protocol": metadata.get("user_role_protocol"),
        },
    }
    return users, train_events, auth, embedding_config, train_users


def _load_pair_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_mapping_config(path)
    if config.get("schema_version") != "geoembeddings-context-session-preflight/1.0":
        raise ValueError("unsupported context-session preflight configuration")
    pairing = config.get("pairing")
    if not isinstance(pairing, Mapping):
        raise ValueError("context-session preflight pairing configuration is missing")
    required = {
        "session_gap_hours", "min_intervening_groups_for_positive",
        "positive_pairs_per_anchor", "negative_pairs_per_anchor", "seed",
        "positive_local_day_timezone", "positive_same_local_day",
    }
    if set(pairing) != required:
        raise ValueError("context-session preflight pairing configuration is not frozen")
    sensitivity = config.get("sensitivity")
    sensitivity_required = {
        "session_gap_hours", "min_intervening_groups", "daily_gap_flag_hours",
        "adjacent_prefix_flag_share", "high_overlap_ratio", "high_overlap_flag_share",
        "daily_span_flag_share",
    }
    if not isinstance(sensitivity, Mapping) or set(sensitivity) != sensitivity_required:
        raise ValueError("context-session preflight sensitivity configuration is not frozen")
    if sorted(float(value) for value in sensitivity["session_gap_hours"]) != [2.0, 4.0, 6.0, 8.0, 12.0]:
        raise ValueError("context-session preflight sensitivity session gaps are not frozen")
    if sorted(int(value) for value in sensitivity["min_intervening_groups"]) != [0, 1]:
        raise ValueError("context-session preflight sensitivity intervening groups are not frozen")
    normalized = {
        "session_gap_hours": float(pairing["session_gap_hours"]),
        "min_intervening_groups_for_positive": int(pairing["min_intervening_groups_for_positive"]),
        "positive_pairs_per_anchor": int(pairing["positive_pairs_per_anchor"]),
        "negative_pairs_per_anchor": int(pairing["negative_pairs_per_anchor"]),
        "seed": int(pairing["seed"]),
        "positive_local_day_timezone": str(pairing["positive_local_day_timezone"]),
        "positive_same_local_day": bool(pairing["positive_same_local_day"]),
    }
    normalized["sensitivity"] = {
        "session_gap_hours": [float(value) for value in sensitivity["session_gap_hours"]],
        "min_intervening_groups": [int(value) for value in sensitivity["min_intervening_groups"]],
        "daily_gap_flag_hours": float(sensitivity["daily_gap_flag_hours"]),
        "adjacent_prefix_flag_share": float(sensitivity["adjacent_prefix_flag_share"]),
        "high_overlap_ratio": float(sensitivity["high_overlap_ratio"]),
        "high_overlap_flag_share": float(sensitivity["high_overlap_flag_share"]),
        "daily_span_flag_share": float(sensitivity["daily_span_flag_share"]),
    }
    return config, normalized


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish(output_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
    manifest_path = output_dir / "context_pair_manifest.json"
    report_path = output_dir / "context_pair_preflight.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("context pair preflight outputs are immutable")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_tmp = output_dir / f".context_pair_manifest.{os.getpid()}.tmp"
    report_tmp = output_dir / f".context_pair_preflight.{os.getpid()}.tmp"
    manifest_tmp.write_bytes(_json_bytes(manifest))
    report_tmp.write_bytes(_json_bytes(report))
    os.replace(manifest_tmp, manifest_path)
    os.replace(report_tmp, report_path)


def run_context_pair_preflight(
    run_dir: str | Path,
    experiment_dir: str | Path,
    pair_config_path: str | Path,
    embedding_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Authenticate an existing preparation and publish an immutable pair preflight."""
    pair_config, pairing = _load_pair_config(pair_config_path)
    users, train_events, auth, embedding_config, training_user_ids = _auth_metadata(
        run_dir, experiment_dir, embedding_config_path,
    )
    pairs, diagnostics = build_context_pairs(
        train_events,
        train_end=auth["preparation_authentication"]["train_cutoff"],
        min_history_events=int(embedding_config["data"]["min_history_events"]),
        **{key: value for key, value in pairing.items() if key != "sensitivity"},
    )
    sensitivity = pairing["sensitivity"]
    sensitivity_results = build_session_definition_sensitivity(
        train_events,
        train_end=auth["preparation_authentication"]["train_cutoff"],
        min_history_events=int(embedding_config["data"]["min_history_events"]),
        training_user_ids=training_user_ids,
        session_gap_hours=sensitivity["session_gap_hours"],
        min_intervening_groups=sensitivity["min_intervening_groups"],
        positive_pairs_per_anchor=pairing["positive_pairs_per_anchor"],
        seed=pairing["seed"],
        **{key: value for key, value in sensitivity.items() if key not in {"session_gap_hours", "min_intervening_groups"}},
    )
    pair_configuration = {
        "schema_version": pair_config["schema_version"],
        "pair_config_sha256": sha256_file(pair_config_path),
        **pairing,
        "same_timestamp_policy": "atomic_by_user_id_timestamp",
        "negative_rule": "same_user_different_observed_session",
        "fallback_policy": "exclude",
        "embedding_base": "two_timescale_pc_future_intended_base",
        "sensitivity": sensitivity,
    }
    manifest = {
        "schema_version": CONTEXT_PAIR_MANIFEST_SCHEMA,
        **auth,
        "pair_configuration": pair_configuration,
        "coverage": diagnostics["coverage"],
        "pairs": pairs,
        "creation_provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "geoembeddings_version": __version__,
            "python": platform.python_version(),
            "command": "geoembed context-pair-preflight",
        },
    }
    validate_context_pair_manifest(manifest)
    manifest_sha256 = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    report = {
        "schema_version": CONTEXT_PAIR_PREFLIGHT_SCHEMA,
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha256,
        **auth,
        "pair_configuration": pair_configuration,
        **diagnostics,
        "session_definition_sensitivity": sensitivity_results,
        "coverage_sufficiency": {
            "status": "manual_review",
            "manual_review_required": True,
            "reason": "preflight reports coverage and does not pre-judge model implementation sufficiency",
            "usable_positive_coverage": diagnostics["coverage"]["positive_pair_count"] > 0,
            "usable_negative_coverage": diagnostics["coverage"]["negative_pair_count"] > 0,
        },
        "gate_checks": {
            "source_hashes_match": True,
            "preparation_authenticated": True,
            "cutoffs_match": True,
            "training_user_protocol_authenticated": True,
            "truth_access_absent": True,
            "duplicates_absent": True,
            "conflicts_absent": True,
            "timestamp_policy_valid": True,
            "usable_positive_coverage": diagnostics["coverage"]["positive_pair_count"] > 0,
            "usable_negative_coverage": diagnostics["coverage"]["negative_pair_count"] > 0,
            "minimum_coverage_met": None,
        },
    }
    output = Path(output_dir).expanduser().resolve()
    _publish(output, manifest, report)
    return {
        "status": report["status"],
        "manifest": str(output / "context_pair_manifest.json"),
        "report": str(output / "context_pair_preflight.json"),
        "coverage": diagnostics["coverage"],
        "exclusions": diagnostics["exclusions"],
    }
