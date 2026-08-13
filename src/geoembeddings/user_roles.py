"""Versioned, deterministic target-user role protocol.

This module is observed-only.  Roles are a modeling protocol, not simulator
truth and not an estimate of deployment membership prevalence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


PROTOCOL_SCHEMA = "geoembeddings-user-role-protocol/1.0"
PREPARATION_SCHEMA = "geoembeddings-preparation/2.0"
HASH_DEFINITION = "sha256-canonical-sorted-identifiers/1.0"
ROLES = ("target_train", "target_validation", "target_test")


def canonical_set(values: Iterable[str]) -> dict[str, Any]:
    users = sorted({str(value) for value in values})
    return {
        "count": len(users),
        "identity_sha256": hashlib.sha256("\n".join(users).encode()).hexdigest(),
    }


def protocol_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = config.get("data", {}).get("user_role_protocol")
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError(f"data.user_role_protocol.schema_version must be {PROTOCOL_SCHEMA}")
    fractions = value.get("fractions")
    if not isinstance(fractions, Mapping) or set(fractions) != set(ROLES):
        raise ValueError(f"user-role fractions must contain exactly {list(ROLES)}")
    numbers = [float(fractions[role]) for role in ROLES]
    if any(number <= 0 for number in numbers) or abs(sum(numbers) - 1.0) > 1e-9:
        raise ValueError("user-role fractions must be positive and sum to 1")
    if not isinstance(value.get("seed"), int):
        raise ValueError("user-role seed must be an integer")
    return value


def assign_users(users: Iterable[str], protocol: Mapping[str, Any]) -> dict[str, str]:
    """Assign canonical IDs without depending on input or CSV row order."""
    seed = int(protocol["seed"])
    fractions = protocol["fractions"]
    train_edge = float(fractions["target_train"])
    validation_edge = train_edge + float(fractions["target_validation"])
    result: dict[str, str] = {}
    for user in sorted({str(value) for value in users}):
        digest = hashlib.sha256(f"{PROTOCOL_SCHEMA}\0{seed}\0{user}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        result[user] = ("target_train" if draw < train_edge else
                        "target_validation" if draw < validation_edge else "target_test")
    return result


def role_summary(assignments: Mapping[str, str]) -> dict[str, Any]:
    if set(assignments.values()) - set(ROLES):
        raise ValueError("user-role assignment contains an unknown role")
    groups = {role: canonical_set(user for user, value in assignments.items() if value == role)
              for role in ROLES}
    groups["all_target_users"] = canonical_set(assignments)
    return groups


def assignment_hash(assignments: Mapping[str, str]) -> str:
    payload = json.dumps(sorted(assignments.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def authenticate_roles(metadata: Mapping[str, Any], config: Mapping[str, Any], users: Iterable[str]) -> dict[str, str] | None:
    """Recompute roles and reject source/config drift or post-hoc reassignment."""
    protocol = protocol_config(config)
    recorded = metadata.get("user_role_protocol")
    if protocol is None:
        if recorded is not None:
            raise ValueError("Prepared user-role protocol cannot be removed post hoc")
        return None
    if not isinstance(recorded, Mapping) or metadata.get("preparation_schema_version") != PREPARATION_SCHEMA:
        raise ValueError("User-role configuration requires preparation schema 2.0; rerun prepare")
    assignments = assign_users(users, protocol)
    expected = {
        "schema_version": PROTOCOL_SCHEMA,
        "seed": int(protocol["seed"]),
        "fractions": {role: float(protocol["fractions"][role]) for role in ROLES},
        "assignment_sha256": assignment_hash(assignments),
        "roles": role_summary(assignments),
    }
    if dict(recorded) != expected:
        raise ValueError("Prepared user roles drifted from the declared canonical assignment")
    return assignments
