"""Observed-only exposure propensity estimation and clipping diagnostics (T3.6)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def fit_position_propensities(rows: Iterable[Mapping[str, Any]], training_request_ids: set[str],
                              *, smoothing: float = 1.0) -> dict[int, float]:
    """Fit P(shown|candidate_position) from training impressions only.

    Candidate position and ``is_shown`` are observable platform logging fields;
    this estimate is a diagnostic logging-policy propensity, not a probability
    of a user's latent choice or an externally identified causal propensity.
    """
    if smoothing <= 0:
        raise ValueError("propensity smoothing must be positive")
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if str(row["request_id"]) not in training_request_ids:
            continue
        position = int(row["candidate_position"])
        shown = int(row["is_shown"])
        if position < 1 or shown not in (0, 1):
            raise ValueError("invalid observable impression fields")
        counts[position][0] += shown
        counts[position][1] += 1
    if not counts:
        raise ValueError("no training impressions available for propensity fitting")
    return {position: (shown + smoothing) / (total + 2 * smoothing)
            for position, (shown, total) in sorted(counts.items())}


def clipped_inverse_propensity_weights(propensities: Sequence[float], *, minimum: float,
                                       maximum_weight: float) -> tuple[np.ndarray, dict[str, float]]:
    p = np.asarray(propensities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p <= 0) | (p > 1)):
        raise ValueError("propensities must be a non-empty finite vector in (0, 1]")
    if not 0 < minimum <= 1 or maximum_weight < 1:
        raise ValueError("invalid propensity clipping thresholds")
    raw = 1.0 / p
    weights = np.minimum(1.0 / np.maximum(p, minimum), maximum_weight)
    clipped = ~np.isclose(weights, raw)
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    diagnostics = {
        "rows": float(len(weights)), "effective_sample_size": ess,
        "effective_sample_fraction": ess / len(weights), "clipping_rate": float(clipped.mean()),
        "weight_min": float(weights.min()), "weight_median": float(np.median(weights)),
        "weight_mean": float(weights.mean()), "weight_p95": float(np.quantile(weights, .95)),
        "weight_max": float(weights.max()),
    }
    return weights, diagnostics
