from __future__ import annotations

import numpy as np

from scripts.recoverability_gate import _cross_validated_ridge


def test_recoverability_ridge_is_stable_with_rare_columns() -> None:
    rng = np.random.default_rng(20260803)
    users = [f"user_{index:04d}" for index in range(200)]
    signal = rng.normal(size=len(users))
    common_noise = rng.normal(size=(len(users), 8))
    rare = np.zeros((len(users), 80), dtype=float)
    rare[np.arange(80), np.arange(80)] = 1.0
    x = np.column_stack([signal, common_noise, rare])
    y = 0.8 * signal + rng.normal(scale=0.15, size=len(users))

    result = _cross_validated_ridge(
        x,
        y,
        users,
        folds=5,
        alphas=[1.0, 10.0, 100.0],
    )

    assert result["status"] == "ok"
    assert result["r2"] > 0.8
    assert result["alpha"] in {1.0, 10.0, 100.0}
    assert len(result["alpha_candidates"]) == 3
