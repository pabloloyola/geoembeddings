from pathlib import Path

import pytest

from geoembeddings.factorization_comparison import REQUIRED_VARIANTS, _require_equal


def test_factorization_matrix_has_all_decision_controls():
    assert REQUIRED_VARIANTS == (
        "factorized_pc", "capacity_matched_single", "persistent_only", "context_only",
        "factorized_no_persistent_loss", "factorized_no_context_loss",
    )


def test_factorization_comparison_rejects_any_identity_mismatch():
    with pytest.raises(ValueError, match="user mask mismatch"):
        _require_equal("user mask", {"factorized_pc": "a", "context_only": "b"})
