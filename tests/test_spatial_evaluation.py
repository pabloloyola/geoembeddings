from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoembeddings.spatial_evaluation import (
    fit_spatial_contract,
    haversine_km,
    validate_train_only_geography,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame({"user_id": ["a", "a", "b"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"], utc=True),
        "latitude": [35.0, 35.01, 35.1], "longitude": [139.0, 139.0, 139.1],
        "region_id": ["seen", "seen", "seen"], "geohash_5": ["abcde", "abcdf", "abcde"],
        "geohash_7": ["abcde00", "abcdf00", "abcde00"]})


def test_haversine_distance_boundary_cases() -> None:
    assert haversine_km(35.0, 139.0, 35.0, 139.0) == pytest.approx(0.0)
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, rel=1e-4)
    assert haversine_km(0.0, 179.9, 0.0, -179.9) == pytest.approx(22.239, rel=1e-3)
    with pytest.raises(ValueError, match="finite"):
        haversine_km(np.nan, 0, 0, 0)


def test_spatial_fit_uses_only_supplied_training_rows_and_unknowns_remain_unknown() -> None:
    settings = {"geohash_fields": ["geohash_5", "geohash_7"], "distance_relevance_quantile": 0.5}
    fitted = fit_spatial_contract(_events(), settings)
    assert fitted["known_labels"]["geohash_5"] == ["abcde", "abcdf"]
    test_only = pd.concat([_events(), pd.DataFrame({"user_id": ["z"], "timestamp": pd.to_datetime(["2027-01-01"], utc=True),
        "latitude": [36.0], "longitude": [140.0], "region_id": ["held"],
        "geohash_5": ["zzzzz"], "geohash_7": ["zzzzz00"]})], ignore_index=True)
    assert "zzzzz" not in fitted["known_labels"]["geohash_5"]
    assert test_only.iloc[-1].geohash_5 not in fitted["known_labels"]["geohash_5"]


def test_empty_spatial_training_slice_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_spatial_contract(_events().iloc[:0], {"geohash_fields": ["geohash_5"],
                                                  "distance_relevance_quantile": 0.5})


def test_invalid_train_fitted_threshold_quantile_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_spatial_contract(_events(), {"geohash_fields": ["geohash_5"],
                                         "distance_relevance_quantile": 1.1})


def test_cross_stage_geographic_split_leakage_is_rejected() -> None:
    fitted = fit_spatial_contract(_events(), {"geohash_fields": ["geohash_5", "geohash_7"],
                                               "distance_relevance_quantile": 0.5})
    leaked_preparation_vocabularies = {
        "geohash_5": {"<PAD>": 0, "<UNK>": 1, "abcde": 2, "abcdf": 3, "test_only": 4},
        "geohash_7": {"<PAD>": 0, "<UNK>": 1, "abcde00": 2, "abcdf00": 3},
    }
    with pytest.raises(ValueError, match="Geographic split leakage"):
        validate_train_only_geography(leaked_preparation_vocabularies, fitted)
