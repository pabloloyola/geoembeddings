from pathlib import Path

import pandas as pd
import pytest

from geoembeddings.privacy import (
    AuthenticatedPrivacyInputs,
    PrivacyInputIdentity,
    load_protected_labels,
)


def _authenticated() -> AuthenticatedPrivacyInputs:
    identity = PrivacyInputIdentity(
        "geoembeddings-privacy-input/1.0", "control", "learned", "diagnostic_control",
        "export", 1, "0" * 64, "export", "single_vector", ("combined",), (1,),
        "1" * 64, ("test_end",), "2" * 64, "3" * 64, "4" * 64, "2.0", (), (),
        (), 1, "5" * 64, "6" * 64,
    )
    return AuthenticatedPrivacyInputs(
        "7" * 64, "geoembeddings-factorization-evidence-index/1.0", "T2.7",
        "do not advance", (identity,),
    )


def _truth(tmp_path: Path) -> Path:
    path = tmp_path / "truth" / "user_latents.csv.gz"
    path.parent.mkdir()
    rows = []
    for index in range(90):
        rows.append({
            "user_id": f"u{index}",
            # Each frozen user split covers all train-fitted bins.
            "price_sensitivity": float(index % 30),
            "family_orientation": float(index % 17),
            "home_latitude": 35.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_continuous_bins_fit_train_only_and_emit_aggregate_metadata(tmp_path: Path) -> None:
    path = _truth(tmp_path)
    splits = {f"u{i}": ("train" if i < 30 else "validation" if i < 60 else "test") for i in range(90)}
    result = load_protected_labels(
        _authenticated(), path, split_by_user=splits,
        attributes=("price_sensitivity_group",), minimum_total=3,
        minimum_per_class=1, minimum_cell_support=1,
    )
    summary = result.summaries[0]
    assert summary.status == "available"
    assert summary.fit_split == "train"
    assert summary.bin_boundaries == pytest.approx((29 / 3, 58 / 3))
    assert "u0" not in repr(result)
    assert not hasattr(result, "labels")


def test_missing_labels_are_not_imputed_and_small_cells_are_excluded(tmp_path: Path) -> None:
    path = _truth(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[0, "price_sensitivity"] = None
    frame.to_csv(path, index=False)
    splits = {f"u{i}": ("train" if i < 30 else "validation" if i < 60 else "test") for i in range(90)}
    result = load_protected_labels(
        _authenticated(), path, split_by_user=splits,
        attributes=("price_sensitivity_group",), minimum_total=3,
        minimum_per_class=1, minimum_cell_support=20,
    )
    summary = result.summaries[0]
    assert summary.status == "excluded"
    assert summary.reason == "minimum_class_by_split_cell_support_not_met"
    assert summary.missing_count == 1
    assert summary.counts == ()


def test_public_and_prohibited_attributes_fail_closed(tmp_path: Path) -> None:
    path = _truth(tmp_path)
    public = load_protected_labels(
        _authenticated(), path, split_by_user={}, attributes=("age_group",),
    ).summaries[0]
    assert public.status == "excluded"
    assert public.reason == "public_or_observed_not_a_protected_leakage_target"
    assert public.unsupported_count == 90
    with pytest.raises(ValueError, match="prohibited"):
        load_protected_labels(
            _authenticated(), path, split_by_user={}, attributes=("true_utility",),
        )


def test_authentication_is_checked_before_truth_path_is_opened(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="authentication"):
        load_protected_labels(None, tmp_path / "truth" / "user_latents.csv.gz", split_by_user={})  # type: ignore[arg-type]
