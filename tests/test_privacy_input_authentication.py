from pathlib import Path

import pytest

from geoembeddings.io import write_json
from geoembeddings.privacy import PrivacyInput, authenticate_privacy_inputs


def _input(tmp_path: Path) -> PrivacyInput:
    return PrivacyInput(
        name="baseline",
        kind="statistical_baseline",
        export_path=tmp_path / "embeddings.npz",
        prepared_metadata_path=tmp_path / "prepared_metadata.json",
        utility_report_path=tmp_path / "utility.json",
        selection_role="diagnostic_control",
        parameter_count=0,
        eligible_users=(),
        utility_report_users=(),
        checkpoint_identity="not_applicable",
        model_variant="statistical_baseline",
    )


def test_privacy_authentication_rejects_evidence_before_opening_inputs(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    write_json({"schema_version": "wrong", "task_id": "T2.7"}, index)

    with pytest.raises(ValueError, match="evidence-index schema"):
        authenticate_privacy_inputs(index, [_input(tmp_path)])


def test_privacy_authentication_requires_negative_t2_7_decision(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    write_json(
        {
            "schema_version": "geoembeddings-factorization-evidence-index/1.0",
            "task_id": "T2.4-T2.7",
            "decision": "advance",
            "matched_identity": {},
        },
        index,
    )

    with pytest.raises(ValueError, match="T2.7 decision mismatch"):
        authenticate_privacy_inputs(index, [_input(tmp_path)])
