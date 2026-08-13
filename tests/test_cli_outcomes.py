import json

import pytest

from geoembeddings import cli


def _invoke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], outcome):
    monkeypatch.setattr(cli, "_run_command", outcome)
    cli.main(["inspect-evidence"])
    return capsys.readouterr()


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (ValueError("protected/schema.json contained secret"), 2, "schema or identity authentication failed"),
        (FileExistsError("/protected/existing.json"), 3, "output already exists and is immutable"),
        (FileNotFoundError("/protected/source.json"), 4, "required source artifact is missing"),
        (RuntimeError("token at /protected/truth.csv"), 1, "unexpected internal error"),
    ],
)
def test_cli_failure_categories_are_actionable_and_path_safe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    code: int,
    message: str,
) -> None:
    def fail(_args):
        raise error

    monkeypatch.setattr(cli, "_run_command", fail)
    with pytest.raises(SystemExit) as raised:
        cli.main(["inspect-evidence"])
    captured = capsys.readouterr()
    assert raised.value.code == code
    assert message in captured.err
    assert "/protected" not in captured.err
    assert captured.out == ""


def test_scientifically_unavailable_metric_is_a_successful_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(_args):
        raise cli.ScientificMetricUnavailable("membership_inference", "insufficient_common_support")

    captured = _invoke(monkeypatch, capsys, unavailable)
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "unavailable",
        "section": "membership_inference",
        "reason": "insufficient_common_support",
    }


def test_successful_report_preserves_explicitly_unavailable_sections(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = {
        "status": "passed",
        "metrics": {"retrieval": {"status": "available", "value": 0.5}},
        "selection_dependent": {"status": "unavailable", "reason": "no_selected_candidate"},
    }
    captured = _invoke(monkeypatch, capsys, lambda _args: report)
    assert captured.err == ""
    assert json.loads(captured.out) == report
