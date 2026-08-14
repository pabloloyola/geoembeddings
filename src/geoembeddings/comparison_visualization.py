"""Deterministic, no-aggregate SVG presentation of authenticated comparisons."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

from .io import read_json

SCORECARD_SCHEMA = "geoembeddings-comparison-scorecard/1.0"


def render_comparison_scorecard(
    comparison_path: str | Path,
    output_path: str | Path,
    *,
    factorized_path: str | Path | None = None,
    ranking_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Read authenticated reports and render separate R1--R9 diagnostic axes.

    Authentication here means that the input is an evaluator-produced report with
    its comparison identity/contract intact. Supplemental ranking reports must
    explicitly say that authentication passed; unauthenticated ranking JSON is
    rejected rather than silently displayed.
    """
    comparison_path = Path(comparison_path)
    report = read_json(comparison_path)
    if not isinstance(report.get("comparison_contract"), dict):
        raise ValueError("Comparison scorecard requires an authenticated comparison_contract")
    source_hashes = {"comparison": _sha256(comparison_path)}

    factorized = None
    if factorized_path is not None:
        factorized_path = Path(factorized_path)
        factorized = read_json(factorized_path)
        if (factorized.get("schema_version") != "geoembeddings-factorized-comparison/1.0"
                or not isinstance(factorized.get("matched_identity"), dict)):
            raise ValueError("T2.7 scorecard input is not an authenticated factorized comparison")
        source_hashes["factorized"] = _sha256(factorized_path)

    rankings = []
    for raw_path in ranking_paths:
        path = Path(raw_path); value = read_json(path)
        authentication = value.get("authentication", {})
        if authentication.get("status") != "passed":
            raise ValueError(f"Ranking report is not authenticated: {path}")
        rankings.append((path.stem, value))
        source_hashes[f"ranking:{path.stem}"] = _sha256(path)

    panels = _panels(report, rankings)
    collapse_warning = _collapse_warning(report)
    svg = _svg(panels, factorized, collapse_warning, source_hashes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return {
        "schema_version": SCORECARD_SCHEMA,
        "source_sha256": source_hashes,
        "output_sha256": hashlib.sha256(svg.encode()).hexdigest(),
        "panel_count": len(panels),
        "collapse_warning": collapse_warning,
        "no_aggregate_winner": True,
    }


def _panels(report: dict[str, Any], rankings: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, list[str]]]:
    geometry = report.get("stability_and_distinctiveness", {})
    persistent = report.get("persistent_information", {})
    panels = [
        ("R1 · temporal stability ↔ persistent task information", [
            _pair("train→test cosine", _dig(geometry, "baseline", "same_user_cosine", "train_to_test", "mean"), _dig(geometry, "learned", "same_user_cosine", "train_to_test", "mean"), "higher only with retained information"),
            _pair("persistent probe mean R²", _dig(persistent, "baseline", "mean_r2"), _dig(persistent, "learned", "mean_r2"), "higher"),
        ]),
        ("R1 · same-user similarity ↔ different-user separation", [
            _pair("same−different cosine", _dig(geometry, "baseline", "same_minus_different_train_test_cosine"), _dig(geometry, "learned", "same_minus_different_train_test_cosine"), "higher"),
            _pair("different-user cosine", _dig(geometry, "baseline", "different_user_cosine", "mean"), _dig(geometry, "learned", "different_user_cosine", "mean"), "lower is more separated"),
        ]),
        ("R1 · retrieval ↔ centered effective rank", [
            _pair("temporal retrieval top-1", _dig(geometry, "baseline", "temporal_user_retrieval", "train_query_test_gallery_top1"), _dig(geometry, "learned", "temporal_user_retrieval", "train_query_test_gallery_top1"), "higher"),
            _pair("centered effective-rank ratio", _dig(geometry, "baseline", "test_geometry", "effective_rank_ratio"), _dig(geometry, "learned", "test_geometry", "effective_rank_ratio"), "higher; diagnostic"),
        ]),
        ("R3–R4 · episode response ↔ post-episode recovery", _episode_lines(report)),
        ("R5–R8 · robustness and transfer slices", _robustness_transfer_lines(report)),
        ("R9 · ranking metrics and control deltas", _ranking_lines(rankings)),
    ]
    requirements = report.get("requirements", {})
    panels.append(("R1–R9 · requirement availability", [
        f"{key.split('_', 1)[0]}: {value.get('status', 'unavailable')} · coverage: {value.get('coverage', 'not reported')} · confidence: {value.get('confidence', 'not reported')}"
        for key, value in requirements.items() if key.startswith(tuple(f"R{i}_" for i in range(1, 10)))
    ] or ["unavailable · coverage: 0 · confidence: not reported"]))
    return panels


def _episode_lines(report: dict[str, Any]) -> list[str]:
    values = report.get("episode_response_comparison")
    if not values:
        return ["episode response: unavailable · coverage: 0", "post-episode recovery: unavailable · coverage: 0"]
    return [_comparison_line(name.replace("_", " "), metric, "diagnostic") for name, metric in sorted(values.items())]


def _robustness_transfer_lines(report: dict[str, Any]) -> list[str]:
    lines = []
    robust = report.get("R6_R7_robustness_comparison")
    if robust:
        for axis in ("R6_views", "R7_views"):
            for row in robust.get(axis, []):
                lines.append(_comparison_line(f"{axis[:2]} {row.get('view_id', 'slice')}", row.get("cosine_drift_mean", {}), "lower drift") + f" · coverage: {_fmt(row.get('coverage', {}).get('learned'))}")
    transfer = report.get("R2_R8_spatial_transfer_comparison")
    if transfer:
        lines.append(_comparison_line("distance retrieval", transfer.get("distance_retrieval", {}), "lower distance"))
        lines.append(_comparison_line("geohash boundary", transfer.get("geohash_boundary_pairs", {}), "diagnostic"))
        lines.append(f"transfer coverage: {_fmt(transfer.get('coverage'))}")
    return lines or ["robustness: unavailable · coverage: 0", "transfer: unavailable · coverage: 0"]


def _ranking_lines(rankings: list[tuple[str, dict[str, Any]]]) -> list[str]:
    if not rankings:
        return ["authenticated ranking reports: unavailable · coverage: 0"]
    lines = []
    for name, report in rankings:
        coverage = report.get("coverage", report.get("eligible_request_coverage", "not reported"))
        metrics = report.get("metrics", {})
        lines.append(f"{name} · coverage: {_fmt(coverage)} · confidence: {_fmt(report.get('confidence', 'not reported'))}")
        for metric, value in sorted(metrics.items()):
            lines.append(f"  {metric}: {_fmt(value)} · direction: {report.get('metric_directions', {}).get(metric, 'reported axis')}")
        for control, deltas in sorted(report.get("baseline_comparisons", {}).items()):
            lines.append(f"  diagnostic control {control}: {_fmt(deltas)}")
    return lines


def _collapse_warning(report: dict[str, Any]) -> str | None:
    geometry = report.get("stability_and_distinctiveness", {}).get("learned", {})
    stability = _dig(geometry, "same_user_cosine", "train_to_test", "mean")
    weak = [
        _dig(report, "persistent_information", "learned", "mean_r2"),
        geometry.get("same_minus_different_train_test_cosine"),
        _dig(geometry, "temporal_user_retrieval", "train_query_test_gallery_top1"),
        _dig(geometry, "test_geometry", "effective_rank_ratio"),
    ]
    if isinstance(stability, (int, float)) and stability >= .9 and any(isinstance(v, (int, float)) and v < .1 for v in weak):
        return "COLLAPSE WARNING: high stability coexists with weak task information, separation, retrieval, or effective rank."
    return None


def _svg(panels: list[tuple[str, list[str]]], factorized: dict[str, Any] | None,
         warning: str | None, hashes: dict[str, str]) -> str:
    gate = "T2.7 gate: unavailable"
    if factorized is not None:
        gate = f"T2.7 FAILED GATE · {factorized.get('decision', 'do not advance').upper()}"
    content_lines = sum(2 + len(lines) for _, lines in panels)
    height = max(640, 150 + (58 if warning else 0) + content_lines * 22)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
           f'<metadata>{html.escape(json.dumps({"schema_version": SCORECARD_SCHEMA, "source_sha256": hashes, "deterministic": True, "aggregate": None}, sort_keys=True, separators=(",", ":")))}</metadata>',
           '<rect width="1200" height="100%" fill="#f7f7f5"/>',
           '<style>text{font-family:system-ui,sans-serif;fill:#202124}.title{font-size:26px;font-weight:700}.gate{font-size:20px;font-weight:700;fill:#8b1e1e}.head{font-size:17px;font-weight:650}.body{font-size:14px}.warn{font-size:15px;font-weight:650;fill:#6d4c00}</style>',
           '<text x="40" y="42" class="title">R1–R9 comparison scorecard · no aggregate winner</text>',
           f'<text x="40" y="76" class="gate">{html.escape(gate)}</text>',
           '<text x="40" y="100" class="body">Representations and named branches are diagnostic controls; branch names do not establish semantic recovery.</text>']
    y = 128
    if warning:
        out.append(f'<rect x="30" y="{y-18}" width="1140" height="42" rx="6" fill="#fff3cd"/>')
        out.append(f'<text x="40" y="{y+7}" class="warn">{html.escape(warning)}</text>'); y += 58
    for title, lines in panels:
        box_height = 48 + 22 * len(lines)
        out.append(f'<rect x="30" y="{y-20}" width="1140" height="{box_height}" rx="6" fill="#fff" stroke="#d5d7da"/>')
        out.append(f'<text x="40" y="{y+5}" class="head">{html.escape(title)}</text>'); y += 32
        for line in lines:
            out.append(f'<text x="52" y="{y}" class="body">{html.escape(line)}</text>'); y += 22
        y += 22
    out.append('</svg>\n')
    return "\n".join(out)


def _pair(label: str, baseline: Any, learned: Any, direction: str) -> str:
    delta = learned - baseline if isinstance(baseline, (int, float)) and isinstance(learned, (int, float)) else None
    return f"{label}: diagnostic control A {_fmt(baseline)} · diagnostic control B {_fmt(learned)} · Δ {_fmt(delta)} · direction: {direction}"


def _comparison_line(label: str, value: dict[str, Any], direction: str) -> str:
    return f"{label}: A {_fmt(value.get('baseline'))} · B {_fmt(value.get('learned'))} · Δ {_fmt(value.get('learned_minus_baseline'))} · direction: {direction}"


def _dig(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict): return None
        value = value.get(key)
    return value


def _fmt(value: Any) -> str:
    if value is None: return "unavailable"
    if isinstance(value, float): return f"{value:+.4f}" if value < 0 else f"{value:.4f}"
    if isinstance(value, (dict, list)): return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
