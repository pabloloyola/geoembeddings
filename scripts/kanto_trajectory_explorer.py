#!/usr/bin/env python3
"""Explore observed Kanto trajectories without opening simulator truth.

The command-line interface is headless-safe.  ``notebook_explorer`` provides an
optional ipywidgets surface when this module is imported from a notebook.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from geoembeddings.contract import OBSERVED_FILES

FILTER_COLUMNS = {
    "age_group": "age_group",
    "household_type": "household_type",
    "user_id": "user_id",
    "service": "service_id",
    "action_type": "action_type",
    "observation_mode": "observation_mode",
    "region": "region_id",
}
LOCAL_TIMEZONE = "Asia/Tokyo"


class EmptySelectionError(ValueError):
    """Raised when valid filters select no observed events."""


@dataclass(frozen=True)
class ExplorerData:
    users: pd.DataFrame
    events: pd.DataFrame


def load_observed(run_dir: Path | str) -> ExplorerData:
    """Load only public demographics and observed event coordinates."""
    observed = Path(run_dir) / "observed"
    users_path = observed / OBSERVED_FILES["users"]
    events_path = observed / OBSERVED_FILES["events"]
    for path in (users_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing observed table: {path}")
    users = pd.read_csv(users_path, dtype={"user_id": "string"})
    events = pd.read_csv(events_path, dtype={"user_id": "string"})
    required = {"user_id", "timestamp", "latitude", "longitude"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Observed events are missing columns: {sorted(missing)}")
    if users["user_id"].duplicated().any():
        raise ValueError("Observed users must contain one row per user")
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="raise").dt.tz_convert(LOCAL_TIMEZONE)
    if not np.isfinite(events[["latitude", "longitude"]].to_numpy(dtype=float)).all():
        raise ValueError("Observed event coordinates must be finite")
    # Preserve a deterministic tie break independent of input row order.
    tie_columns = [c for c in ("service_id", "action_type", "observation_mode", "region_id", "object_id") if c in events]
    events = events.sort_values(["user_id", "timestamp", *tie_columns], kind="stable").reset_index(drop=True)
    return ExplorerData(users=users, events=events)


def available_filter_values(data: ExplorerData) -> dict[str, list[str]]:
    """Return filter choices from columns actually present in the run."""
    values: dict[str, list[str]] = {}
    for name, column in FILTER_COLUMNS.items():
        frame = data.users if name in {"age_group", "household_type"} else data.events
        if column in frame:
            values[name] = sorted(frame[column].dropna().astype(str).unique().tolist())
    return values


def _normalize_values(values: str | Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [item.strip() for value in values for item in str(value).split(",") if item.strip()]


def filter_events(
    data: ExplorerData,
    filters: Mapping[str, str | Iterable[str] | None] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compose validated demographic, event, and inclusive local-date filters."""
    filters = filters or {}
    unknown_names = set(filters) - set(FILTER_COLUMNS)
    if unknown_names:
        raise ValueError(f"Unknown filters: {sorted(unknown_names)}")
    available = available_filter_values(data)
    selected_users = data.users.copy()
    for name in ("age_group", "household_type"):
        requested = _normalize_values(filters.get(name))
        if not requested:
            continue
        if name not in available:
            raise ValueError(f"Filter {name!r} is not present in this dataset")
        invalid = sorted(set(requested) - set(available[name]))
        if invalid:
            raise ValueError(f"Invalid {name} value(s) {invalid}; available values: {available[name]}")
        selected_users = selected_users[selected_users[FILTER_COLUMNS[name]].astype(str).isin(requested)]

    result = data.events[data.events["user_id"].isin(selected_users["user_id"])].copy()
    for name in ("user_id", "service", "action_type", "observation_mode", "region"):
        requested = _normalize_values(filters.get(name))
        if not requested:
            continue
        if name not in available:
            raise ValueError(f"Filter {name!r} is not present in this dataset")
        invalid = sorted(set(requested) - set(available[name]))
        if invalid:
            raise ValueError(f"Invalid {name} value(s) {invalid}; available values: {available[name]}")
        result = result[result[FILTER_COLUMNS[name]].astype(str).isin(requested)]

    local_dates = result["timestamp"].dt.date
    if start_date:
        start = pd.Timestamp(start_date).date()
        result = result[local_dates >= start]
        local_dates = result["timestamp"].dt.date
    if end_date:
        end = pd.Timestamp(end_date).date()
        result = result[local_dates <= end]
    if start_date and end_date and pd.Timestamp(start_date).date() > pd.Timestamp(end_date).date():
        raise ValueError("start_date must be on or before end_date")
    return result.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def selection_summary(events: pd.DataFrame) -> dict[str, int]:
    return {
        "selected_users": int(events["user_id"].nunique()),
        "selected_events": int(len(events)),
        "selected_local_dates": int(events["timestamp"].dt.date.nunique()),
    }


def split_trajectories(events: pd.DataFrame, max_gap: pd.Timedelta | str = "6h") -> list[pd.DataFrame]:
    """Split by user and large time gaps, returning stable chronological parts."""
    gap = pd.Timedelta(max_gap)
    if gap <= pd.Timedelta(0):
        raise ValueError("max_gap must be positive")
    parts: list[pd.DataFrame] = []
    ordered = events.sort_values(["user_id", "timestamp"], kind="stable")
    for _, user_events in ordered.groupby("user_id", sort=True):
        group_ids = user_events["timestamp"].diff().gt(gap).cumsum()
        parts.extend(part.reset_index(drop=True) for _, part in user_events.groupby(group_ids, sort=True))
    return parts


def render_trajectories(
    events: pd.DataFrame,
    output: Path | str,
    *,
    color_by: str = "user",
    max_gap: pd.Timedelta | str = "6h",
    max_trajectories: int = 100,
):
    """Render capped, direction-marked trajectories and return figure metadata."""
    if events.empty:
        raise EmptySelectionError("No observed events match the requested filters")
    if color_by not in {"user", "service", "time"}:
        raise ValueError("color_by must be one of: user, service, time")
    if max_trajectories < 1:
        raise ValueError("max_trajectories must be at least 1")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_parts = split_trajectories(events, max_gap)
    parts = all_parts[:max_trajectories]
    fig, ax = plt.subplots(figsize=(11, 8))
    categorical = "user_id" if color_by == "user" else "service_id"
    categories = sorted(events[categorical].astype(str).unique()) if color_by != "time" else []
    palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
    time_min = events["timestamp"].min()
    time_span = max((events["timestamp"].max() - time_min).total_seconds(), 1.0)

    for part in parts:
        xy = part[["longitude", "latitude"]].to_numpy()
        if color_by == "time":
            colors = [plt.cm.viridis((value - time_min).total_seconds() / time_span) for value in part["timestamp"]]
            color = colors[0]
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=20, zorder=3)
        else:
            colors = [palette[str(value)] for value in part[categorical]]
            color = colors[0]
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=20, zorder=3)
        if len(part) > 1:
            # Arrowheads on every segment provide chronological direction cues.
            for index, (start, end) in enumerate(zip(xy[:-1], xy[1:])):
                segment_color = colors[index]
                ax.plot([start[0], end[0]], [start[1], end[1]], color=segment_color, alpha=.55, linewidth=1.2)
                ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": segment_color, "alpha": .55, "lw": 1})
        ax.scatter(*xy[0], marker="o", facecolors="none", edgecolors="black", s=65, zorder=4)
        ax.scatter(*xy[-1], marker="X", color="black", s=55, zorder=4)
    if color_by == "time":
        from matplotlib.colors import Normalize

        scalar = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(0, time_span))
        colorbar = fig.colorbar(scalar, ax=ax)
        colorbar.set_label(f"Seconds since {time_min.isoformat()}")
    else:
        from matplotlib.lines import Line2D

        handles = [Line2D([0], [0], marker="o", color=color, label=label, linewidth=1) for label, color in palette.items()]
        ax.legend(handles=handles, title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set(title=f"Observed trajectories ({len(parts)} of {len(all_parts)} segments)", xlabel="Longitude", ylabel="Latitude")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    return fig, {"rendered_trajectories": len(parts), "available_trajectories": len(all_parts)}


def notebook_explorer(run_dir: Path | str, output_dir: Path | str | None = None):
    """Display dataset-derived ipywidgets controls in a Jupyter notebook."""
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise RuntimeError("Install the viz extra to use notebook controls") from exc
    data = load_observed(run_dir)
    available = available_filter_values(data)
    controls = {name: widgets.SelectMultiple(options=value, description=name) for name, value in available.items()}
    controls["start_date"] = widgets.DatePicker(description="start date")
    controls["end_date"] = widgets.DatePicker(description="end date")
    controls["color_by"] = widgets.Dropdown(options=["user", "service", "time"], description="color")
    output_widget = widgets.Output()

    def update(**values):
        with output_widget:
            clear_output(wait=True)
            chosen = {name: values[name] for name in FILTER_COLUMNS if name in values}
            selected = filter_events(data, chosen, str(values["start_date"]) if values["start_date"] else None, str(values["end_date"]) if values["end_date"] else None)
            print(json.dumps(selection_summary(selected), sort_keys=True))
            if selected.empty:
                print("No observed events match the requested filters")
                return
            path = Path(output_dir or Path(run_dir) / "exploration_artifacts") / "observed_trajectories.png"
            figure, _ = render_trajectories(selected, path, color_by=values["color_by"])
            display(figure)

    interactive = widgets.interactive_output(update, controls)
    display(widgets.VBox([*controls.values(), output_widget, interactive]))
    return controls


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    for name in FILTER_COLUMNS:
        parser.add_argument(f"--{name.replace('_', '-')}", action="append", help="Repeat or use comma-separated values")
    parser.add_argument("--start-date", help="Inclusive Asia/Tokyo calendar date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Inclusive Asia/Tokyo calendar date (YYYY-MM-DD)")
    parser.add_argument("--color-by", choices=("user", "service", "time"), default="user")
    parser.add_argument("--max-gap", default="6h", help="Pandas duration separating trajectories")
    parser.add_argument("--max-trajectories", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/exploration_artifacts/observed_trajectories.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = load_observed(args.run_dir)
        filters = {name: getattr(args, name) for name in FILTER_COLUMNS if getattr(args, name)}
        events = filter_events(data, filters, args.start_date, args.end_date)
        print(json.dumps(selection_summary(events), sort_keys=True), flush=True)
        output = args.output or args.run_dir / "exploration_artifacts" / "observed_trajectories.png"
        _, rendered = render_trajectories(events, output, color_by=args.color_by, max_gap=args.max_gap, max_trajectories=args.max_trajectories)
        print(json.dumps({"output": str(output), **rendered}, sort_keys=True))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
