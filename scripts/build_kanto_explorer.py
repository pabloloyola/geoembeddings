#!/usr/bin/env python3
"""Build an inline interactive explorer for a Kanto simulator dataset.

The output is an HTML fragment for ChatGPT Work's visualization surface. It
contains a representative subset of users and keeps all data inline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compact_event(row: dict[str, str]) -> dict[str, Any]:
    return {
        "t": row["timestamp"],
        "s": row["service_id"],
        "a": row["action_type"],
        "m": row["observation_mode"],
        "r": row["region_id"],
        "c": row["object_category"],
    }


def compact_trajectory(row: dict[str, str]) -> dict[str, Any]:
    return {
        "t": row["timestamp"],
        "a": row["activity"],
        "r": row["true_region_id"],
    }


def compact_episode(row: dict[str, str]) -> dict[str, Any]:
    return {
        "d": row["start_time"][:10],
        "p": row["primary_intent"],
        "s": row["secondary_intent"],
        "o": row["origin_region_id"],
        "x": row["destination_region_id"],
    }


def choose_users(
    users: list[dict[str, str]],
    events: list[dict[str, str]],
    episodes: list[dict[str, str]],
    limit: int,
) -> list[str]:
    event_counts = Counter(row["user_id"] for row in events)
    services: defaultdict[str, set[str]] = defaultdict(set)
    for row in events:
        services[row["user_id"]].add(row["service_id"])
    travel_users = []
    for row in episodes:
        if row["primary_intent"] == "travel" and row["user_id"] not in travel_users:
            travel_users.append(row["user_id"])
    zero_users = [row["user_id"] for row in users if event_counts[row["user_id"]] == 0]
    diverse = sorted(
        (row["user_id"] for row in users),
        key=lambda user_id: (len(services[user_id]), event_counts[user_id]),
        reverse=True,
    )
    selected: list[str] = []
    for group in (travel_users[:12], zero_users[:8], diverse):
        for user_id in group:
            if user_id not in selected:
                selected.append(user_id)
            if len(selected) >= limit:
                return selected
    return selected


def build_payload(dataset: Path, report_path: Path, sample_users: int) -> dict[str, Any]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    users = read_rows(dataset / "observed/users_observed.csv.gz")
    events = read_rows(dataset / "observed/observed_events.csv.gz")
    episodes = read_rows(dataset / "truth/episodes_truth.csv.gz")
    trajectories = read_rows(dataset / "truth/trajectories_truth.csv.gz")
    selected = set(choose_users(users, events, episodes, sample_users))
    user_meta = {row["user_id"]: row for row in users if row["user_id"] in selected}
    grouped: dict[str, dict[str, Any]] = {
        user_id: {"meta": user_meta[user_id], "events": [], "trajectories": [], "episodes": []}
        for user_id in sorted(selected)
    }
    for row in events:
        if row["user_id"] in selected:
            grouped[row["user_id"]]["events"].append(compact_event(row))
    for row in trajectories:
        if row["user_id"] in selected:
            grouped[row["user_id"]]["trajectories"].append(compact_trajectory(row))
    for row in episodes:
        if row["user_id"] in selected:
            grouped[row["user_id"]]["episodes"].append(compact_episode(row))
    return {
        "manifest": {
            "scenario": manifest["scenario"],
            "users": manifest["users"],
            "days": manifest["days"],
            "start_date": manifest["start_date"],
        },
        "summary": report["summary"],
        "checks": report["checks"],
        "distributions": report["distributions"],
        "users": grouped,
    }


FRAGMENT = r'''<div id="kanto-simulator-explorer">
  <div class="viz-grid kse-stats" aria-label="Dataset summary">
    <div class="card viz-stat">
      <div class="text-muted">Observed events</div>
      <div class="viz-stat-value" id="kse-events"></div>
      <div class="text-small text-muted" id="kse-world"></div>
    </div>
    <div class="card viz-stat">
      <div class="text-muted">Validation</div>
      <div class="viz-stat-value" id="kse-validation"></div>
      <div class="text-small text-muted">structural + behavioral checks</div>
    </div>
    <div class="card viz-stat">
      <div class="text-muted">Passive trace match</div>
      <div class="viz-stat-value" id="kse-match"></div>
      <div class="text-small text-muted" id="kse-error"></div>
    </div>
  </div>

  <div class="viz-controls kse-controls" aria-label="Timeline controls">
    <label class="form-label" for="kse-user">Representative user
      <select class="form-select" id="kse-user"></select>
    </label>
    <label class="form-label" for="kse-day">Day
      <select class="form-select" id="kse-day"></select>
    </label>
  </div>

  <div class="kse-intent viz-row" aria-live="polite">
    <span class="viz-badge" id="kse-intent"></span>
    <span class="text-small text-muted" id="kse-context"></span>
  </div>

  <div class="kse-chart-wrap">
    <svg id="kse-timeline" class="kse-chart" role="img" aria-labelledby="kse-svg-title kse-svg-desc"></svg>
  </div>
  <div class="text-small text-muted kse-detail" id="kse-detail" aria-live="polite"></div>

  <div class="table-responsive kse-table-wrap">
    <table class="table table-sm" aria-label="Validation checks">
      <thead><tr><th>Validation check</th><th>Observed result</th><th class="text-center">Status</th></tr></thead>
      <tbody id="kse-checks"></tbody>
    </table>
  </div>
</div>

<style>
  #kanto-simulator-explorer { width: 100%; color: var(--foreground); }
  #kanto-simulator-explorer .kse-stats { margin-bottom: 16px; }
  #kanto-simulator-explorer .kse-controls { margin: 4px 0 10px; }
  #kanto-simulator-explorer .kse-intent { margin: 4px 0 8px; min-height: 24px; }
  #kanto-simulator-explorer .kse-chart-wrap { width: 100%; }
  #kanto-simulator-explorer .kse-chart { display: block; width: 100%; height: 270px; color: var(--foreground); }
  #kanto-simulator-explorer .kse-detail { margin: 6px 0 16px; min-height: 18px; }
  #kanto-simulator-explorer .kse-table-wrap { margin-top: 8px; }
  #kanto-simulator-explorer .kse-chart .axis,
  #kanto-simulator-explorer .kse-chart .grid { stroke: var(--border); stroke-width: 1; }
  #kanto-simulator-explorer .kse-chart .label { fill: var(--muted-foreground); font-size: 11px; font-weight: 400; }
  #kanto-simulator-explorer .kse-chart .lane-label { fill: var(--foreground); font-size: 11px; font-weight: 500; }
  #kanto-simulator-explorer .kse-chart .truth-line { fill: none; stroke: var(--viz-series-1); stroke-width: 2; }
  #kanto-simulator-explorer .kse-chart .truth-mark { fill: var(--background); stroke: var(--viz-series-1); stroke-width: 2; }
  #kanto-simulator-explorer .kse-chart .passive-mark { fill: var(--viz-series-2); stroke: var(--background); stroke-width: 1; }
  #kanto-simulator-explorer .kse-chart .local-mark { fill: var(--viz-series-3); stroke: var(--background); stroke-width: 1; }
  #kanto-simulator-explorer .kse-chart .ecommerce-mark { fill: var(--viz-series-4); stroke: var(--background); stroke-width: 1; }
  #kanto-simulator-explorer .kse-chart .travel-mark { fill: var(--viz-series-5); stroke: var(--background); stroke-width: 1; }
  #kanto-simulator-explorer .kse-chart .empty-label { fill: var(--muted-foreground); font-size: 12px; }
  #kanto-simulator-explorer .kse-status { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  #kanto-simulator-explorer .kse-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; background: var(--green); }
  #kanto-simulator-explorer .kse-dot-fail { background: var(--red); }
  @media (max-width: 520px) {
    #kanto-simulator-explorer .kse-chart { height: 300px; }
  }
</style>

<script>
(() => {
  const payload = __PAYLOAD__;
  const root = document.getElementById("kanto-simulator-explorer");
  const userSelect = root.querySelector("#kse-user");
  const daySelect = root.querySelector("#kse-day");
  const svg = root.querySelector("#kse-timeline");
  const users = payload.users;
  const ns = "http://www.w3.org/2000/svg";

  root.querySelector("#kse-events").textContent = payload.summary.observed_events.toLocaleString();
  root.querySelector("#kse-world").textContent = payload.manifest.users + " users · " + payload.manifest.days + " days · " + payload.manifest.scenario;
  root.querySelector("#kse-validation").textContent = payload.summary.checks_passed + "/" + payload.summary.checks_total;
  root.querySelector("#kse-match").textContent = (payload.summary.passive_trace_match_rate * 100).toFixed(1) + "%";
  root.querySelector("#kse-error").textContent = payload.summary.median_gps_error_m.toFixed(1) + " m median displacement";

  Object.keys(users).forEach((userId) => {
    const option = document.createElement("option");
    const meta = users[userId].meta;
    const services = new Set(users[userId].events.map((event) => event.s)).size;
    option.value = userId;
    option.textContent = userId + " · " + meta.home_region_id + " · " + services + " services";
    userSelect.appendChild(option);
  });

  function formatValue(value) {
    if (typeof value === "string" || typeof value === "number") return String(value);
    return Object.entries(value).map(([key, val]) => key.replaceAll("_", " ") + ": " + val).join(" · ");
  }

  const checksBody = root.querySelector("#kse-checks");
  payload.checks.forEach((item) => {
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    const valueCell = document.createElement("td");
    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    const dot = document.createElement("span");
    nameCell.textContent = item.name;
    valueCell.className = "text-small";
    valueCell.textContent = formatValue(item.value);
    statusCell.className = "text-center";
    status.className = "kse-status";
    dot.className = item.passed ? "kse-dot" : "kse-dot kse-dot-fail";
    dot.setAttribute("aria-hidden", "true");
    status.append(dot, item.passed ? "Pass" : "Review");
    statusCell.appendChild(status);
    row.append(nameCell, valueCell, statusCell);
    checksBody.appendChild(row);
  });

  function datesForUser(userId) {
    return users[userId].episodes.map((episode) => episode.d);
  }

  function refreshDays() {
    const previous = daySelect.value;
    daySelect.replaceChildren();
    datesForUser(userSelect.value).forEach((date) => {
      const option = document.createElement("option");
      option.value = date;
      option.textContent = date;
      daySelect.appendChild(option);
    });
    if ([...daySelect.options].some((option) => option.value === previous)) daySelect.value = previous;
    draw();
  }

  function add(tag, attrs, parent = svg) {
    const element = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    parent.appendChild(element);
    return element;
  }

  function hour(timestamp) {
    const date = new Date(timestamp);
    return date.getHours() + date.getMinutes() / 60;
  }

  function draw() {
    const userId = userSelect.value;
    const day = daySelect.value;
    const user = users[userId];
    const episode = user.episodes.find((item) => item.d === day);
    const dayEvents = user.events.filter((item) => item.t.slice(0, 10) === day);
    const dayTruth = user.trajectories.filter((item) => item.t.slice(0, 10) === day);
    const width = Math.max(320, Math.round(svg.getBoundingClientRect().width || 736));
    const height = width <= 520 ? 300 : 270;
    const left = width <= 520 ? 68 : 82;
    const right = 12;
    const top = 34;
    const bottom = 30;
    const lanes = [
      {key: "truth", label: "True path", y: top + 10},
      {key: "passive", label: "Passive", y: top + 52},
      {key: "local_commerce", label: "Local", y: top + 94},
      {key: "ecommerce", label: "E-commerce", y: top + 136},
      {key: "travel", label: "Travel", y: top + 178},
    ];
    const plotWidth = width - left - right;
    const x = (value) => left + Math.max(0, Math.min(1, (value - 5) / 19)) * plotWidth;
    svg.replaceChildren();
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("height", String(height));
    const title = add("title", {id: "kse-svg-title"});
    title.textContent = "Latent and observed event timeline for " + userId + " on " + day;
    const desc = add("desc", {id: "kse-svg-desc"});
    desc.textContent = "Five aligned lanes compare true trajectory stops with passive location, local commerce, e-commerce, and travel events from 05:00 to midnight.";

    [6, 9, 12, 15, 18, 21, 24].forEach((tick) => {
      add("line", {x1: x(tick), y1: top - 10, x2: x(tick), y2: height - bottom, class: "grid"});
      const label = add("text", {x: x(tick), y: height - 8, class: "label", "text-anchor": tick === 24 ? "end" : tick === 6 ? "start" : "middle"});
      label.textContent = String(tick).padStart(2, "0") + ":00";
    });
    lanes.forEach((lane) => {
      add("line", {x1: left, y1: lane.y, x2: width - right, y2: lane.y, class: "axis"});
      const label = add("text", {x: left - 8, y: lane.y + 4, class: "lane-label", "text-anchor": "end"});
      label.textContent = lane.label;
    });

    if (dayTruth.length) {
      const points = dayTruth.map((item) => x(hour(item.t)) + "," + lanes[0].y).join(" ");
      add("polyline", {points, class: "truth-line"});
      dayTruth.forEach((item) => add("rect", {x: x(hour(item.t)) - 4, y: lanes[0].y - 4, width: 8, height: 8, rx: 1, class: "truth-mark"}));
    }
    const laneByService = Object.fromEntries(lanes.map((lane) => [lane.key, lane.y]));
    const classByService = {
      location: "passive-mark",
      local_commerce: "local-mark",
      ecommerce: "ecommerce-mark",
      travel: "travel-mark",
    };
    dayEvents.forEach((item) => {
      const cy = item.s === "location" ? laneByService.passive : laneByService[item.s];
      add("circle", {cx: x(hour(item.t)), cy, r: item.m === "transaction" ? 5 : 4, class: classByService[item.s]});
    });
    if (!dayEvents.length) {
      const empty = add("text", {x: left + plotWidth / 2, y: lanes[2].y + 4, class: "empty-label", "text-anchor": "middle"});
      empty.textContent = "No events were observed for this user on this day";
    }

    root.querySelector("#kse-intent").textContent = episode ? episode.p.replaceAll("_", " ") : "no episode";
    root.querySelector("#kse-context").textContent = episode ? episode.o + (episode.o === episode.x ? "" : " → " + episode.x) + (episode.s === "none" ? "" : " · secondary: " + episode.s.replaceAll("_", " ")) : "";
    const byService = [...new Set(dayEvents.map((item) => item.s))].map((service) => service.replaceAll("_", " "));
    const regions = [...new Set(dayTruth.map((item) => item.r))];
    root.querySelector("#kse-detail").textContent = dayTruth.length + " true stops · " + dayEvents.length + " observed events" + (byService.length ? " across " + byService.join(", ") : "") + (regions.length ? " · true regions: " + regions.join(" → ") : "");
  }

  userSelect.addEventListener("change", refreshDays);
  daySelect.addEventListener("change", draw);
  const resizeObserver = new ResizeObserver(draw);
  resizeObserver.observe(root.querySelector(".kse-chart-wrap"));
  refreshDays();
})();
</script>
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("kanto-simulator-explorer.html"))
    parser.add_argument("--sample-users", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_payload(args.dataset.resolve(), args.report.resolve(), args.sample_users)
    rendered = FRAGMENT.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    if len(rendered.encode("utf-8")) >= 1_000_000:
        raise SystemExit("Explorer exceeds the 1 MB visualization limit; reduce --sample-users")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "bytes": len(rendered.encode("utf-8")), "sample_users": len(payload["users"])}, indent=2))


if __name__ == "__main__":
    main()
