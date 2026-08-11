# %% [markdown]
# # GeoEmbeddings Kanto simulator: visualization and validation
#
# This notebook inspects whether a generated synthetic world is internally
# coherent and whether its intended mechanisms are visible in the data.
#
# It deliberately reads both `observed/` and `truth/`, so it is an evaluator
# notebook—not training code. Training pipelines must read `observed/` only.
#
# From the simulator package directory, a clean environment can be started with:
#
# ```bash
# uv run jupyter lab
# ```

# %%
from __future__ import annotations

import json
import math
import os
from datetime import timedelta
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap, MarkerCluster

try:
    from IPython.display import display
except ImportError:  # Allows the cells to be smoke-tested as an ordinary script.
    display = print

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.max_columns", 30)

# Set GEOEMBED_RUN_DIR to inspect a different simulator run.
DATA_DIR = Path(os.environ.get("GEOEMBED_RUN_DIR", "runs/kanto_pilot"))
SCENARIO_REPORT = DATA_DIR / "scenario_validation.json"
ARTIFACT_DIR = DATA_DIR / "notebook_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_COLORS = {
    "location": "#4C78A8",
    "local_commerce": "#F58518",
    "ecommerce": "#54A24B",
    "travel": "#E45756",
}
MODE_COLORS = {
    "passive": "#4C78A8",
    "user_triggered": "#F58518",
    "transaction": "#54A24B",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a reproducible copy while keeping the figure visible in Jupyter."""
    fig.savefig(ARTIFACT_DIR / name, dpi=160, bbox_inches="tight")
    plt.show()


assert DATA_DIR.exists(), f"Dataset not found: {DATA_DIR.resolve()}"

# %% [markdown]
# ## 1. Load the simulator tables

# %%
TABLE_PATHS = {
    "events": "observed/observed_events.csv.gz",
    "users": "observed/users_observed.csv.gz",
    "latents": "truth/user_latents.csv.gz",
    "episodes": "truth/episodes_truth.csv.gz",
    "candidates": "truth/candidate_sets.csv.gz",
    "choices": "truth/choices_truth.csv.gz",
    "trajectories": "truth/trajectories_truth.csv.gz",
    "observation": "truth/observation_process.csv.gz",
}


def read_table(name: str) -> pd.DataFrame:
    path = DATA_DIR / TABLE_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


tables = {name: read_table(name) for name in TABLE_PATHS}
events = tables["events"]
users = tables["users"]
latents = tables["latents"]
episodes = tables["episodes"]
candidates = tables["candidates"]
choices = tables["choices"]
trajectories = tables["trajectories"]
observation = tables["observation"]

events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
trajectories["timestamp"] = pd.to_datetime(trajectories["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
episodes["start_time"] = pd.to_datetime(episodes["start_time"], utc=True).dt.tz_convert("Asia/Tokyo")
episodes["end_time"] = pd.to_datetime(episodes["end_time"], utc=True).dt.tz_convert("Asia/Tokyo")
choices["timestamp"] = pd.to_datetime(choices["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")

manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
deep_report_path = DATA_DIR / "deep_validation_report.json"
deep_report = json.loads(deep_report_path.read_text(encoding="utf-8")) if deep_report_path.exists() else None

overview = pd.DataFrame(
    {
        "table": list(TABLE_PATHS),
        "rows": [len(tables[name]) for name in TABLE_PATHS],
        "columns": [tables[name].shape[1] for name in TABLE_PATHS],
        "side": ["observed" if path.startswith("observed/") else "truth" for path in TABLE_PATHS.values()],
    }
)
display(overview)

# %% [markdown]
# ## 2. Validation checklist
#
# This summarizes the deep validator if its report is present. The final section
# independently recomputes the most important checks from the tables.

# %%
if deep_report:
    report_checks = pd.DataFrame(deep_report["checks"])
    report_checks["result"] = np.where(report_checks["passed"], "PASS", "FAIL")
    display(report_checks[["result", "layer", "severity", "name", "value", "expectation"]])
    print(
        f"Report status: {deep_report['status'].upper()} | "
        f"{deep_report['summary']['checks_passed']}/{deep_report['summary']['checks_total']} checks passed"
    )
else:
    print("No deep_validation_report.json found; continue to the independent checks below.")

# %% [markdown]
# ## 3. Dataset coverage and balance

# %%
events_by_user = events.groupby("user_id").size().reindex(users["user_id"], fill_value=0)
services_per_user = events.groupby("user_id")["service_id"].nunique().reindex(users["user_id"], fill_value=0)
service_counts = events["service_id"].value_counts()
mode_counts = events["observation_mode"].value_counts()
intent_counts = episodes["primary_intent"].value_counts()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

service_counts.sort_values().plot.barh(
    ax=axes[0, 0],
    color=[SERVICE_COLORS.get(x, "#999999") for x in service_counts.sort_values().index],
)
axes[0, 0].set(title="Observed events by service", xlabel="Event count", ylabel="")

mode_counts.sort_values().plot.barh(
    ax=axes[0, 1],
    color=[MODE_COLORS.get(x, "#999999") for x in mode_counts.sort_values().index],
)
axes[0, 1].set(title="Observed events by mode", xlabel="Event count", ylabel="")

sns.histplot(events_by_user, bins=25, ax=axes[1, 0], color="#4C78A8")
axes[1, 0].axvline(events_by_user.median(), color="#E45756", linestyle="--", label=f"median={events_by_user.median():.0f}")
axes[1, 0].set(title="Observed events per user", xlabel="Events", ylabel="Users")
axes[1, 0].legend()

intent_counts.sort_values().plot.barh(ax=axes[1, 1], color="#72B7B2")
axes[1, 1].set(title="Latent episode coverage", xlabel="Episodes", ylabel="")

fig.suptitle(f"Coverage diagnostics — {manifest['scenario']} scenario", y=1.01)
fig.tight_layout()
save_figure(fig, "01_coverage_diagnostics.png")

coverage_summary = pd.Series(
    {
        "users": len(users),
        "observed_events": len(events),
        "zero_event_users": int((events_by_user == 0).sum()),
        "users_with_2+_services_share": float((services_per_user >= 2).mean()),
        "event_p10": float(events_by_user.quantile(0.10)),
        "event_p50": float(events_by_user.quantile(0.50)),
        "event_p90": float(events_by_user.quantile(0.90)),
    },
    name="value",
)
display(coverage_summary.to_frame())

# %% [markdown]
# ## 3.1 Continuous population geography and overlap
#
# Home/work coordinates live only in evaluator truth. They are sampled from
# overlapping catchments rather than copied from region centroids. The scatter
# makes overlap visible; the OpenStreetMap view supplies geographic context.
# No API key is required, although tiles are downloaded when the map is viewed.

# %%
home_points = latents[["user_id", "home_region_id", "home_latitude", "home_longitude"]].copy()

coords = np.radians(home_points[["home_latitude", "home_longitude"]].to_numpy())
region_values = home_points["home_region_id"].to_numpy()
nearest_cross_region_km = np.full(len(home_points), np.inf)
for idx in range(len(home_points)):
    mask = region_values != region_values[idx]
    if not mask.any():
        continue
    lat1, lon1 = coords[idx]
    lat2, lon2 = coords[mask, 0], coords[mask, 1]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    nearest_cross_region_km[idx] = float(
        (6371.0 * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0.0, 1 - value)))).min()
    )

overlap_summary = pd.Series(
    {
        "unique_home_coordinate_share": len(home_points.drop_duplicates(["home_latitude", "home_longitude"])) / len(home_points),
        "cross_region_neighbor_within_3km_share": np.mean(nearest_cross_region_km <= 3.0),
        "cross_region_neighbor_within_5km_share": np.mean(nearest_cross_region_km <= 5.0),
        "median_nearest_cross_region_km": np.median(nearest_cross_region_km[np.isfinite(nearest_cross_region_km)]),
    },
    name="value",
)
display(overlap_summary.to_frame())

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sns.scatterplot(
    data=home_points,
    x="home_longitude",
    y="home_latitude",
    hue="home_region_id",
    alpha=0.48,
    s=25,
    linewidth=0,
    ax=ax,
)
ax.set(
    title="Synthetic home locations: continuous, overlapping catchments",
    xlabel="Longitude",
    ylabel="Latitude",
)
ax.legend(title="Assigned hub", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.ticklabel_format(axis="both", style="plain", useOffset=False)
fig.tight_layout()
save_figure(fig, "01b_population_spatial_overlap.png")

# %%
population_map = folium.Map(
    location=[home_points["home_latitude"].median(), home_points["home_longitude"].median()],
    zoom_start=8,
    tiles="OpenStreetMap",
    control_scale=True,
)
HeatMap(
    home_points[["home_latitude", "home_longitude"]].values.tolist(),
    radius=13,
    blur=17,
    min_opacity=0.25,
    name="Synthetic home density",
).add_to(population_map)
sampled_homes = home_points.sample(min(600, len(home_points)), random_state=7)
cluster = MarkerCluster(name="Sampled synthetic users", show=False).add_to(population_map)
for row in sampled_homes.itertuples(index=False):
    folium.CircleMarker(
        [row.home_latitude, row.home_longitude],
        radius=3,
        weight=1,
        fill=True,
        fill_opacity=0.60,
        tooltip=f"{row.user_id} · {row.home_region_id}",
    ).add_to(cluster)
folium.LayerControl(collapsed=False).add_to(population_map)
population_map.save(ARTIFACT_DIR / "01c_population_openstreetmap.html")
display(population_map)

# %% [markdown]
# ## 4. Inspect one latent-versus-observed user-day
#
# By default, the notebook selects the first travel episode because it exercises
# the cross-region and cross-service parts of the simulator. Override `USER_ID`
# and `DAY` to inspect any other case.

# %%
travel_episodes = episodes.loc[episodes["primary_intent"].eq("travel")]
selected_episode = travel_episodes.iloc[0] if len(travel_episodes) else episodes.iloc[0]

USER_ID = selected_episode["user_id"]
DAY = selected_episode["start_time"].date()

day_start = pd.Timestamp(DAY, tz="Asia/Tokyo")
day_end = day_start + timedelta(days=1)
truth_day = trajectories.loc[
    trajectories["user_id"].eq(USER_ID)
    & trajectories["timestamp"].between(day_start, day_end, inclusive="left")
].sort_values("timestamp")
observed_day = events.loc[
    events["user_id"].eq(USER_ID)
    & events["timestamp"].between(day_start, day_end, inclusive="left")
].sort_values("timestamp")

print(f"Selected {USER_ID} on {DAY}: {len(truth_day)} truth stops and {len(observed_day)} observed events")
display(selected_episode.to_frame("value"))

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.05, 1.45]})

# Spatial trace. The axes are independently scaled so long, nearly north-south
# trips remain inspectable; the panel is a diagnostic trace, not a distance map.
axes[0].plot(
    truth_day["true_longitude"],
    truth_day["true_latitude"],
    color="#333333",
    linewidth=1.5,
    alpha=0.7,
    zorder=1,
)
axes[0].scatter(
    truth_day["true_longitude"],
    truth_day["true_latitude"],
    marker="x",
    s=55,
    color="#333333",
    label="Latent stop",
    zorder=3,
)
for service, group in observed_day.groupby("service_id"):
    axes[0].scatter(
        group["longitude"],
        group["latitude"],
        s=42,
        alpha=0.80,
        color=SERVICE_COLORS.get(service, "#999999"),
        label=f"Observed: {service}",
        zorder=2,
    )
axes[0].set(title="Spatial trace", xlabel="Longitude", ylabel="Latitude")
axes[0].ticklabel_format(axis="both", style="plain", useOffset=False)
axes[0].legend(fontsize=8, loc="best")

# Timeline aligned on one x-axis.
lanes = ["latent"] + sorted(observed_day["service_id"].unique().tolist())
lane_y = {name: idx for idx, name in enumerate(lanes)}
axes[1].scatter(
    truth_day["timestamp"],
    np.repeat(lane_y["latent"], len(truth_day)),
    marker="x",
    s=55,
    color="#333333",
    label="Latent stops",
)
for service, group in observed_day.groupby("service_id"):
    axes[1].scatter(
        group["timestamp"],
        np.repeat(lane_y[service], len(group)),
        s=45,
        color=SERVICE_COLORS.get(service, "#999999"),
    )
axes[1].set_yticks(range(len(lanes)), lanes)
axes[1].set(title="Latent and observed event timing", xlabel="Local time", ylabel="")
axes[1].grid(axis="x", alpha=0.3)
axes[1].grid(axis="y", visible=False)
fig.autofmt_xdate()
fig.suptitle(f"{USER_ID} — {DAY} — {selected_episode['primary_intent']}", y=1.01)
fig.tight_layout()
save_figure(fig, "02_user_day_latent_vs_observed.png")

# %% [markdown]
# ### OpenStreetMap trajectory view
#
# Toggle the latent and service layers to inspect observation gaps or spatial
# noise. Lines connect simulated stops directly; they do not claim to follow
# actual roads or rail lines.

# %%
map_center = [truth_day["true_latitude"].mean(), truth_day["true_longitude"].mean()]
trajectory_map = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)

truth_layer = folium.FeatureGroup(name="Latent trajectory", show=True).add_to(trajectory_map)
truth_coordinates = truth_day[["true_latitude", "true_longitude"]].values.tolist()
folium.PolyLine(truth_coordinates, weight=4, opacity=0.75, tooltip="Latent stop sequence").add_to(truth_layer)
for row in truth_day.itertuples(index=False):
    folium.CircleMarker(
        [row.true_latitude, row.true_longitude],
        radius=6,
        weight=2,
        fill=True,
        fill_opacity=0.85,
        tooltip=f"truth · {row.timestamp:%H:%M} · {row.activity}",
    ).add_to(truth_layer)

for service, group in observed_day.groupby("service_id"):
    layer = folium.FeatureGroup(name=f"Observed: {service}", show=True).add_to(trajectory_map)
    for row in group.itertuples(index=False):
        folium.CircleMarker(
            [row.latitude, row.longitude],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.75,
            tooltip=f"{service} · {row.timestamp:%H:%M} · {row.action_type}",
        ).add_to(layer)

if truth_coordinates:
    trajectory_map.fit_bounds(truth_coordinates, padding=(25, 25))
folium.LayerControl(collapsed=False).add_to(trajectory_map)
trajectory_map.save(ARTIFACT_DIR / "02b_user_day_openstreetmap.html")
display(trajectory_map)

# %% [markdown]
# ## 5. Passive-location coherence and GPS noise

# Every passive event should align with a same-user latent trajectory stop within
# 15 minutes. The spatial displacement should be nonzero but consistent with the
# configured synthetic GPS noise.

# %%
passive = events.loc[events["observation_mode"].eq("passive")].copy()
matched_parts = []
for user_id, observed_group in passive.groupby("user_id", sort=False):
    truth_group = trajectories.loc[trajectories["user_id"].eq(user_id)].sort_values("timestamp")
    observed_group = observed_group.sort_values("timestamp")
    if truth_group.empty:
        continue
    matched = pd.merge_asof(
        observed_group,
        truth_group[["timestamp", "true_region_id", "true_latitude", "true_longitude"]],
        on="timestamp",
        direction="nearest",
        tolerance=timedelta(minutes=15),
    )
    matched_parts.append(matched)

passive_match = pd.concat(matched_parts, ignore_index=True)
passive_match["matched"] = passive_match["true_latitude"].notna()


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1 - a)))


matched_only = passive_match.loc[passive_match["matched"]].copy()
matched_only["gps_error_m"] = haversine_m(
    matched_only["latitude"].to_numpy(),
    matched_only["longitude"].to_numpy(),
    matched_only["true_latitude"].to_numpy(),
    matched_only["true_longitude"].to_numpy(),
)

alignment_summary = pd.Series(
    {
        "passive_events": len(passive_match),
        "match_rate_within_15m": passive_match["matched"].mean(),
        "region_match_rate": (matched_only["region_id"] == matched_only["true_region_id"]).mean(),
        "gps_error_median": matched_only["gps_error_m"].median(),
        "gps_error_p95": matched_only["gps_error_m"].quantile(0.95),
    },
    name="value",
)
display(alignment_summary.to_frame())

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(
    matched_only["gps_error_m"].clip(upper=matched_only["gps_error_m"].quantile(0.995)),
    bins=40,
    ax=axes[0],
    color="#4C78A8",
)
axes[0].axvline(matched_only["gps_error_m"].median(), color="#E45756", linestyle="--", label="median")
axes[0].set(title="Observed-to-truth displacement", xlabel="GPS error (m)", ylabel="Passive events")
axes[0].legend()

sample = matched_only.sample(min(5000, len(matched_only)), random_state=7)
sns.scatterplot(
    data=sample,
    x="location_accuracy_m",
    y="gps_error_m",
    alpha=0.20,
    s=18,
    ax=axes[1],
    color="#4C78A8",
)
axes[1].set(
    title="Configured accuracy versus realized error",
    xlabel="Reported location accuracy (m)",
    ylabel="Realized GPS error (m)",
)
fig.tight_layout()
save_figure(fig, "03_passive_alignment_and_gps_noise.png")

# %% [markdown]
# ## 6. Choice mechanism validation
#
# The plots test three intended effects: nearer alternatives should be selected
# more often, exposed alternatives should be overrepresented among choices, and
# selected alternatives should have higher total utility on average.

# %%
candidate_plot = candidates.copy()
candidate_plot["choice_status"] = np.where(candidate_plot["is_chosen"].eq(1), "chosen", "not chosen")
if len(candidate_plot) > 30_000:
    chosen_rows = candidate_plot.loc[candidate_plot["is_chosen"].eq(1)]
    nonchosen_rows = candidate_plot.loc[candidate_plot["is_chosen"].eq(0)].sample(
        min(23_000, int((candidate_plot["is_chosen"] == 0).sum())), random_state=7
    )
    candidate_plot = pd.concat([chosen_rows, nonchosen_rows], ignore_index=True)

choice_metrics = pd.Series(
    {
        "chosen_mean_distance_km": candidates.loc[candidates["is_chosen"].eq(1), "distance_km"].mean(),
        "all_candidates_mean_distance_km": candidates["distance_km"].mean(),
        "chosen_exposed_share": candidates.loc[candidates["is_chosen"].eq(1), "exposed"].mean(),
        "all_candidates_exposed_share": candidates["exposed"].mean(),
        "chosen_mean_utility": candidates.loc[candidates["is_chosen"].eq(1), "utility_total"].mean(),
        "not_chosen_mean_utility": candidates.loc[candidates["is_chosen"].eq(0), "utility_total"].mean(),
    },
    name="value",
)
display(choice_metrics.to_frame())

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.ecdfplot(
    data=candidate_plot,
    x="distance_km",
    hue="choice_status",
    hue_order=["not chosen", "chosen"],
    palette=["#BAB0AC", "#72B7B2"],
    ax=axes[0],
)
axes[0].set_xlim(0, candidate_plot["distance_km"].quantile(0.98))
axes[0].set(title="Geographic opportunity", xlabel="Candidate distance (km)", ylabel="Cumulative share")

exposure = (
    candidates.groupby(candidates["is_chosen"].map({0: "all alternatives", 1: "chosen"}))["exposed"]
    .mean()
    .reindex(["all alternatives", "chosen"])
)
exposure.plot.bar(ax=axes[1], color=["#BAB0AC", "#F58518"])
axes[1].set(title="Exposure effect", xlabel="", ylabel="Exposed share", ylim=(0, 1))
axes[1].tick_params(axis="x", rotation=0)

sns.kdeplot(
    data=candidate_plot,
    x="utility_total",
    hue="choice_status",
    hue_order=["not chosen", "chosen"],
    common_norm=False,
    fill=False,
    ax=axes[2],
    palette=["#BAB0AC", "#54A24B"],
)
axes[2].set(title="Choice responds to utility", xlabel="Total utility", ylabel="Density")

fig.tight_layout()
save_figure(fig, "04_choice_mechanisms.png")

# %% [markdown]
# ## 7. Non-random observability
#
# This verifies that recorded event volume depends on latent digital engagement,
# rather than missing completely at random. That bias is a deliberate simulator
# feature and must be visible to downstream evaluations.

# %%
observability = (
    latents[["user_id", "digital_engagement"]]
    .merge(events_by_user.rename("observed_events"), left_on="user_id", right_index=True, how="left")
    .fillna({"observed_events": 0})
)
observability_correlation = observability[["digital_engagement", "observed_events"]].corr().iloc[0, 1]

fig, ax = plt.subplots(figsize=(8, 5.5))
sns.regplot(
    data=observability,
    x="digital_engagement",
    y="observed_events",
    scatter_kws={"alpha": 0.35, "s": 24},
    line_kws={"color": "#E45756"},
    ax=ax,
)
ax.set(
    title=f"Digital engagement and observability (r={observability_correlation:.3f})",
    xlabel="Latent digital engagement",
    ylabel="Recorded events",
)
fig.tight_layout()
save_figure(fig, "05_observability_bias.png")

# %% [markdown]
# ## 8. Controlled scenario comparisons
#
# If `kanto_scenario_validation.json` is present, this section visualizes whether
# each stress scenario moves its target mechanism in the intended direction.

# %%
if SCENARIO_REPORT.exists():
    scenario_report = json.loads(SCENARIO_REPORT.read_text(encoding="utf-8"))
    comparisons = scenario_report.get("scenario_validation", {}).get("comparisons", [])
    scenario_rows = []
    for comparison in comparisons:
        for key, value in comparison["value"].items():
            scenario_rows.append(
                {
                    "comparison": comparison["name"],
                    "measure": key,
                    "value": value,
                    "passed": comparison["passed"],
                }
            )
    scenario_df = pd.DataFrame(scenario_rows)
    display(scenario_df)

    if not scenario_df.empty:
        names = scenario_df["comparison"].drop_duplicates().tolist()
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), squeeze=False)
        for ax, name in zip(axes.flat, names):
            group = scenario_df.loc[scenario_df["comparison"].eq(name)].copy()
            if "Observation-bias" in name:
                values = dict(zip(group["measure"], group["value"]))
                ratios = pd.Series(
                    {
                        "recorded events": values["biased_events"] / values["mixed_events"],
                        "engagement correlation": values["biased_correlation"] / values["mixed_correlation"],
                    }
                )
                ratios.plot.bar(ax=ax, color=["#4C78A8", "#F58518"])
                ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="mixed baseline")
                ax.set(title="Observation bias\nrelative to mixed", xlabel="", ylabel="Ratio to mixed")
                ax.tick_params(axis="x", rotation=25)
                ax.legend(fontsize=8)
            else:
                sns.barplot(data=group, x="measure", y="value", hue="measure", legend=False, ax=ax, palette="muted")
                ax.set(title=name.replace(" scenario", "\nscenario"), xlabel="", ylabel="Value")
                ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        save_figure(fig, "06_scenario_controls.png")
else:
    print(f"Scenario comparison report not found: {SCENARIO_REPORT}")

# %% [markdown]
# ## 9. Independent executable validation
#
# These checks are recomputed directly from the current tables. A failing error
# check raises an exception. Warnings remain visible without stopping execution.

# %%
FORBIDDEN_FRAGMENTS = (
    "latent",
    "utility",
    "episode_id",
    "true_",
    "price_sensitivity",
    "distance_sensitivity",
    "digital_engagement",
    "travel_propensity",
)

observed_columns = set(events.columns) | set(users.columns)
leaked_columns = sorted(
    column for column in observed_columns if any(fragment in column for fragment in FORBIDDEN_FRAGMENTS)
)
manifest_mismatches = {
    path: {"manifest": manifest["table_rows"].get(path), "actual": len(tables[name])}
    for name, path in TABLE_PATHS.items()
    if manifest["table_rows"].get(path) != len(tables[name])
}

candidate_selected = candidates.groupby("decision_id")["is_chosen"].sum()
choice_ids = set(choices["decision_id"])
candidate_ids = set(candidates["decision_id"])
user_ids = set(users["user_id"])
foreign_key_failures = sum(
    (~frame["user_id"].isin(user_ids)).sum()
    for frame in [events, latents, episodes, trajectories, observation]
)

validation_rows = [
    {
        "check": "Manifest row counts",
        "passed": not manifest_mismatches,
        "value": manifest_mismatches or "all match",
        "severity": "error",
    },
    {
        "check": "Observed/truth boundary",
        "passed": not leaked_columns,
        "value": leaked_columns or "no forbidden columns",
        "severity": "error",
    },
    {
        "check": "User foreign keys",
        "passed": foreign_key_failures == 0,
        "value": int(foreign_key_failures),
        "severity": "error",
    },
    {
        "check": "Continuous user locations",
        "passed": bool(overlap_summary["unique_home_coordinate_share"] >= 0.99),
        "value": round(float(overlap_summary["unique_home_coordinate_share"]), 3),
        "severity": "error",
    },
    {
        "check": "Neighboring hub catchments overlap",
        "passed": bool(overlap_summary["cross_region_neighbor_within_5km_share"] >= 0.10),
        "value": round(float(overlap_summary["cross_region_neighbor_within_5km_share"]), 3),
        "severity": "error",
    },
    {
        "check": "One selected candidate per decision",
        "passed": bool((candidate_selected == 1).all() and choice_ids == candidate_ids),
        "value": int((candidate_selected != 1).sum()),
        "severity": "error",
    },
    {
        "check": "Passive events align within 15 minutes",
        "passed": bool(passive_match["matched"].mean() >= 0.99),
        "value": round(float(passive_match["matched"].mean()), 5),
        "severity": "error",
    },
    {
        "check": "Chosen POIs are nearer",
        "passed": bool(choice_metrics["chosen_mean_distance_km"] < choice_metrics["all_candidates_mean_distance_km"]),
        "value": round(float(choice_metrics["chosen_mean_distance_km"]), 3),
        "severity": "error",
    },
    {
        "check": "Chosen POIs are more often exposed",
        "passed": bool(choice_metrics["chosen_exposed_share"] > choice_metrics["all_candidates_exposed_share"]),
        "value": round(float(choice_metrics["chosen_exposed_share"]), 3),
        "severity": "error",
    },
    {
        "check": "Chosen POIs have higher utility",
        "passed": bool(choice_metrics["chosen_mean_utility"] > choice_metrics["not_chosen_mean_utility"]),
        "value": round(float(choice_metrics["chosen_mean_utility"]), 3),
        "severity": "error",
    },
    {
        "check": "Observability is non-random",
        "passed": bool(observability_correlation > 0.15),
        "value": round(float(observability_correlation), 3),
        "severity": "error",
    },
    {
        "check": "All episode types are present",
        "passed": {"routine", "shopping", "leisure", "family_outing", "travel"}.issubset(intent_counts.index),
        "value": sorted(intent_counts.index.tolist()),
        "severity": "error",
    },
    {
        "check": "Travel events are sufficient for a sparse probe",
        "passed": int(service_counts.get("travel", 0)) >= 100,
        "value": int(service_counts.get("travel", 0)),
        "severity": "warning",
    },
]

validation = pd.DataFrame(validation_rows)
validation.insert(0, "result", np.where(validation["passed"], "PASS", "FAIL"))
display(validation)
validation.to_csv(ARTIFACT_DIR / "notebook_validation.csv", index=False)

error_failures = validation.loc[(~validation["passed"]) & validation["severity"].eq("error")]
warning_failures = validation.loc[(~validation["passed"]) & validation["severity"].eq("warning")]

assert error_failures.empty, "Validation failed:\n" + error_failures.to_string(index=False)
print(
    f"VALIDATION PASSED: {validation['passed'].sum()}/{len(validation)} checks passed; "
    f"{len(warning_failures)} warning(s)."
)
print(f"Figures and validation CSV saved to: {ARTIFACT_DIR.resolve()}")

# %% [markdown]
# ## Reading the result
#
# A healthy simulator run should show all of the following together:
#
# - observed events cover several services and both passive and active modes;
# - home locations are continuous and neighboring hub catchments overlap;
# - the OpenStreetMap population and trajectory layers are geographically plausible;
# - latent stops and observed passive events align temporally while retaining
#   nonzero spatial noise;
# - choices respond to distance, exposure, and utility;
# - digital engagement predicts observability, confirming deliberate non-random
#   missingness;
# - all intended episode types appear;
# - stress scenarios move their targeted mechanisms in the expected direction.
#
# These checks validate the synthetic data-generating process. They do not yet
# establish that an embedding model recovers the hidden factors; that is the
# next representation-gauntlet notebook.
