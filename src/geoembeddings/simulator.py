#!/usr/bin/env python3
"""Kanto semi-synthetic mobility and cross-service behavior simulator.

The simulator deliberately separates what a platform observes from the latent
causes used to generate it. Its data-generating process is configured in YAML
and is exposed through the unified `geoembed simulate` command.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from .contract import (
    DATASET_CONTRACT_NAME,
    DATASET_CONTRACT_VERSION,
    OBSERVED_FILES,
    TRUTH_FILES,
    SIMULATION_IDENTITY_HASH_ALGORITHM,
    SIMULATION_IDENTITY_MANIFEST_SCHEMA,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path.
    raise SystemExit(
        "PyYAML is required for simulator configuration. Install the project with `uv sync`."
    ) from exc


JST = timezone(timedelta(hours=9))


SIMULATOR_VERSION = "0.4.0"
RANDOM_STREAM_ALGORITHM = "sha256-root-seed-and-stream-name/1.0"
RANDOM_STREAM_NAMES = ("world", "user_latents", "episodes", "choices", "observation")
IDENTITY_GENERATION_VERSION = "sha256-semantic-key/1.0"


def stable_identifier(namespace: str, *components: object) -> str:
    """Return a durable ID from a typed semantic key, never row order or ``hash()``."""
    if not namespace or not components or any(component is None or str(component) == "" for component in components):
        raise ValueError("stable identifiers require a namespace and non-empty semantic components")
    material = json.dumps(
        {"version": IDENTITY_GENERATION_VERSION, "namespace": namespace, "components": [str(value) for value in components]},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return f"{namespace}_{hashlib.sha256(material).hexdigest()[:24]}"


def identity_set_hash(identifiers: Sequence[str]) -> str:
    """Hash an identity set canonically so CSV row reordering cannot alter it."""
    values = [str(value) for value in identifiers]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("identity hashing requires unique, non-empty identifiers")
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RandomStreams:
    """Independent simulator RNGs and their reproducible seed provenance."""

    root_seed: int
    seeds: dict[str, int]
    generators: dict[str, random.Random]


def derive_stream_seed(root_seed: int, name: str) -> int:
    """Derive a stable seed without Python's process-randomized ``hash``."""
    if name not in RANDOM_STREAM_NAMES:
        raise ValueError(f"Unknown simulator random stream: {name!r}")
    material = f"{RANDOM_STREAM_ALGORITHM}\0{root_seed}\0{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def make_random_streams(root_seed: int, overrides: dict[str, Any] | None = None) -> RandomStreams:
    """Resolve configured overrides and construct all named generators."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(RANDOM_STREAM_NAMES))
    if unknown:
        raise ValueError(f"Unknown run.random_streams entries: {unknown}")
    seeds = {
        name: int(overrides[name]) if name in overrides else derive_stream_seed(root_seed, name)
        for name in RANDOM_STREAM_NAMES
    }
    return RandomStreams(int(root_seed), seeds, {name: random.Random(seed) for name, seed in seeds.items()})


@dataclass(frozen=True)
class Region:
    region_id: str
    name: str
    prefecture: str
    lat: float
    lon: float
    density: float
    price_index: float
    holdout: bool


CONFIG: dict[str, Any] = {}
REGIONS: tuple[Region, ...] = ()
POI_CATEGORIES: dict[str, tuple[float, float]] = {}
SCENARIO_SETTINGS: dict[str, dict[str, float]] = {}


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the simulator's external YAML contract."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping.")
    required = {"run", "world", "population", "episodes", "choice", "observation", "events", "scenarios", "interventions"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    if not raw["world"].get("regions") or not raw["world"].get("poi_categories"):
        raise ValueError("world.regions and world.poi_categories cannot be empty.")
    if raw["run"].get("scenario") not in raw["scenarios"]:
        raise ValueError("run.scenario must name an entry under scenarios.")
    for name in ("exposure", "opportunity", "observation"):
        definition = raw["interventions"].get(name)
        if not isinstance(definition, dict) or not definition.get("config_overrides"):
            raise ValueError(f"interventions.{name} must declare config_overrides")
    return raw


def activate_config(config: dict[str, Any]) -> None:
    """Expose resolved configuration to the small pure generation helpers."""
    global CONFIG, REGIONS, POI_CATEGORIES, SCENARIO_SETTINGS
    CONFIG = config
    REGIONS = tuple(
        Region(
            str(row["id"]),
            str(row["name"]),
            str(row["prefecture"]),
            float(row["lat"]),
            float(row["lon"]),
            float(row["density"]),
            float(row["price_index"]),
            bool(row.get("holdout", False)),
        )
        for row in config["world"]["regions"]
    )
    POI_CATEGORIES = {
        str(name): (float(values["appeal"]), float(values["base_price"]))
        for name, values in config["world"]["poi_categories"].items()
    }
    SCENARIO_SETTINGS = {
        str(name): {str(key): float(value) for key, value in values.items()}
        for name, values in config["scenarios"].items()
    }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 6371.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int) -> str:
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits = (16, 8, 4, 2, 1)
    even, bit, ch = True, 0, 0
    result: list[str] = []
    while len(result) < precision:
        interval = lon_range if even else lat_range
        value = lon if even else lat
        mid = (interval[0] + interval[1]) / 2
        if value >= mid:
            ch |= bits[bit]
            interval[0] = mid
        else:
            interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            result.append(BASE32[ch])
            bit, ch = 0, 0
    return "".join(result)


def japan_mesh(lat: float, lon: float) -> tuple[str, str]:
    """Return approximate standard 1 km (third) and 500 m mesh codes."""
    lat_scaled = lat * 1.5
    p = int(lat_scaled)
    q = int(lon) - 100
    lat_second = (lat_scaled - p) * 8
    lon_second = (lon - int(lon)) * 8
    r, s = min(7, int(lat_second)), min(7, int(lon_second))
    lat_third = (lat_second - r) * 10
    lon_third = (lon_second - s) * 10
    t, u = min(9, int(lat_third)), min(9, int(lon_third))
    mesh_1km = f"{p:02d}{q:02d}{r}{s}{t}{u}"
    north = (lat_third - t) >= 0.5
    east = (lon_third - u) >= 0.5
    quadrant = 1 + int(east) + 2 * int(north)
    return mesh_1km, f"{mesh_1km}{quadrant}"


def jitter_point(rng: random.Random, lat: float, lon: float, km: float) -> tuple[float, float]:
    angle = rng.random() * 2 * math.pi
    radius = km * math.sqrt(rng.random())
    return lat + (radius * math.sin(angle)) / 111.0, lon + (radius * math.cos(angle)) / (111.0 * math.cos(math.radians(lat)))


def gaussian_point(rng: random.Random, lat: float, lon: float, sd_km: float, anisotropy: float = 1.0) -> tuple[float, float]:
    """Sample from an unbounded 2-D catchment so neighboring hubs can overlap."""
    north_km = rng.gauss(0.0, sd_km)
    east_km = rng.gauss(0.0, sd_km * anisotropy)
    return lat + north_km / 111.0, lon + east_km / (111.0 * math.cos(math.radians(lat)))


def weighted_choice(rng: random.Random, items: Sequence[Any], weights: Sequence[float]) -> Any:
    return rng.choices(items, weights=[max(1e-8, value) for value in weights], k=1)[0]


def softmax_choice(rng: random.Random, rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_utility = max(float(row["utility_total"]) for row in rows)
    weights = [math.exp(min(30.0, float(row["utility_total"]) - max_utility)) for row in rows]
    return weighted_choice(rng, rows, weights)


def iso_at(day: date, hour: float) -> str:
    # Round once in total minutes so values such as 21.999 do not become 21:00
    # when the minute component rounds from 59.94 to 60.
    total_minutes = int(round(hour * 60))
    return (datetime.combine(day, time(0, 0), tzinfo=JST) + timedelta(minutes=total_minutes)).isoformat()


def normal01(rng: random.Random, mean: float, sd: float = 0.18) -> float:
    return clamp(rng.gauss(mean, sd))


def make_world(rng: random.Random) -> list[dict[str, Any]]:
    spatial = CONFIG["world"]["spatial"]
    attributes = CONFIG["world"]["poi_attributes"]
    pois: list[dict[str, Any]] = []
    for region in REGIONS:
        for category, (appeal, base_price) in POI_CATEGORIES.items():
            tourism_bonus = (
                int(spatial["poi_tourism_bonus"])
                if category in set(spatial["tourism_categories"]) and region.region_id in set(spatial["tourism_regions"])
                else 0
            )
            count = max(
                int(spatial["poi_min_count"]),
                round(float(spatial["poi_base_count"]) + region.density * float(spatial["poi_density_count"]) + tourism_bonus),
            )
            for object_slot in range(count):
                spread = float(spatial["poi_spread_km_base"]) + (1.0 - region.density) * float(spatial["poi_spread_km_low_density_bonus"])
                lat, lon = gaussian_point(rng, region.lat, region.lon, spread, float(spatial["east_west_anisotropy"]))
                pois.append(
                    {
                        "poi_id": stable_identifier("poi", region.region_id, category, object_slot),
                        "region_id": region.region_id,
                        "prefecture": region.prefecture,
                        "category": category,
                        "lat": lat,
                        "lon": lon,
                        "quality": clamp(rng.gauss(appeal, float(attributes["quality_sd"]))),
                        "price": clamp(
                            rng.gauss(base_price * region.price_index, float(attributes["price_sd"])),
                            float(attributes["price_min"]),
                            float(attributes["price_max"]),
                        ),
                        "popularity": clamp(
                            rng.betavariate(
                                float(attributes["popularity_alpha_base"])
                                + region.density * float(attributes["popularity_alpha_density_weight"]),
                                float(attributes["popularity_beta"]),
                            )
                        ),
                    }
                )
    return pois


def create_user(rng: random.Random, index: int, full_kanto: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    population = CONFIG["population"]
    spatial = CONFIG["world"]["spatial"]
    available = list(REGIONS if full_kanto else [region for region in REGIONS if not region.holdout])
    home = weighted_choice(rng, available, [region.density ** float(population["home_density_power"]) for region in available])
    excluded_work = set(population["excluded_work_regions"])
    works = [region for region in REGIONS if not region.holdout and region.region_id not in excluded_work]
    work = weighted_choice(
        rng,
        works,
        [
            region.density
            * math.exp(-haversine_km(home.lat, home.lon, region.lat, region.lon) / float(population["work_commute_decay_km"]))
            for region in works
        ],
    )
    home_sd = float(spatial["home_spread_km_base"]) + (1.0 - home.density) * float(spatial["home_spread_km_low_density_bonus"])
    work_sd = float(spatial["work_spread_km_base"]) + (1.0 - work.density) * float(spatial["work_spread_km_low_density_bonus"])
    home_lat, home_lon = gaussian_point(rng, home.lat, home.lon, home_sd, float(spatial["east_west_anisotropy"]))
    work_lat, work_lon = gaussian_point(rng, work.lat, work.lon, work_sd, float(spatial["east_west_anisotropy"]))

    age_labels = list(population["age_groups"])
    age_group = weighted_choice(rng, age_labels, [population["age_groups"][name] for name in age_labels])
    household_labels = list(population["households"])
    household = weighted_choice(rng, household_labels, [population["households"][name] for name in household_labels])
    family_mean = float(population["family_orientation_means"]["family_household" if household in set(population["family_households"]) else "other"])
    digital_mean = float(population["digital_engagement_means"]["younger" if age_group in set(population["digitally_younger_groups"]) else "older"])
    latent_means = population["latent_means"]
    latent_sd = float(population["latent_sd"])
    latent = {
        "user_id": stable_identifier("user", index),
        "price_sensitivity": normal01(rng, float(latent_means["price_sensitivity"]), latent_sd),
        "distance_sensitivity": normal01(rng, float(latent_means["distance_sensitivity"]), latent_sd),
        "novelty_seeking": normal01(rng, float(latent_means["novelty_seeking"]), latent_sd),
        "family_orientation": normal01(rng, family_mean, latent_sd),
        "travel_propensity": normal01(rng, float(latent_means["travel_propensity"]), latent_sd),
        "time_flexibility": normal01(rng, float(latent_means["time_flexibility"]), latent_sd),
        "transit_preference": normal01(
            rng,
            float(population["transit_preference_means"]["dense" if home.density > float(population["dense_threshold"]) else "sparse"]),
            latent_sd,
        ),
        "digital_engagement": normal01(rng, digital_mean, latent_sd),
        "home_region_id": home.region_id,
        "work_region_id": work.region_id,
        "home_latitude": round(home_lat, 6),
        "home_longitude": round(home_lon, 6),
        "work_latitude": round(work_lat, 6),
        "work_longitude": round(work_lon, 6),
    }
    for category in population["preference_categories"]:
        latent[f"pref_{category}"] = normal01(rng, float(population["preference_mean"]), latent_sd)
    observed = {
        "user_id": latent["user_id"],
        "age_group": age_group,
        "household_type": household,
        "home_prefecture": home.prefecture,
        "home_region_id": home.region_id,
        "geo_split": "holdout" if home.holdout else "development",
    }
    return observed, latent


def make_observation_process(rng: random.Random, user: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    settings = SCENARIO_SETTINGS[scenario]
    observation = CONFIG["observation"]
    engagement = float(user["digital_engagement"])
    records: list[dict[str, Any]] = []
    services = {}
    for service, values in observation["services"].items():
        services[service] = clamp(
            float(values["adoption_intercept"])
            + float(values["engagement_coefficient"]) * engagement
            + float(values.get("travel_propensity_coefficient", 0.0)) * float(user["travel_propensity"])
        )
    for service, adoption_probability in services.items():
        adopted = rng.random() < adoption_probability
        service_cfg = observation["services"][service]
        base_record = float(service_cfg["base_record_probability"])
        if scenario == "observation_biased":
            record_probability = clamp(
                base_record
                * (
                    float(observation["biased_record_intercept"])
                    + float(observation["biased_record_engagement_coefficient"]) * engagement
                )
            )
        else:
            record_probability = clamp(
                base_record
                * (
                    float(observation["standard_record_intercept"])
                    + float(observation["standard_record_engagement_coefficient"]) * engagement
                )
            )
        gps_sd = (
            (float(observation["gps_min_m"]) + float(observation["gps_engagement_range_m"]) * (1.0 - engagement))
            * settings["gps_scale"]
            if service == "location"
            else float(service_cfg["event_location_accuracy_m"])
        )
        records.append(
            {
                "user_id": user["user_id"],
                "source_service": service,
                "service_adopted": int(adopted),
                "record_probability": round(record_probability / settings["dropout_scale"], 5),
                "gps_sd_m": round(gps_sd, 2),
                "passive_interval_minutes": round(
                    float(observation["passive_interval_min_minutes"])
                    + float(observation["passive_interval_engagement_range_minutes"]) * (1.0 - engagement)
                ),
            }
        )
    return records


def select_episode(rng: random.Random, user: dict[str, Any], day: date, is_weekend: bool) -> tuple[str, str]:
    episode_cfg = CONFIG["episodes"]
    weights_cfg = episode_cfg["weights"]
    travel = float(weights_cfg["travel"]["base"]) + float(weights_cfg["travel"]["latent"]) * float(user["travel_propensity"]) + (float(weights_cfg["travel"]["weekend"]) if is_weekend else 0.0)
    family = float(weights_cfg["family_outing"]["base"]) + float(weights_cfg["family_outing"]["weekend" if is_weekend else "weekday"]) * float(user["family_orientation"])
    leisure = float(weights_cfg["leisure"]["base"]) + float(weights_cfg["leisure"]["weekend" if is_weekend else "weekday"]) * float(user["novelty_seeking"])
    shopping = float(weights_cfg["shopping"]["base"]) + float(weights_cfg["shopping"]["weekend" if is_weekend else "weekday"])
    primary = weighted_choice(
        rng,
        ["routine", "shopping", "leisure", "family_outing", "travel"],
        [max(float(episode_cfg["routine_min_weight"]), 1.0 - travel - family - leisure - shopping), shopping, leisure, family, travel],
    )
    secondary = "none"
    if primary in {"family_outing", "travel"} and rng.random() < float(episode_cfg["secondary_shopping_probability"]):
        secondary = "shopping"
    elif primary == "routine" and rng.random() < float(episode_cfg["secondary_restaurant_probability"]):
        secondary = "restaurant_search"
    return primary, secondary


def nearby_candidates(
    pois: list[dict[str, Any]], category: str, lat: float, lon: float, scenario: str
) -> list[dict[str, Any]]:
    distances = [(haversine_km(lat, lon, poi["lat"], poi["lon"]), poi) for poi in pois if poi["category"] == category]
    distances.sort(key=lambda pair: pair[0])
    limits = CONFIG["choice"]["candidate_count"]
    limit = int(limits["opportunity_confounded"] if scenario == "opportunity_confounded" else limits["default"])
    return [poi | {"distance_km": distance} for distance, poi in distances[:limit]]


def category_for_episode(rng: random.Random, episode: str) -> str:
    options = CONFIG["episodes"]["category_weights"][episode]
    values = list(options)
    return weighted_choice(rng, values, [options[value] for value in values])


def choose_poi(
    rng: random.Random,
    user: dict[str, Any],
    candidates: list[dict[str, Any]],
    episode: str,
    scenario: str,
    decision_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = SCENARIO_SETTINGS[scenario]
    choice = CONFIG["choice"]
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        category = candidate["category"]
        pref = float(user.get(f"pref_{category}", choice["default_category_preference"]))
        family_fit = float(user["family_orientation"]) * (
            float(choice["family_fit_high"]) if category in set(choice["family_fit_categories"]) else float(choice["family_fit_low"])
        )
        episode_fit = (
            float(choice["travel_episode_fit_high"])
            if episode == "travel" and category in set(choice["travel_fit_categories"])
            else float(choice["episode_fit_default"])
        )
        preference_utility = float(choice["preference_weight"]) * pref + float(choice["quality_weight"]) * float(candidate["quality"]) + family_fit + episode_fit
        price_penalty = float(choice["price_weight"]) * float(user["price_sensitivity"]) * float(candidate["price"])
        distance_penalty = (
            settings["distance_scale"]
            * (float(choice["distance_intercept"]) + float(choice["distance_sensitivity_weight"]) * float(user["distance_sensitivity"]))
            * math.log1p(float(candidate["distance_km"]))
        )
        exposure_probability = clamp(
            float(choice["exposure_intercept"])
            + float(choice["exposure_popularity_weight"]) * float(candidate["popularity"])
            + float(choice["exposure_engagement_weight"]) * float(user["digital_engagement"])
        )
        exposed = rng.random() < exposure_probability
        exposure_utility = settings["exposure_scale"] * (
            float(choice["exposure_positive"]) if exposed else float(choice["exposure_negative"])
        )
        noise = rng.gammavariate(float(choice["noise_shape"]), float(choice["noise_scale"]))
        total = preference_utility - price_penalty - distance_penalty + exposure_utility + noise
        scored.append(
            {
                "decision_id": decision_id,
                "candidate_poi_id": candidate["poi_id"],
                "candidate_region_id": candidate["region_id"],
                "candidate_category": category,
                "distance_km": round(float(candidate["distance_km"]), 4),
                "price": round(float(candidate["price"]), 4),
                "quality": round(float(candidate["quality"]), 4),
                "exposed": int(exposed),
                "utility_preference": round(preference_utility, 5),
                "utility_price_penalty": round(price_penalty, 5),
                "utility_distance_penalty": round(distance_penalty, 5),
                "utility_exposure": round(exposure_utility, 5),
                "utility_total": round(total, 5),
                "is_chosen": 0,
            }
        )
    chosen_score = softmax_choice(rng, scored)
    chosen_score["is_chosen"] = 1
    chosen_poi = next(poi for poi in candidates if poi["poi_id"] == chosen_score["candidate_poi_id"])
    return chosen_poi, scored


def observed_location(rng: random.Random, lat: float, lon: float, accuracy_m: float) -> tuple[float, float]:
    return jitter_point(rng, lat, lon, abs(rng.gauss(0.0, accuracy_m / 1000.0)))


def event_row(
    rng: random.Random,
    user_id: str,
    timestamp: str,
    service: str,
    action: str,
    mode: str,
    category: str,
    object_id: str,
    region: Region,
    lat: float,
    lon: float,
    accuracy_m: float,
    source: str,
    session_id: str,
) -> dict[str, Any]:
    observed_lat, observed_lon = observed_location(rng, lat, lon, accuracy_m)
    mesh_1km, mesh_500m = japan_mesh(observed_lat, observed_lon)
    return {
        "user_id": user_id,
        "timestamp": timestamp,
        "service_id": service,
        "action_type": action,
        "observation_mode": mode,
        "object_id": object_id,
        "object_category": category,
        "region_id": region.region_id,
        "prefecture": region.prefecture,
        "latitude": round(observed_lat, 6),
        "longitude": round(observed_lon, 6),
        "geohash_5": geohash_encode(observed_lat, observed_lon, 5),
        "geohash_7": geohash_encode(observed_lat, observed_lon, 7),
        "mesh_1km": mesh_1km,
        "mesh_500m": mesh_500m,
        "location_accuracy_m": round(accuracy_m, 2),
        "source_table": source,
        "session_id": session_id,
    }


def write_csv_gz(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer schema for empty table: {path}")
        fieldnames = list(rows[0])
    # Fix the gzip timestamp and omit the source filename so identical seeds and
    # configurations produce byte-identical artifacts, not just identical rows.
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            try:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
            finally:
                # Let GzipFile own finalization so its end-of-stream trailer is
                # always written before the raw file closes.
                handle.detach()


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    streams = make_random_streams(args.seed, CONFIG["run"].get("random_streams"))
    world_rng = streams.generators["world"]
    user_rng = streams.generators["user_latents"]
    episode_rng = streams.generators["episodes"]
    choice_rng = streams.generators["choices"]
    observation_rng = streams.generators["observation"]
    # Persist the fully resolved values, including derived defaults, so a run's
    # stochastic lineage can be reconstructed from config.resolved.yaml alone.
    CONFIG["run"]["seed"] = streams.root_seed
    CONFIG["run"]["random_streams"] = dict(streams.seeds)
    output = Path(args.output).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output}. Pass --overwrite to replace it.")
        if output == Path("/") or len(output.parts) < 3:
            raise ValueError(f"Refusing to replace broad path: {output}")
        shutil.rmtree(output)
    observed_dir, truth_dir = output / "observed", output / "truth"
    observed_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)

    regions_by_id = {region.region_id: region for region in REGIONS}
    pois = make_world(world_rng)
    users_observed: list[dict[str, Any]] = []
    user_latents: list[dict[str, Any]] = []
    observation_process: list[dict[str, Any]] = []
    observed_events: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    candidate_sets: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    start_day = date.fromisoformat(args.start_date)
    schedule = CONFIG["episodes"]["schedule_hours"]
    event_cfg = CONFIG["events"]
    spatial = CONFIG["world"]["spatial"]
    for user_index in range(1, args.users + 1):
        observed_user, latent_user = create_user(user_rng, user_index, args.full_kanto)
        users_observed.append(observed_user)
        user_latents.append(latent_user)
        obs_records = make_observation_process(observation_rng, latent_user, args.scenario)
        observation_process.extend(obs_records)
        obs_by_service = {row["source_service"]: row for row in obs_records}
        home = regions_by_id[str(latent_user["home_region_id"])]
        work = regions_by_id[str(latent_user["work_region_id"])]
        home_lat, home_lon = float(latent_user["home_latitude"]), float(latent_user["home_longitude"])
        work_lat, work_lon = float(latent_user["work_latitude"]), float(latent_user["work_longitude"])

        for day_offset in range(args.days):
            current_day = start_day + timedelta(days=day_offset)
            weekend = current_day.weekday() >= 5
            primary, secondary = select_episode(episode_rng, latent_user, current_day, weekend)
            episode_id = stable_identifier("episode", latent_user["user_id"], current_day.isoformat())
            session_id = stable_identifier("session", latent_user["user_id"], current_day.isoformat())
            active_region = home
            active_lat, active_lon = home_lat, home_lon
            if primary == "travel":
                travel_regions = [region for region in REGIONS if region.region_id != home.region_id]
                active_region = weighted_choice(
                    episode_rng,
                    travel_regions,
                    [
                        (float(CONFIG["episodes"]["travel_region_bonus"]) if region.region_id in set(CONFIG["episodes"]["travel_bonus_regions"]) else 1.0)
                        * math.exp(-haversine_km(home_lat, home_lon, region.lat, region.lon) / float(CONFIG["episodes"]["travel_distance_decay_km"]))
                        for region in travel_regions
                    ],
                )
                active_lat, active_lon = gaussian_point(
                    episode_rng,
                    active_region.lat,
                    active_region.lon,
                    float(spatial["destination_spread_km"]),
                    float(spatial["east_west_anisotropy"]),
                )
            elif primary in {"leisure", "family_outing"} and episode_rng.random() < float(CONFIG["episodes"]["nearby_region_switch_probability"]):
                near_regions = sorted(REGIONS, key=lambda region: haversine_km(home_lat, home_lon, region.lat, region.lon))[
                    : int(CONFIG["episodes"]["nearby_region_pool_size"])
                ]
                active_region = episode_rng.choice(near_regions)
                active_lat, active_lon = gaussian_point(
                    episode_rng,
                    active_region.lat,
                    active_region.lon,
                    float(spatial["destination_spread_km"]),
                    float(spatial["east_west_anisotropy"]),
                )

            episodes.append(
                {
                    "episode_id": episode_id,
                    "user_id": latent_user["user_id"],
                    "start_time": iso_at(current_day, float(schedule["episode_start"])),
                    "end_time": iso_at(current_day, float(schedule["episode_end"])),
                    "primary_intent": primary,
                    "secondary_intent": secondary,
                    "origin_region_id": home.region_id,
                    "destination_region_id": active_region.region_id,
                }
            )

            stops: list[tuple[float, str, Region, float, float]] = [
                (float(schedule["morning_home"]), "home", home, home_lat, home_lon)
            ]
            if not weekend and primary != "travel":
                stops += [
                    (float(schedule["work_arrival"]), "commute", work, work_lat, work_lon),
                    (float(schedule["midday_work"]), "work", work, work_lat, work_lon),
                    (float(schedule["home_commute"]), "commute", home, home_lat, home_lon),
                ]
            elif primary == "travel":
                stops += [(float(schedule["travel_arrival"]), "travel", active_region, active_lat, active_lon)]

            category = category_for_episode(episode_rng, primary)
            if primary == "travel":
                origin, origin_lat, origin_lon = active_region, active_lat, active_lon
            elif not weekend and primary == "routine":
                origin, origin_lat, origin_lon = work, work_lat, work_lon
            else:
                origin, origin_lat, origin_lon = home, home_lat, home_lon
            candidates = nearby_candidates(pois, category, origin_lat, origin_lon, args.scenario)
            if candidates:
                decision_id = stable_identifier("decision", episode_id, "primary_poi_choice")
                chosen, scored = choose_poi(choice_rng, latent_user, candidates, primary, args.scenario, decision_id)
                candidate_sets.extend(scored)
                choice_hour = (
                    float(schedule["routine_choice"])
                    if not weekend and primary == "routine"
                    else float(schedule["other_choice_start"]) + choice_rng.random() * float(schedule["other_choice_span"])
                )
                chosen_region = regions_by_id[chosen["region_id"]]
                stops.append((choice_hour, "poi_visit", chosen_region, chosen["lat"], chosen["lon"]))
                choices.append(
                    {
                        "decision_id": decision_id,
                        "user_id": latent_user["user_id"],
                        "timestamp": iso_at(current_day, choice_hour),
                        "episode_id": episode_id,
                        "choice_context": primary,
                        "origin_region_id": origin.region_id,
                        "chosen_poi_id": chosen["poi_id"],
                        "chosen_category": category,
                        "candidate_count": len(scored),
                    }
                )
                local_obs = obs_by_service["local_commerce"]
                if local_obs["service_adopted"] and observation_rng.random() < float(local_obs["record_probability"]):
                    local_actions = event_cfg["local_commerce"]["action_weights"]
                    action_names = list(local_actions)
                    action = weighted_choice(observation_rng, action_names, [local_actions[name] for name in action_names])
                    observed_events.append(
                        event_row(
                            observation_rng,
                            latent_user["user_id"],
                            iso_at(current_day, choice_hour + float(event_cfg["local_commerce"]["event_delay_hours"])),
                            "local_commerce",
                            action,
                            "transaction" if action == "payment" else "user_triggered",
                            category,
                            chosen["poi_id"],
                            chosen_region,
                            chosen["lat"],
                            chosen["lon"],
                            float(event_cfg["local_commerce"]["location_accuracy_m"]),
                            "local_activity_log",
                            session_id,
                        )
                    )

            stops.append((float(schedule["evening_home"]), "home", home, home_lat, home_lon))
            stops.sort(key=lambda item: item[0])
            location_obs = obs_by_service["location"]
            activity_occurrences: Counter[str] = Counter()
            for stop_index, (hour, activity, region, lat, lon) in enumerate(stops):
                activity_occurrences[activity] += 1
                true_lat, true_lon = jitter_point(
                    episode_rng,
                    lat,
                    lon,
                    float(spatial["routine_stop_jitter_km"])
                    if activity in {"home", "work"}
                    else float(spatial["activity_stop_jitter_km"]),
                )
                trajectories.append(
                    {
                        "trajectory_id": stable_identifier("trajectory", episode_id, activity, activity_occurrences[activity], iso_at(current_day, hour)),
                        "user_id": latent_user["user_id"],
                        "timestamp": iso_at(current_day, hour),
                        "episode_id": episode_id,
                        "activity": activity,
                        "true_region_id": region.region_id,
                        "true_latitude": round(true_lat, 6),
                        "true_longitude": round(true_lon, 6),
                    }
                )
                if location_obs["service_adopted"] and observation_rng.random() < float(location_obs["record_probability"]):
                    accuracy = float(location_obs["gps_sd_m"])
                    observed_events.append(
                        event_row(
                            observation_rng,
                            latent_user["user_id"],
                            iso_at(
                                current_day,
                                hour
                                + observation_rng.uniform(
                                    -float(CONFIG["observation"]["passive_time_jitter_hours"]),
                                    float(CONFIG["observation"]["passive_time_jitter_hours"]),
                                ),
                            ),
                            "location",
                            "location_ping",
                            "passive",
                            "unknown",
                            "",
                            region,
                            true_lat,
                            true_lon,
                            accuracy,
                            "passive_location_log",
                            session_id,
                        )
                    )

            ecommerce_obs = obs_by_service["ecommerce"]
            ecommerce_probability = (
                float(event_cfg["ecommerce"]["event_probability_intercept"])
                + float(event_cfg["ecommerce"]["engagement_coefficient"]) * float(latent_user["digital_engagement"])
                + (float(event_cfg["ecommerce"]["episode_bonus"]) if primary in set(event_cfg["ecommerce"]["bonus_episodes"]) else 0.0)
            )
            if ecommerce_obs["service_adopted"] and observation_rng.random() < ecommerce_probability * float(ecommerce_obs["record_probability"]):
                product_weights = dict(event_cfg["ecommerce"]["product_category_weights"])
                if primary == "travel":
                    product_weights["travel_goods"] = float(event_cfg["ecommerce"]["travel_goods_weight_during_travel"])
                product_names = list(product_weights)
                product_category = weighted_choice(observation_rng, product_names, [product_weights[name] for name in product_names])
                ecommerce_actions = event_cfg["ecommerce"]["action_weights"]
                ecommerce_action_names = list(ecommerce_actions)
                action = weighted_choice(observation_rng, ecommerce_action_names, [ecommerce_actions[name] for name in ecommerce_action_names])
                observed_events.append(
                    event_row(
                        observation_rng, latent_user["user_id"], iso_at(current_day, float(schedule["ecommerce_start"]) + observation_rng.random() * float(schedule["ecommerce_span"])), "ecommerce", action,
                        "transaction" if action == "purchase" else "user_triggered", product_category,
                        f"product_{observation_rng.randrange(1, int(event_cfg['ecommerce']['product_catalog_size']) + 1):04d}", home, home_lat, home_lon,
                        float(event_cfg["ecommerce"]["location_accuracy_m"]), "ecommerce_event_log", session_id,
                    )
                )

            travel_obs = obs_by_service["travel"]
            if primary == "travel" and travel_obs["service_adopted"] and observation_rng.random() < float(travel_obs["record_probability"]):
                for action, hour in event_cfg["travel"]["action_hours"].items():
                    if action == "reservation" and observation_rng.random() > float(event_cfg["travel"]["reservation_probability"]):
                        continue
                    observed_events.append(
                        event_row(
                            observation_rng, latent_user["user_id"], iso_at(current_day, hour), "travel", action,
                            "transaction" if action == "reservation" else "user_triggered", "destination",
                            active_region.region_id, home, home_lat, home_lon,
                            float(event_cfg["travel"]["location_accuracy_m"]), "travel_event_log", session_id,
                        )
                    )

    observed_events.sort(key=lambda row: (row["user_id"], row["timestamp"], row["service_id"]))
    tables = {
        observed_dir / OBSERVED_FILES["events"]: observed_events,
        observed_dir / OBSERVED_FILES["users"]: users_observed,
        truth_dir / TRUTH_FILES["user_latents"]: user_latents,
        truth_dir / TRUTH_FILES["episodes"]: episodes,
        truth_dir / TRUTH_FILES["candidate_sets"]: candidate_sets,
        truth_dir / TRUTH_FILES["choices"]: choices,
        truth_dir / TRUTH_FILES["trajectories"]: trajectories,
        truth_dir / TRUTH_FILES["observation_process"]: observation_process,
    }
    for path, rows in tables.items():
        write_csv_gz(path, rows)

    report = validate_dataset(
        users_observed, user_latents, observed_events, episodes, candidate_sets, choices, trajectories, observation_process
    )
    resolved_yaml = yaml.safe_dump(CONFIG, sort_keys=False, allow_unicode=True)
    config_hash = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    (output / "config.resolved.yaml").write_text(resolved_yaml, encoding="utf-8")
    identity_entities = {
        "users": [row["user_id"] for row in users_observed],
        "regions": [region.region_id for region in REGIONS],
        "pois": [row["poi_id"] for row in pois],
        "episodes": [row["episode_id"] for row in episodes],
        "choices": [row["decision_id"] for row in choices],
        "trajectories": [row["trajectory_id"] for row in trajectories],
    }
    identity_manifest = {
        "schema_version": SIMULATION_IDENTITY_MANIFEST_SCHEMA,
        "identity_generation_version": IDENTITY_GENERATION_VERSION,
        "hash_algorithm": SIMULATION_IDENTITY_HASH_ALGORITHM,
        "random_streams": {"algorithm": RANDOM_STREAM_ALGORITHM, "root_seed": streams.root_seed, "seeds": streams.seeds},
        "entities": {
            name: {"count": len(values), "identity_sha256": identity_set_hash(values)}
            for name, values in identity_entities.items()
        },
    }
    manifest = {
        "simulator_version": SIMULATOR_VERSION,
        "dataset_contract": {
            "name": DATASET_CONTRACT_NAME,
            "version": DATASET_CONTRACT_VERSION,
        },
        "config_schema_version": CONFIG.get("config_version", 1),
        "config_source": str(Path(args.config).resolve()),
        "config_sha256": config_hash,
        "seed": args.seed,
        "random_streams": {
            "algorithm": RANDOM_STREAM_ALGORITHM,
            "root_seed": streams.root_seed,
            "seeds": streams.seeds,
        },
        "identity": identity_manifest,
        "start_date": args.start_date,
        "days": args.days,
        "users": args.users,
        "scenario": args.scenario,
        "intervention": CONFIG["run"].get("intervention"),
        "full_kanto": args.full_kanto,
        "coordinate_system": "WGS84; continuous synthetic locations sampled from overlapping Kanto hub catchments",
        "geographic_splits": {"development": "Tokyo, Kanagawa, Saitama, Chiba", "holdout": "Ibaraki, Tochigi, Gunma"},
        "table_rows": {str(path.relative_to(output)): len(rows) for path, rows in tables.items()},
        "validation": report,
        "training_boundary": "Training code may read observed/ only. truth/ is reserved for evaluation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest


def validate_dataset(
    users: list[dict[str, Any]],
    latents: list[dict[str, Any]],
    events: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    observation: list[dict[str, Any]],
) -> dict[str, Any]:
    user_ids = {row["user_id"] for row in users}
    latent_ids = {row["user_id"] for row in latents}
    candidate_counts: Counter[str] = Counter()
    chosen_counts: Counter[str] = Counter()
    for row in candidates:
        candidate_counts[row["decision_id"]] += 1
        chosen_counts[row["decision_id"]] += int(row["is_chosen"])
    forbidden_fragments = ("latent", "utility", "episode_id", "true_")
    event_columns = set(events[0]) if events else set()
    services_per_user: defaultdict[str, set[str]] = defaultdict(set)
    event_counts: Counter[str] = Counter()
    active_dates: defaultdict[str, set[str]] = defaultdict(set)
    mode_counts: Counter[str] = Counter()
    for row in events:
        services_per_user[row["user_id"]].add(row["service_id"])
        event_counts[row["user_id"]] += 1
        active_dates[row["user_id"]].add(str(row["timestamp"])[:10])
        mode_counts[row["observation_mode"]] += 1
    checks = {
        "unique_observed_users": len(user_ids) == len(users),
        "one_latent_row_per_user": user_ids == latent_ids and len(latents) == len(users),
        "event_users_resolve": all(row["user_id"] in user_ids for row in events),
        "episode_users_resolve": all(row["user_id"] in user_ids for row in episodes),
        "trajectory_users_resolve": all(row["user_id"] in user_ids for row in trajectories),
        "observation_users_resolve": all(row["user_id"] in user_ids for row in observation),
        "one_chosen_candidate_per_decision": bool(chosen_counts) and all(count == 1 for count in chosen_counts.values()),
        "candidate_and_choice_ids_match": set(candidate_counts) == {row["decision_id"] for row in choices},
        "no_truth_columns_in_observed_events": not any(fragment in column for column in event_columns for fragment in forbidden_fragments),
        "passive_and_active_present": mode_counts["passive"] > 0 and sum(value for key, value in mode_counts.items() if key != "passive") > 0,
        "multiple_services_present": len({row["service_id"] for row in events}) >= 3,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Dataset validation failed: {failures}")
    overlap = Counter(len(services_per_user[user_id]) for user_id in user_ids)

    def p(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * fraction)]

    per_user_events = [event_counts[user_id] for user_id in user_ids]
    per_user_days = [len(active_dates[user_id]) for user_id in user_ids]
    return {
        "status": "passed",
        "checks": checks,
        "event_mode_counts": dict(mode_counts),
        "service_counts": dict(Counter(row["service_id"] for row in events)),
        "users_by_observed_service_count": {str(key): value for key, value in sorted(overlap.items())},
        "events_per_user": {"p10": p(per_user_events, 0.10), "p50": p(per_user_events, 0.50), "p90": p(per_user_events, 0.90)},
        "active_days_per_user": {"p10": p(per_user_days, 0.10), "p50": p(per_user_days, 0.50), "p90": p(per_user_days, 0.90)},
        "event_date_min": min((row["timestamp"] for row in events), default=None),
        "event_date_max": max((row["timestamp"] for row in events), default=None),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SIMULATOR_VERSION}")
    parser.add_argument(
        "--config",
        default="configs/simulation/kanto_v1.yaml",
        help="YAML configuration file",
    )
    parser.add_argument("--users", type=int, help="Override run.users")
    parser.add_argument("--days", type=int, help="Override run.days")
    parser.add_argument("--start-date", help="Override run.start_date")
    parser.add_argument("--seed", type=int, help="Override run.seed")
    parser.add_argument("--scenario", help="Override run.scenario")
    parser.add_argument(
        "--full-kanto",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override run.full_kanto",
    )
    parser.add_argument("--output", help="Override run.output")
    parser.add_argument("--overwrite", action="store_true", help="Replace the exact output directory if it exists")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    run = config["run"]
    for name in ("users", "days", "start_date", "seed", "scenario", "output", "full_kanto"):
        if getattr(args, name) is None:
            setattr(args, name, run[name])
        run[name] = getattr(args, name)
    activate_config(config)
    if args.scenario not in SCENARIO_SETTINGS:
        raise SystemExit(f"Unknown scenario {args.scenario!r}; choose one of {sorted(SCENARIO_SETTINGS)}")
    if args.users < 10 or args.days < 2:
        raise SystemExit("Use at least 10 users and 2 days so validation is meaningful.")
    manifest = simulate(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
