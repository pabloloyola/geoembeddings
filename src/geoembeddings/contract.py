"""Shared, versioned file contract between simulation and embedding code."""

from __future__ import annotations


DATASET_CONTRACT_NAME = "geoembeddings-dataset"
DATASET_CONTRACT_VERSION = "1.0"

OBSERVED_FILES = {
    "users": "users_observed.csv.gz",
    "events": "observed_events.csv.gz",
}

TRUTH_FILES = {
    "user_latents": "user_latents.csv.gz",
    "episodes": "episodes_truth.csv.gz",
    "candidate_sets": "candidate_sets.csv.gz",
    "choices": "choices_truth.csv.gz",
    "trajectories": "trajectories_truth.csv.gz",
    "observation_process": "observation_process.csv.gz",
}

