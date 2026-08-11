# Agent handoff

## Project purpose

GeoEmbeddings is a controlled research environment for learning user
representations from cross-service event histories containing mobility,
local-commerce, e-commerce, and travel signals. The immediate business example
is recommending Hakone points of interest after a Tokyo-based user arrives,
while retaining stable preferences and adapting to the trip context.

The repository deliberately contains two logically separate systems:

- a semi-synthetic Kanto data generator with protected latent truth;
- an observed-data-only embedding pipeline plus a protected evaluator.

They share `geoembeddings-dataset/1.0`, a versioned file contract. They should
remain separate modules even though one CLI orchestrates them.

## Scientific decomposition

The final representation should expose three distinct but related components:

| Component | Meaning | Should change when | Should resist |
|---|---|---|---|
| `p_u` | Persistent preferences and traits | sustained evidence accumulates | one trip, exposure changes, GPS noise |
| `r_u,t` | Recurring routine and periodic state | weekday/weekend, time, repeated commute changes | unrelated single events |
| `c_u,t` | Current episode and intent | arrival, search, check-in, local activity | pressure to remain globally stable |

The existing `SingleVectorEncoder` compresses all three pressures into one
128-dimensional vector. It is intentionally a failure-revealing baseline.

## Repository ownership map

| Area | Main files | Responsibility |
|---|---|---|
| Dataset contract | `contract.py`, `layout.py`, `schema.py` | filenames, roots, schemas, leakage checks |
| Simulator | `simulator.py`, simulation YAML | world, users, episodes, choices, observation process |
| Simulator validation | `simulation_validation.py`, scripts/notebook | structural and behavioral checks |
| Preparation | `prepare.py`, `data.py` | splits, train-only vocab/stats, windows, tensors |
| Baseline | `baseline.py` | normalized histograms and continuous moments |
| Learned model | `model.py`, `training.py`, embedding YAML | event encoder, GRU, objectives, checkpoint |
| Export | `export.py` | user embeddings at train/validation/test cutoffs |
| Evaluation | `evaluation.py`, `comparison.py` | protected probes and fair frozen comparisons |
| Orchestration | `cli.py` | public commands and canonical path resolution |

## Current implemented data-generating process

The generator creates overlapping Gaussian catchments around Kanto region
anchors, synthetic POIs, user demographics and persistent latents, daily
episodes, utility-based POI choices, true trajectories, service adoption,
recording/dropout, and noisy observed events. It supports controlled scenarios:

- `clean`
- `mixed`
- `opportunity_confounded`
- `exposure_confounded`
- `observation_biased`

Important limitations:

- coordinates and POIs are synthetic;
- movement is stop-based and uses straight-line distance, not a transport graph;
- POI opening hours, capacity, travel times, and request-time availability are
  not public observed data;
- scenario constants are experimental hypotheses, not calibrated population
  estimates;
- the current simulator does not emit a complete recommendation interaction log.

## Current implemented model

Each history event contains categorical embeddings for service, action,
observation mode, category, region, geohash-5, and geohash-7, plus standardized
coordinates, elapsed time, cyclic hour/day, and location accuracy. `object_id`
is disabled by default.

The event representations pass through a padded GRU. The final valid timestep is
selected with a floating mask, avoiding Apple MPS packed-sequence and
gather/scatter failures. The model predicts the next service, action, category,
region, geohash-5, and geohash-7. A consistency term aligns embeddings from the
early and late parts of the same window.

Default objective:

\[
0.2L_{service}+0.4L_{action}+1.0L_{category}+0.75L_{region}
+0.5L_{gh5}+0.25L_{gh7}+0.1L_{consistency}.
\]

## Current evidence

The code has run successfully on an Apple M5 Pro after the v0.3.x MPS and
categorical-schema fixes. At epoch 8 of the 500-user learned run, the reported
training/validation results were:

| Target | Train accuracy | Validation accuracy |
|---|---:|---:|
| Service | 0.8612 | 0.7997 |
| Action | 0.8242 | 0.7110 |
| Category | 0.8085 | 0.6727 |
| Region | 0.7353 | 0.7252 |
| Geohash-5 | 0.5381 | 0.4674 |
| Geohash-7 | 0.2123 | 0.0881 |

Training loss was `3.3100`; validation loss was `5.5803`. These results show
predictive signal and a meaningful generalization gap. They do not establish
that the embedding satisfies the persistent/context requirements.

The earlier statistical smoke baseline achieved train-to-test cosine stability
near `0.986` but mean persistent-trait probe R2 near `-0.093`. This is the
canonical warning that stability without retained information is not success.

The `compare` command can measure both representations fairly and merge matched
episode, robustness, and spatial-transfer reports when those supplemental
commands have been run. The historical 500-user reports are indexed, but their
bytes are lost/unverifiable; do not interpret that reference as evidence that
either representation is stronger. A replacement must use a new artifact
identity and lineage.

## Known bug history to preserve in tests

1. Packed variable-length GRU execution caused Apple MPS command-buffer errors.
2. Replacing packing with integer `gather` caused an MPS backward `scatter`
   failure.
3. JSON serialization sorted vocabulary keys, while event tensors used
   configured order. The model interpreted category IDs as observation-mode
   IDs and produced out-of-range accelerator indices.

The current implementation uses padded GRU execution, a floating final-state
mask, CPU length metadata, explicit categorical-field order, CPU-side batch
validation, and regression tests. Do not weaken these protections.

## What the current comparison proves and does not prove

It can compare:

- persistent trait and category-preference probe R2;
- preference signal beyond home/work geography and event volume;
- same-user temporal stability and different-user separation;
- temporal identity retrieval and effective rank;
- activity-volume dependence;
- common frozen future-event probes.

It can also execute deterministic sensitivity and transfer diagnostics:

- dense, episode-aligned coherence, response, drift/recovery, and intent probes;
- event removal, GPS perturbation, timestamp jitter, leave-one-service-out, and
  recent-history truncation views;
- distance retrieval, geohash-boundary pairs, held-out-region coverage, and
  seen/unseen geohash slices.

It cannot yet establish:

- persistent/routine/context separation;
- matched counterfactual exposure invariance;
- causal invariance or calibration to real GPS/timestamp/missingness processes;
- unseen-POI or candidate-aware geographic transfer;
- uncertainty, sustained preference change, privacy, or efficiency;
- candidate-aware new-context recommendation.

## Near-term research decision

Do not choose the next model from next-event accuracy alone. Run the implemented
dense/episode, deterministic robustness, and spatial-transfer surfaces for both
representations and inspect their matched `compare` axes. Then compare:

1. statistical baseline;
2. current single-vector GRU;
3. single-vector ablations;
4. two-way persistent/context factorization;
5. three-way persistent/routine/context factorization.

The three-way model earns its additional complexity only if the evaluator can
show that routine is neither merely persistent identity nor current episode.

## Recommendation data contract target

The simulator should eventually publish these observed tables:

```text
observed/
├── poi_catalog.csv.gz
├── recommendation_requests.csv.gz
├── impressions.csv.gz
└── interactions.csv.gz
```

Public candidate features may include category, coordinates, travel time,
opening hours, price level, family suitability, indoor/outdoor state, local
popularity, and request-time availability. True utility, latent preference,
true episode, unshown alternatives, and counterfactual choice probabilities
remain under `truth/`.

## Handoff success criterion

A remote agent should be able to:

1. reproduce a small run;
2. locate every artifact from its two root arguments;
3. select a task tied to requirement IDs;
4. make a tested change without crossing the information boundary;
5. produce a matched comparison and an evidence-based recommendation for the
   next iteration.
