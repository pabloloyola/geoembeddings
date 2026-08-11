# Experiment protocol

## Purpose

This protocol prevents changes in data, splits, probes, or candidate sets from
being mistaken for model improvements.

## Experimental unit

A dataset run is immutable and identified by:

- resolved simulation configuration;
- dataset contract version;
- simulator version;
- random seed;
- hashes of observed source files.

An experiment is one representation/modeling attempt on exactly one dataset run
and is identified by:

- resolved embedding configuration;
- preparation metadata and cutoffs;
- code version/commit when available;
- model seed;
- checkpoint selection rule.

Use descriptive names:

```text
runs/kanto_mixed_seed20260803_u500_d14/
experiments/sv_gru_seed20260806_on_kanto_mixed_seed20260803/
```

Do not reuse a directory for materially different settings.

## Reproduction sequence

```bash
uv sync --locked --extra dev
uv run pytest

uv run geoembed simulate \
  --config configs/simulation/kanto_v1.yaml \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14

uv run geoembed validate \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14

uv run geoembed prepare \
  --config configs/embedding/single_vector.yaml \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed train \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed export \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed baseline \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed evaluate --kind learned \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed evaluate --kind baseline \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed compare \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference
```

## Comparison levels

### Level A: plumbing smoke test

- 50 users, 7 days, one seed.
- Purpose: schemas, paths, MPS/CPU execution, artifacts, and tests.
- Never use it to select a scientific model.

### Level B: development comparison

- 500 users, 14 days, fixed reference seed.
- Compare statistical baseline, current GRU, and one change at a time.
- Use identical preparation and frozen probe splits.

### Level C: robustness and counterfactual suite

- At least three generator/model seeds.
- Matched latent users and world across scenario interventions.
- Report mean, standard deviation, and per-seed values.
- Evaluate all requirements targeted by the change plus regression axes.

### Level D: recommendation suite

- Freeze request/candidate records for model comparisons.
- Separate seen-region, held-out-region, seen-POI, unseen-POI, and first-arrival
  slices.
- Compare frozen encoder plus small ranker and end-to-end fine-tuning separately.

## Matched-seed simulator interventions

Scenario comparison is meaningful only when unchanged latent objects retain
stable identities. When extending the simulator, use independent named random
streams for at least:

- world/POI creation;
- user latent creation;
- episode generation;
- choice noise;
- observation/dropout/GPS noise.

Changing only exposure or observation parameters should not silently regenerate
user preferences, POI qualities, or episodes. Record stream seeds in the
manifest. Until this is implemented, describe current matched-seed scenario
comparisons as approximate.

## Model selection

- Select checkpoints using validation data only.
- Do not tune on protected test probes.
- A next-event validation loss can select within one model family, but the final
  research decision must use the held-out requirement suite.
- When objective weights change, rerun training and all downstream exports and
  evaluations. Reuse the same `prepared/` artifacts only when input fields and
  split/preprocessing settings are unchanged.

## Required baselines

Keep at least:

1. popularity/majority targets for predictive tasks;
2. the statistical history vector;
3. the current single-vector GRU;
4. a last-N-events or recent-only representation for context tasks;
5. persistent-only and context-only ablations for factorized models;
6. nearest-POI, popularity, and category-preference rankers for recommendation.

## Ablation discipline

Change one conceptual factor at a time. For a factorized model, minimum
ablations are:

- no persistent loss;
- no context loss;
- no routine branch;
- no cross-window consistency;
- no event dropout;
- no fine geohash target;
- shared versus separate projection heads;
- with and without object ID;
- full history versus recent window.

## Report contents

Every result report should include:

- command and date;
- code version/commit;
- run and experiment directories;
- simulator and model seeds;
- source hashes and temporal cutoffs;
- device and runtime versions;
- train/validation/test counts and known-label coverage;
- full configuration or path to the resolved copy;
- metric values by requirement and baseline deltas;
- runtime and peak memory when available;
- interpretation, limitations, and next decision.

## Rerun dependency table

| Change | Simulate | Validate | Prepare | Train | Export | Evaluate/compare |
|---|---:|---:|---:|---:|---:|---:|
| Simulator YAML or generator logic | Yes | Yes | Yes | Yes | Yes | Yes |
| Observed schema/contract | Yes | Yes | Yes | Yes | Yes | Yes |
| Split, fields, vocabulary, normalization | No | No | Yes | Yes | Yes | Yes |
| Model architecture or objective | No | No | Usually no | Yes | Yes | Yes |
| Training optimizer/epochs/seed | No | No | No | Yes | Yes | Yes |
| Export format/cutoffs | No | No | No | No if checkpoint compatible | Yes | Yes |
| Evaluator/probe only | No | No | No | No | No unless new export needed | Yes |
| Documentation only | No | No | No | No | No | No |

## Failure handling

- If MPS fails, first run the same batch/test on CPU and inspect the first
  asynchronous operation, not merely the synchronization line in the traceback.
- Validate categorical IDs and target ranges on CPU before accelerator work.
- If a comparison rejects source hashes or cutoffs, do not bypass the guard;
  reproduce both representations under one preparation contract.
- If a requirement cannot be measured from existing exports, extend exports or
  mark it unavailable. Do not infer it from unrelated metrics.

