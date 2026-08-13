# Objectives and evaluation contract

## Representation we ultimately want

The target model should return three related outputs:

\[
z_{u,t} = (p_u, r_{u,t}, c_{u,t}),
\]

where \(p_u\) captures persistent preferences, \(r_{u,t}\) recurring routines,
and \(c_{u,t}\) current episode and intent. v0 emits only one vector \(z_{u,t}\)
to establish what fails when these pressures share one capacity bottleneck.

## Prioritized requirements

| ID | Requirement | Training signal planned | Evaluation test | v0 status |
|---|---|---|---|---|
| R1 | Persistent/context separation | Cross-window agreement plus next-event prediction; later factorized objectives | Latent probes, cross-cutoff drift, episode responsiveness | Partial |
| R2 | Multi-scale spatial fidelity | Geohash-5 and geohash-7 prediction; later coordinate/distance loss | Hierarchy accuracy, metric retrieval, geohash-boundary pairs | Partial |
| R3 | Multi-scale temporal fidelity | Elapsed-time and cyclic calendar features; later duration/periodicity tasks | Next event, routine and periodicity probes | Partial |
| R4 | Episode coherence | Later episode-aware contrastive/predictive tasks | Episode classification and within-episode retrieval | Partial: executable episode diagnostics; factorization pending |
| R5 | Preference/opportunity separation | Later exposure-aware or inverse-propensity objective | Same preference under changed opportunity/exposure | Pending |
| R6 | Cross-service alignment | Next-service transitions; later paired cross-service views | Hold out one service and predict it from the others | Partial: deterministic service-removal executable |
| R7 | Noise/sparsity robustness | Event dropout | Controlled event removal, GPS perturbation, missing-service tests | Partial: deterministic corruption views executable |
| R8 | Geographic/temporal transfer | Compositional event fields and no default POI ID | Held-out regions and later periods | Partial: held-out-region/geohash slices executable |
| R9 | New-context recommendation | Candidate-aware ranking objective | Tokyo-history → Hakone-candidate ranking, regret, NDCG | Partial/executable: T3.4--T3.7 ranking surfaces are implemented; reference-scale and external evidence remain absent |

## Why the baseline objective is useful

The baseline combines short-term predictive and long-term consistency pressures:

\[
\mathcal L =
\sum_{h \in \mathcal H}\lambda_h\,
\operatorname{CE}(h(z_{u,t}), y_{h,t+1})
+ \lambda_{\text{cons}}
\left[1-\cos(z^{\text{early}}_{u,t},z^{\text{late}}_{u,t})\right].
\]

The predictive heads \(\mathcal H\) cover service, action, category, region, and
two geographic levels. The consistency term compares earlier and later halves
of the same observed history window. Event dropout supplies a second robustness
perturbation.

This objective does not prove disentanglement. It creates measurable competing
forces:

- Increasing next-event weights should improve responsiveness but can make the
  vector drift with temporary episodes.
- Increasing consistency should improve stability but can erase meaningful
  context or collapse user differences.
- Increasing geohash-7 weight may improve fine local accuracy but reward
  memorization and reduce transfer.

That is precisely why v0 should be run as an ablation surface before the
factorized architecture.

## Leakage controls

1. `prepare` and `train` accept only a directory literally named `observed`.
2. Truth-like column names are rejected in observed inputs.
3. Vocabularies and continuous normalization statistics are fit using events
   at or before the training cutoff.
4. A prediction target at time \(t\) receives only events with earlier indices
   in the same user's sorted history.
5. Validation and test targets are later in global time; their histories may
   contain earlier events, as they would in online inference.
6. `truth/` is opened only by the final evaluator.
7. Latent probes split by user, preventing a probe from seeing the same user's
   label in train and test.

## Implemented simulator outputs for recommendation

Dataset contract 2.0 provides the observable recommendation tables required by
the business use case:

```text
observed/
├── poi_catalog.csv.gz
├── recommendation_requests.csv.gz
├── impressions.csv.gz
└── interactions.csv.gz
```

Each request identifies the user and timestamp; public candidate records contain
availability and features known at ranking time; impressions record what was
shown; and interactions record the response. Candidate utility, unobserved
preference, true choice probabilities, and true episodes remain evaluator-only.

T3.4 provides recommendation controls, T3.5 provides frozen-embedding candidate
ranking, T3.6 provides observed-only exposure-aware training plus a separately
authenticated protected paired evaluator, and T3.7 provides observed-only
seen/unseen region and POI plus early/late transfer slices. Protected utility
regret and probability recovery are available only after pair-integrity and
ranking-identity authentication; the observed-only T3.7 surface cannot report
utility regret. These synthetic results do not establish real-world causality
or external validity, and no reference-scale recommendation result is archived.

## Implemented development sequence

1. Run v0 and establish next-event, latent-probe, and stability baselines.
2. Add controlled corruption and service/region holdout evaluators without
   changing the model.
3. Add the factorized persistent/context encoder behind the same CLI.
4. Add routine representation only after R1 tests distinguish persistent trait,
   repeated routine, and current episode.
5. Extend the simulator's observable recommendation contract (completed in
   dataset contract 2.0).
6. Add candidate-aware ranking and new-context recommendation evaluation
   (T3.4--T3.7 implemented with the limitations above).

## Frozen embedding comparison

The `compare` command evaluates the statistical and learned representations with
the same users, temporal cutoffs, held-out-user split, and ridge penalty. It does
not compare the learned model's task heads against a baseline that has no heads.
Instead, it fits common frozen probes and reports separate axes:

- Persistent latent-trait and category-preference probe R2.
- Incremental preference R2 beyond home/work region and observed event count.
- Same-user stability together with different-user separation.
- Train-to-test user retrieval and centered effective rank, which detect collapse.
- Dependence on observation quantity.
- Held-out-user future-event probes for service, action, category, region, and
  geohash levels.

No aggregate winner is produced. Supplemental matched axes are added after
`export-dense`/`evaluate --episodes`, `robustness`, and `evaluate --transfer`
have produced both baseline and learned reports. Persistent/context
disentanglement, externally valid counterfactual exposure invariance,
real-noise calibration, reference-scale recommendation transfer, and
selection-dependent calibrated uncertainty remain explicit evidence gaps.
Sustained-change adaptation is executable only as the T4.2 authenticated
diagnostic-control audit, and privacy is executable only as the T4.3a
diagnostic-control audit; neither supports a selection-dependent conclusion
because no representation is a `selected_candidate`. Efficiency has executable
offline and online diagnostic-control benchmarks, but training efficiency and
comparable hardware/reference-scale evidence remain unavailable.

## Executable R2/R8 spatial-transfer contract

`evaluate --transfer` writes `baseline_transfer_evaluation.json` or
`learned_transfer_evaluation.json`. The versioned `evaluation.transfer` YAML
definition fixes held-out regions, geohash levels, retrieval cutoffs, boundary
distance, distance metric, and the train-fitted relevance quantile. Known
labels and the distance radius are fitted only from events at or before
`train_end`; stale vocabularies containing test-only geography are rejected.

The report keeps distance retrieval, cross-geohash boundary-pair cosine,
held-out-region coverage, and seen/unseen geohash-5/geohash-7 slices separate.
Every slice reports rows, users, later-time coverage, and known-label coverage.

## Report runtime provenance (R13)

Scientific JSON reports retain their existing metric and lineage contracts and
add a separate top-level `runtime_metadata` object. Version
`geoembeddings-runtime-metadata/1.0` records the interpreter, package, PyTorch,
OS, device class, Git commit, resolved integer seed, and finite non-negative
wall-clock duration. Accelerator details are nullable and deliberately separate
from required portable fields. Missing package, source, or device information is
represented by explicit `null`, not omitted or guessed. This metadata supports
reproduction and later efficiency evaluation; by itself it is neither a peak
memory measurement nor a fair hardware benchmark, so R13 remains pending the
T1.7 offline-efficiency harness.
Empty slices are valid zero-coverage results, not silently substituted data.
These public-observation metrics do not use protected truth, do not establish
causal geographic invariance, and cannot measure unseen-POI transfer.

## Executable R1/R4 episode metric contract

`evaluate --episodes` assigns dense timestamps to half-open truth intervals (`start_time <= timestamp < end_time`). Boundary-relative hour-bin edges come from `evaluation.episode_response.boundary_bin_edges_hours` in versioned YAML. Within/adjacent-episode cosine and response curves are paired with different-user cosine and effective rank. Intent probes use deterministic held-out users and report class counts, majority accuracy, accuracy, macro-F1, and balanced accuracy. Temporary drift and recovery are single-vector R1 diagnostics and do not establish persistent/context disentanglement.

## T1.6 temporal and routine diagnostics

`evaluate --temporal-routine` joins dense public timestamps to protected labels
only in evaluator code. It reports cyclic hour/day probes, episode duration,
elapsed and remaining-duration tasks, periodic identity/state retrieval, and
recurrent routine versus singleton non-routine episodes. User, label, class,
temporal-bin, and history coverage accompany different-user separation and
effective-rank checks. These observational diagnostics do not establish causal
schedule invariance. The simulator audit found no matched schedule intervention,
so schedule-shift evaluation is explicitly blocked.
