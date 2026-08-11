# Requirements matrix

This is the acceptance contract for GeoEmbeddings. A requirement is not
considered satisfied because one proxy improves; it needs the specified
controlled evaluation and must be interpreted with adjacent failure checks.

## Core requirements

| ID | Requirement | Desired behavior | Required simulator/evaluation support | Primary metrics | Current status |
|---|---|---|---|---|---|
| R1 | Persistent/context separation | Temporary trips alter context without rewriting persistent preference | episode-aligned exports; matched temporary and sustained changes | persistent probe R2, context accuracy, persistent drift, adaptation delay | Partial |
| R2 | Multi-scale spatial fidelity | Representations support region, city, and fine local distinctions without boundary brittleness | distance/boundary pair generator; held-out cells | region/geohash accuracy, distance retrieval, boundary-pair consistency | Partial |
| R3 | Multi-scale temporal fidelity | Capture elapsed time, daily/weekly periodicity, and duration | periodic/routine labels and time perturbations | hour/day probes, duration error, periodic retrieval | Partial: T1.6 observational probes executable; controlled schedule shift blocked |
| R4 | Episode coherence | Events within an episode share context; adjacent episodes separate | episode-boundary embeddings and protected episode labels | within/between retrieval, episode classification, boundary change | Partial: episode coherence metrics executable; factorized persistent/context separation and matched change scenarios pending |
| R5 | Preference/opportunity separation | Same latent preference remains identifiable when candidate availability or exposure changes | matched-seed counterfactual scenarios with shared users/world | trait/preference invariance, representation drift, utility recovery | Pending |
| R6 | Cross-service alignment | One service history supports another without service identity dominating | leave-one-service-out views and targets | cross-service retrieval/prediction, missing-service degradation | Partial: deterministic leave-one-service-out evaluation is executable; candidate-aware transfer and archived T0.2 evidence remain pending |
| R7 | Noise/sparsity robustness | Moderate GPS noise, missing events, and dropout cause graceful degradation | deterministic corruption operators | performance/deviation curves, worst-group degradation | Partial: deterministic event-removal, GPS, timestamp, service-removal, and recent-truncation views are executable; archived T0.2 evidence, real-noise calibration, and causal invariance remain pending |
| R8 | Geographic/temporal transfer | Useful in later periods, unseen regions, and unseen POIs | explicit region/POI/time holdouts | frozen-probe/ranking deltas, cold-start coverage | Partial: held-out-region and seen/unseen-geohash slices executable; unseen-POI transfer pending |
| R9 | New-context recommendation | Tokyo history improves ranking after Hakone arrival and contextual actions | public requests, catalog, candidates, impressions, interactions; protected utility | Recall/NDCG/MRR, regret, calibration, adaptation delay | Blocked by data contract |

## Operational requirements

| ID | Requirement | Desired behavior | Required support | Primary metrics | Current status |
|---|---|---|---|---|---|
| R10 | Representation uncertainty | Low-evidence or unstable users receive lower confidence | window/event bootstrap or stochastic views | embedding variance, reliability-error relation, coverage-risk | Pending |
| R11 | Nonstationarity | Temporary change decays; sustained change updates persistent state | matched transient and sustained-change simulations | adaptation/forgetting time, permanent drift, regret over time | Pending |
| R12 | Privacy | Embedding does not unnecessarily memorize identities or sensitive attributes | membership/attribute audit splits | attack AUC, sensitive probe performance, utility/privacy curve | Pending |
| R13 | Computational efficiency | Feasible offline training and online update/export | benchmark harness | time, peak memory, examples/s, update latency, artifact size | Pending |

## Required controlled comparisons

### Persistent versus context

| User | Context | Expected persistent vector | Expected context/ranking |
|---|---|---|---|
| Same user | Tokyo routine | Stable | Routine-local |
| Same user | Hakone trip | Stable | Hakone/travel-adapted |
| Different user | Same Hakone context | Different | Personalized within context |
| Same user/context | Changed exposure set | Stable | Ranking may change only through available candidates |

### Temporary versus sustained change

- A one- or two-day trip should create a quick contextual shift and decay after
  return, with small persistent drift.
- A repeated, long-duration preference change should eventually update the
  persistent component.
- The simulator must expose change points only under `truth/`; training receives
  ordinary observed events.

### Stability versus collapse

Always pair temporal cosine similarity with:

- same-minus-different-user similarity;
- temporal same-user retrieval;
- centered effective rank;
- held-out preference and trait probes.

A representation that maps everyone to nearly the same vector is stable but
fails R1.

## Acceptance gates by milestone

### Evaluator gate

- Every reported metric uses held-out users or future time as appropriate.
- Baseline and learned inputs have matching source hashes and cutoffs.
- Corruption and counterfactual operators are deterministic under a seed.
- Requirements without direct evidence remain `pending` or `not_measurable`.

### Factorized-model gate

- `p`, `r`, and `c` are exported separately with explicit schemas.
- `p` improves persistent/preference recovery or invariance over the
  single-vector model.
- `c` improves episode/context responsiveness and adaptation delay.
- `r` improves periodic/routine tasks beyond both `p` and `c`.
- Improvements do not come from collapse, truth leakage, object-ID memorization,
  or inconsistent datasets.

### Recommendation gate

- The ranker sees only public candidate/request features.
- Candidate sets are frozen across compared models.
- Both stochastic-choice accuracy and true-utility regret are reported.
- Evaluate seen/unseen region, seen/unseen POI, and early/late trip stages.
- Compare frozen embeddings plus a small head separately from end-to-end tuning.

## Metric interpretation rules

- Negative probe R2 is valid evidence that the representation is worse than a
  train-mean predictor on held-out users.
- Top-5 accuracy is weak evidence when a field has few classes; always report
  class count and a popularity baseline.
- Geohash metrics require known-label coverage because test cells can be unseen
  in the train vocabulary.
- Classification accuracy alone is insufficient under imbalance; add macro-F1
  or balanced accuracy when implementing the new evaluator suite.
- Simulator truth permits utility regret and probability recovery, which are
  more informative than reproducing one stochastic chosen item.
- Never average R1--R13 into one score.
