# Research and implementation roadmap

## Phase 0 — Preserve the verified reference

Goal: establish a reproducible baseline before new scientific code.

Evidence milestones:

1. **Execution/contract smoke passed — verified.** The 50-user, 7-day learned
   pipeline, statistical baseline, and comparison completed under a shared
   preparation contract. `docs/VERIFICATION.md` records the commands and
   limitations. This state verifies plumbing only and is not an archived
   scientific baseline.
2. **Complete smoke manifest registered — pending.** Preserve or register an
   index of the smoke outputs with immutable paths or external-storage
   identifiers, source commit, dataset manifest hash, and a statement that the
   baseline and learned reports share the same preparation contract. Large
   generated binaries are not required to be committed to Git.
3. **500-user reference evidence disposition recorded — closed as evidence
   lost/unverifiable.** The registered `docs/artifacts/t0.2-reference500.json`
   preserves historical hashes, but the exact bytes and recorded source commit
   are unavailable and no durable archive was registered. Recover those exact
   bytes, or run the matched baseline-versus-learned reference using the known
   working settings under a new run/experiment identity and new evidence-index
   lineage. Preserve resolved simulation and embedding configs, source hashes,
   seeds, cutoffs, checkpoint and training report, both cutoff exports, both
   episode reports, both event-removal reports, and comparison JSON/Markdown. Its
   artifact index must contain immutable paths or external-storage
   identifiers, source commit, dataset manifest hash, and an explicit statement
   that baseline and learned reports share source hashes, cutoffs, categorical
   field order, users, and the same preparation contract.
4. **T0.2a absence decision reconciled — complete.** The decision record selects
   only `finish the evaluator gate`, declares no aggregate winner, and leaves
   every scientific axis without an authenticated conclusion. Completion closes
   the evidence-disposition decision; it does not make T0.2 scientific evidence.
5. **Runtime/device metadata recorded.** Attach the environment evidence needed
   to reproduce and interpret the scientific reference.

Exit gate: milestone 3 is archived and indexed; the baseline and learned
exports and reports share the same preparation contract, and the reports
contain no unexplained missing users or non-finite values. Passing milestone 1
alone does not satisfy the Phase 0 exit gate.

## Phase 1 — Make the requirements measurable

Goal: create evaluation surfaces before optimizing new models.

### 1.1 Dense and episode-aligned export

Add a backward-compatible export containing embeddings at configurable times and
around protected episode boundaries. Training remains observed-only; the
evaluator joins timestamps to truth labels after export.

Suggested arrays:

```text
user_id
timestamp
cutoff_kind
embedding
history_event_count
```

Do not write episode IDs into the model export.

Status: the observed-only dense export foundation is implemented through
`export-dense`, including configurable event stride and the schema above.
Protected episode-boundary joining remains Phase 1.2 work.

### 1.2 Robustness operators

Implement deterministic evaluator-side observed-input views:

- event thinning by rate;
- GPS perturbation by distance distribution;
- service removal;
- recent-history truncation;
- timestamp jitter.

Re-encode the same users and report representation drift and downstream
degradation curves.

Status: versioned deterministic GPS, timestamp, leave-one-service-out, and
recent-truncation views are implemented through `geoembed robustness`. View
exports use `robustness/{kind}/{view_id}.npz`, and reports use
`robustness/{kind}_robustness.json`. These deterministic simulator/evaluator
views do not establish causal invariance or calibration to real-world noise.

### 1.3 Explicit transfer splits

Status: `evaluate --transfer` implements train-fitted distance retrieval,
geohash-boundary pairs, held-out-region coverage, and seen/unseen
geohash-5/geohash-7 slices with later-time and known-label coverage. Unseen-POI
transfer remains gated by the observable recommendation contract. These
synthetic observed-geography slices do not establish causal transfer or external
geographic validity.

### 1.4 Counterfactual exposure/opportunity pairs

Refactor simulator RNG streams so matched scenarios preserve users, latent
preferences, world, and ideally episodes while changing only opportunity,
exposure, or observation. Add pair manifests and an evaluator that matches
user/time records.

Exit gate: R1--R8 have direct executable tests or explicit, well-scoped blockers.

## Phase 2 — Two-way factorized representation

Goal: test whether separate persistent and contextual capacity improves the
stability/responsiveness tradeoff.

Add a model interface that returns:

```python
{
    "persistent": p_u,
    "context": c_u_t,
    "combined": z_u_t,
}
```

Candidate design:

- a shared event encoder;
- a long-history persistent encoder with conservative updates;
- a recent-window context encoder;
- a gated fusion head for predictive/ranking tasks;
- branch-specific objectives rather than forcing both branches to solve every
  target.

Candidate signals:

- persistent: cross-context agreement, latent/preference proxy tasks from
  observed data, slow update consistency;
- context: next-event prediction, local episode contrast, time-to-next-event;
- combined: downstream prediction and later candidate ranking;
- independence: diagnostic cross-prediction or covariance penalty, used
  cautiously because strict orthogonality is not semantic disentanglement.

Compare against single-vector capacity-matched baselines. Export branches
separately and evaluate each on tasks it should and should not solve.

Exit gate: `persistent` improves invariance/persistent information, `context`
improves episode response, and neither result is explained by collapse.

## Phase 3 — Explicit routine representation

Goal: distinguish recurring periodic behavior from both identity and one-off
context.

Possible components:

- cyclic time-conditioned routine memory;
- day-type/time-bin prototypes;
- periodic attention over prior matching calendar positions;
- multi-resolution duration or point-process objectives.

Required tests:

- weekday commute versus weekend behavior;
- repeated routine recovery after a trip;
- novel one-off episode separation;
- schedule-shift intervention;
- routine branch ablation.

Exit gate: the routine branch materially improves R3/R4 without absorbing all
persistent preference or current context signal.

## Phase 4 — Recommendation-ready simulator contract

Goal: expose realistic rank-time information without leaking utility.

Add:

```text
observed/poi_catalog.csv.gz
observed/recommendation_requests.csv.gz
observed/impressions.csv.gz
observed/interactions.csv.gz
```

At each request, distinguish:

- available candidates;
- shown candidates and ranks;
- unshown but available candidates when observable by the platform;
- user response;
- metadata known at request time.

Keep under truth:

- candidate utility decomposition;
- true choice probabilities;
- latent intent/episode;
- counterfactual outcomes;
- inaccessible candidates.

Add opening hours, family suitability, indoor/outdoor, price, popularity,
capacity, temporary availability, coordinates, and travel time. Begin with a
formal onsen/restaurant/cafe/shop/hotel/attraction catalog around Hakone.

Exit gate: simple popularity, nearest-POI, and category-preference baselines can
run end to end without truth access.

## Phase 5 — Candidate-aware recommendation

Goal: measure actual business usefulness of frozen and fine-tuned embeddings.

Implement:

1. candidate metadata encoder;
2. user-context/candidate scoring function;
3. sampled-softmax or listwise/pairwise ranking objective;
4. exposure-aware training variant;
5. calibrated choice-probability output when appropriate.

Evaluate:

- first recommendation after Hakone arrival;
- next local POI;
- after hotel check-in or an onsen search;
- unseen region and unseen POI;
- different users in the same context;
- same user under changed candidate availability.

Metrics: Recall@K, NDCG@K, MRR, Hit Rate@K, utility regret, probability recovery,
context adaptation delay, and persistent drift after return.

Exit gate: frozen embeddings improve over non-personalized and statistical
baselines; end-to-end tuning is reported separately.

## Phase 6 — Reliability, privacy, efficiency, and external validity

Goal: make the representation usable beyond the controlled prototype.

- uncertainty from bootstrapped windows/stochastic views;
- transient versus sustained preference-change evaluation;
- membership and sensitive-attribute audits;
- training/update/export latency and peak-memory benchmarks;
- simulator calibration against aggregate external statistics where licensing
  and privacy allow;
- later, evaluation on real de-identified data under the same observed contract.

Exit gate: tradeoffs are reported explicitly; no simulator-only result is
presented as real-world validation.

## T1.2 implementation note (R1, R4)

The episode-response evaluator is executable through `evaluate --episodes`. Configured half-open response bins, coherence/boundary change, single-vector drift/recovery, held-out intent probes, separation, and effective rank are reported. It does not claim persistent/context disentanglement; matched dense baseline and learned reports are required for deltas.

### T1.4 robustness gate (completed)

The completed T0.2a absence decision selects the evaluator-gate path as its only
action; it makes no scientific conclusion from the missing reference evidence.
Independently, `robustness --views`
now covers versioned GPS, timestamp, leave-one-service-out, and recent-history
truncation views for R6/R7, with matched baseline/learned populations and strict
comparison contracts. The sensitivity suite does not establish causal
invariance, calibration to real noise, or representation disentanglement.
