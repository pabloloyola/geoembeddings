# Current status

## Release

- Package/repository version: `0.5.0`
- Behavioral lineage: tested v0.4.x simulator and modeling code
- v0.5.0 changes: agent handoff, research protocol, roadmap, command reference,
  task backlog, literature guide, and locked dependencies
- Dataset contract: `geoembeddings-dataset/1.0`

## Working capabilities

- Semi-synthetic Kanto simulation across five controlled scenarios
- Structural and deep simulator validation
- Observed/truth information boundary
- Leakage-safe global temporal preparation
- Statistical history-vector baseline
- MPS-safe single-vector GRU training
- Learned embedding export at three cutoffs
- Observed-only dense timestamped embedding export with configurable event
  stride and no protected truth labels
- Protected latent probes and next-event evaluation
- Fair frozen baseline-versus-learned comparison

The historical release verification recorded in `docs/VERIFICATION.md` reports
13 passing tests and a 50-user, 7-day learned pipeline plus baseline comparison
completed end to end. That is a dated verification record, not the current test
inventory: the repository now contains 24 test functions across seven test
modules, including dense-export coverage added after that record.

## Pending scientific capabilities

- T1.2 protected evaluator alignment of dense timestamps to episode boundaries
  and direct episode-response evidence
- Direct persistent/context and routine/context metrics
- Controlled corruption and missing-service robustness
- Explicit geographic holdout experiments
- Matched counterfactual exposure/opportunity invariance
- Factorized encoder
- Recommendation request/impression/interaction data contract
- Candidate-aware ranking and Tokyo-to-Hakone evaluation
- Representation uncertainty, nonstationarity, privacy, and efficiency audits

## Immediate instruction

Run the comparison on the same 500-user dataset already used for learned
training. Preserve `embedding_comparison.json` and `.md` as the pre-factorization
reference. Then begin Phase 1 of `docs/ROADMAP.md`.

## T1.1 dense export and remaining T1.2 work (R1, R4)

The observed-only dense timestamped export is implemented. Statistical and
learned exports contain public user IDs, observed timestamps, cutoff kinds,
history counts, and embeddings, with no episode IDs or other protected truth
labels. T1.2 remains the protected evaluator work that aligns those timestamps
to truth episode boundaries and produces direct episode-response evidence.

No episode-coherence or persistent/context-disentanglement claim follows from
the dense export alone. Those claims must remain pending until T1.2 provides
direct protected evidence and the relevant baseline-versus-learned diagnostics.

The historical 50-user, 7-day smoke comparison remains an execution and
contract check only. Its small held-out sample and incomplete fine-geohash
coverage support no scientific model-quality, episode-coherence, or
disentanglement claim.
