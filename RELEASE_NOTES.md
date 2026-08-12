# v0.5.0 release notes

The repository has continued beyond the original agent-handoff snapshot. New
simulations use `geoembeddings-dataset/2.0`; event-only 1.0 runs remain
explicitly compatible with the legacy modeling path.

Added:

- root `AGENTS.md` with binding invariants and development protocol;
- `START_HERE.md` for remote setup and reproduction;
- complete command/input/output/storage reference;
- architecture and evidence handoff;
- R1--R13 measurable requirement matrix;
- fair experiment and rerun protocol;
- phased simulator/model/recommendation roadmap;
- executable task backlog;
- literature-to-project guide based on the supplied papers;
- release verification evidence;
- reproducible `uv.lock`.

Verified:

- 13 tests passed;
- package reports version 0.5.0;
- locked dependency resolution passes;
- 50-user, 7-day learned pipeline completed on CPU;
- baseline export, protected evaluation, and comparison completed on the same
  smoke dataset.

Known scientific limitations remain explicit in `docs/CURRENT_STATUS.md` and
`docs/REQUIREMENTS_MATRIX.md`.

Implemented since the initial handoff:

- immutable paired exposure, opportunity, observation, schedule-shift,
  temporary-trip, and sustained-preference simulation with integrity-gated
  paired/change evaluation;
- accepted, indexed T0.4 replacement reference diagnostics and their no-aggregate
  decision record;
- typed component checkpoint/export schemas, the `factorized_pc` model, a
  capacity-matched control, and branch/loss ablation configurations;
- dataset contract 2.0 public POI, request, impression, and interaction tables,
  including synthetic Hakone request-time attributes and explicit 1.0 reading;
- seeded reliability/repeatability reports and observed-only offline benchmarks;
- T3.5 observed-only frozen-embedding candidate ranking;
- T3.6 observed-only exposure-aware training and protected regret/probability
  recovery that execute only after pair-integrity and ranking authentication;
- T3.7 observed-only frozen seen/unseen region/POI and early/late slices, with
  utility regret explicitly unavailable because truth is not an input.

T2.7 is complete with a **do not advance** decision: its matched persistent and
combined gates failed, so the routine branch remains closed. T3.4--T3.7 are
implemented, but their synthetic results are not real-world causal or
external-validity evidence and do not make R5, R8, or R9 scientifically
complete. Work now follows P4 priority order: T4.1 uncertainty calibration,
T4.2 adaptation/forgetting, T4.3 privacy, T4.4 online benchmarks, and T4.5
external-validity limits.
