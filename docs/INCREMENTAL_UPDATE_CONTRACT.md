# Incremental-update and online benchmark contract

## Status and scope

This document is the pre-implementation design contract for **T4.4 / R13**.
It fixes the observable update semantics, recomputation oracle, and benchmark
workloads before an online API is implemented. It changes neither the dataset
contract nor the current model and export formats. There is no T4.4 baseline
artifact yet: T1.7's `benchmarks/offline.json` measures frozen-artifact work,
not updates, and the negative T2.7 decision means that no current checkpoint is
a `selected_candidate`. Initial T4.4 results must therefore identify each
eligible representation as a `diagnostic_control` and must not make deployment
or factor-semantics claims.

The affected layers are the observed-only model/update path and the benchmark
evaluator. The benchmark and update API must read only `observed/`, preparation
metadata, configuration, and an authenticated checkpoint/export. They must
never receive simulator truth.

## Update unit and state

The public primitive is an **atomic batch append**. One call contains one or
more complete observed-event rows for one or more users. A single-event update
is a batch of size one. A session is not an update unit: a call may contain a
partial session, multiple sessions, or no usable session grouping, and the API
must not wait for a session boundary.

Each state is bound to one checkpoint, preparation-definition hash, ordered
categorical and continuous field lists, vocabulary/statistics hashes, model
variant, component schema, maximum-history policy, dtype, and device. State is
not portable across any mismatch. The call validates every row before doing
accelerator work and commits all users or none; validation or execution failure
must leave the prior state and output unchanged.

For each accepted event:

- only the addressed user's state and outputs may change;
- a single-vector model may change only its `combined` vector;
- `factorized_pc` may change `persistent`, `context`, and `combined` because
  component names describe intended branches, not proven immutable semantics;
- a future routine-capable model may additionally change `routine` only when
  that component is declared in its checkpoint/export schema; and
- dimensions, ordered component names, dtype, finiteness, and normalization
  contract must remain fixed for the state's lifetime.

An output is the representation after the entire atomic batch. Implementations
may optionally return per-event intermediate outputs, but those are outside the
T4.4 comparison contract and must not be substituted for batch results.

## Event validation and edge cases

### Time and deterministic order

Events are ordered per user by the pair `(parsed UTC timestamp,
event_fingerprint)`. The fingerprint is SHA-256 over canonical UTF-8 JSON of
the complete observed event in the explicit prepared field order, with stable
null and numeric encoding. The API may receive rows in any batch order, but it
must canonicalize them before validation and execution. Every new ordering key
must be strictly greater than that user's last committed key. An out-of-order
or late row rejects the whole batch; online update never silently inserts,
replays, or recomputes a suffix. Timestamp parsing, timezone normalization, and
tie behavior must match full-history preparation.

### Duplicate events

Committed fingerprints are idempotency keys. Repeating an already committed
fingerprint is a no-op: it is counted as a duplicate, creates no update, and
returns the existing representation. Duplicate copies inside one call are
coalesced before execution. A row with the same timestamp but different public
content is distinct and follows fingerprint tie order. Reports must give input,
accepted, and duplicate event counts. Hash collisions are a hard error rather
than an equality assumption.

### Unseen categorical values

Categorical vocabularies are immutable and train-fitted. Any value absent from
the authenticated vocabulary maps to that field's existing unknown token.
Online work must not expand, reorder, or refit a vocabulary or normalization
statistic. Missing required fields, invalid categorical IDs, non-finite
continuous values, or a vocabulary without its declared unknown token reject
the batch on CPU before accelerator execution.

### Empty histories

An initialized user has an explicit empty state and **no representation**; the
API must not invent a zero vector or learned anonymous-user vector. Export
omits that user/cutoff and records the omission, matching the existing export
contract. The first accepted event transitions the user to a non-empty state
and may emit a representation. Training-window `min_history_events` remains a
training rule and does not prohibit first-event inference.

### Device placement and Apple MPS

All model parameters, tensor features, cached tensor state, and emitted vectors
must share the explicitly requested execution device and dtype. Inputs may be
constructed and validated on CPU before a deliberate transfer; implicit
cross-device copies are forbidden. User IDs, event fingerprints, timestamps,
ordering cursors, and **sequence lengths remain CPU control metadata**.

Padded sequence execution and floating-mask final-state selection remain the
portable reference. An implementation must not introduce
`pack_padded_sequence` or integer `gather` to select final valid GRU states
unless the same change adds and passes a dedicated Apple MPS forward/backward
incremental-update regression test. CUDA or MPS results may be reported only
when that device actually executed the measured region; fallback must be an
explicit failure or exclusion, never labeled as accelerator timing.

## Correctness oracle

Incremental output is required to equal a clean full-history recomputation on
the same device and dtype. "Full history" means all accepted, deduplicated
events through the batch cutoff, transformed under the identical preparation
contract and then limited by the model's configured maximum-history policy. It
does not mean an unlimited history when the authenticated model uses a bounded
window.

For every emitted component and every finite element, correctness passes when
`abs(incremental - recomputed) <= 1e-5 + 1e-4 *
abs(recomputed)`. Shape, component order, user/cutoff key, event count, dtype,
and finiteness must match exactly. The report also records maximum absolute and
relative error per component. Comparisons use the same device and dtype;
cross-device equality is diagnostic only. Any oracle mismatch invalidates the
corresponding latency and throughput result rather than being averaged away.

The oracle must cover first-event, duplicate-only, equal-timestamp distinct
events, unknown-category, maximum-history rollover, multi-user batch, and every
exported component. Recompute code must not call the incremental state-update
implementation internally.

## Immutable benchmark workload contract

The benchmark command must first materialize a read-only workload manifest
under schema `geoembeddings-online-workload/1.0`. Reusing a named workload
requires byte-identical input artifacts and manifest; regeneration writes a new
path rather than overwriting it. The manifest records its derivation version,
seed, ordered event fingerprints, user IDs, prefill/update boundaries, batch
membership and order, duplicate/unknown-value fixtures, component/cutoff
schema, source and preparation hashes, checkpoint hash, and its own canonical
SHA-256. Timing may be repeated from that manifest, but event selection may not
be redrawn.

Unless a versioned benchmark configuration declares different values before
the manifest is created, select at most 128 users by seeded SHA-256 order and
use these four separate workloads:

1. **Cold start:** empty state to the first event for each selected non-empty
   user, issued as single-event calls. Setup and checkpoint loading are outside
   timed regions. Correctness is checked after every call.
2. **Steady-state single event:** prefill up to the first 32 events per user,
   then append each of up to the next 32 events as a batch of one. Prefill is
   outside the timed region; state cloning/reset is measured separately and
   excluded. Users without an update event are recorded as exclusions.
3. **Batched updates:** use the same prefill rule and update pool as steady
   state, but freeze batches of 8, 32, and 128 events where available. A batch
   may span users and sessions; no user contributes more than one event to a
   batch unless the frozen manifest explicitly records the necessary per-user
   order. Report each realized batch size separately and never pad by replaying
   events.
4. **Export serialization:** serialize the frozen post-update representation
   state for all workload users, with the canonical component and field order,
   to an in-memory payload and to a newly created temporary file. Report the
   two paths separately, including payload/artifact bytes and SHA-256; file
   cleanup is outside the timed region. Deserialization plus schema/hash and
   finite-value validation is the correctness check.

Sparse source data may reduce realized users, events, or batch sizes but may
not cause event reuse or synthetic padding. Requested/realized counts and every
exclusion are part of workload identity. Cold-start, steady-state, batched, and
serialization numbers must never be pooled into one efficiency score.

## Measurement and report requirements

Each timed operation uses a configurable warmup count and iteration count; the
defaults are 10 warmups and 100 measured iterations and must be positive.
Every iteration restores an equivalent pre-operation state outside the timed
region. Device synchronization occurs immediately before and after timing.
Warmups execute the identical operation and oracle but do not enter statistics.

`benchmarks/online.json` must use a versioned schema and, for every workload,
representation component set, device, dtype, and realized batch size, report:

- latency samples or an authenticated samples hash plus p50 and p95 latency;
- throughput in accepted events/second and completed calls/second (and
  serialized bytes/second for export);
- peak process RSS and device-appropriate peak allocated/reserved memory,
  identifying unsupported measures as `null` with a reason;
- warmup count, measured iteration count, requested/realized users, events,
  calls, batch size, history lengths, and exclusions;
- CPU model, accelerator name/index, total device memory when available, OS,
  architecture, Python/package/PyTorch versions, backend/runtime versions,
  thread settings, dtype, device, and synchronization method;
- Git commit, dirty-tree state, benchmark seed and configuration hash,
  checkpoint SHA-256, export SHA-256 where applicable, preparation metadata
  and definition hashes, observed source hashes, workload-manifest hash, and
  command line; and
- oracle status, checked row/component counts, duplicate/unknown/empty-history
  checks, maximum absolute/relative error, serialization round-trip result,
  and all failures.

Latency quantiles use the recorded per-iteration wall-clock samples and a
documented deterministic quantile method. Throughput is total accepted work
divided by synchronized measured time, not the reciprocal of a percentile.
Peak-memory counters are reset after warmup and before each measured iteration;
the report states whether the result is a maximum across iterations. Artifact
loading, checkpoint loading, workload construction, prefill, state reset, and
oracle recomputation are excluded from operation latency and timed separately
if reported.

Hardware-specific reports are comparable only when workload-manifest,
checkpoint, source/preparation hashes, dtype, software versions, thread
settings, warmups, iterations, and synchronization semantics match. Even then,
report device results separately; do not name an aggregate hardware winner.

## Implementation acceptance gate

T4.4 implementation is not complete until unit tests cover validation,
idempotency, ordering, atomic rollback, workload freezing, quantiles, and
statistics; a CPU cross-stage test runs every workload against the independent
recomputation oracle; optional CUDA/MPS runs preserve identical workload
metadata; documentation names the produced immutable artifact; and limitations
and missing device measurements remain explicit. A dedicated MPS regression is
mandatory for any packed-sequence or integer-gather final-state change.
