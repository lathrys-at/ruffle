# Roadmap

A living list of planned and candidate changes. Items move out of this document
when they ship (into the changelog) or when they are rejected (with a note here
recording why, so the reasoning is not lost).

## Proposed for 0.4

### Label-efficient cross-channel weighting

Two stages. The first ships the demonstrated benefit with no engine change; the
second is an experiment that decides whether a persisted layer follows.

Stage 1 is an offline fit helper: graded queries and their per-channel runs go
in, a fitted `base_weight` per channel (floored at a small positive value) comes
out. It writes channel configuration, not state, so it needs no format change.
It carries the fitted-weights experiment's demonstrated gain, most of the
label-fitted ceiling from a few dozen graded queries on dominant-channel
collections, into a deployable tool.

Stage 2 is an estimator bake-off in the harness: which statistic a cumulative
feedback layer should learn, tested against the Stage 1 helper at equal judgment
budget with a do-no-harm tail gate against warm Ruffle. It runs against the
helper and the harness, with no engine or state change.

A cumulative, persisted feedback layer follows only if an estimator survives
Stage 2. It is tracked as a candidate below until then. Full design and the open
questions are in
[`proposals/feedback-learned-weights.md`](proposals/feedback-learned-weights.md).

## Candidate, not yet committed

### Persisted feedback-learned weights

The cumulative form of the 0.4 offline helper: the engine consumes explicit
relevance judgments and maintains the cross-channel tilt itself as a bounded,
shrunk, mergeable statistic, with a state format bump and a state migration path.
Committed only if the 0.4 bake-off selects an estimator that beats the offline
helper at equal judgment budget and passes the do-no-harm tail gate. It would be
the first state format bump since v2. Design and open questions in
[`proposals/feedback-learned-weights.md`](proposals/feedback-learned-weights.md).

### Heterogeneous-channel evaluation

The clean BEIR grid measures fusion over three correlated text channels, the
regime with the least per-query variance in channel competence and so the least
for adaptive weighting to do. A targeted follow-up would approximate a multi-modal deployment without
leaving text: a pooled multi-domain corpus where channel competence varies by
query domain, or a channel pair with deliberately disjoint strengths (dense
over titles against BM25 over bodies). The question it answers: how much of the
fixed-weight ceiling does per-query adaptation recover when no single weight
vector fits all queries.

### Larger-model evaluation rerun

A rerun of the benchmark grid with a stronger dense channel, for example
`Qwen/Qwen3-Embedding-0.6B`. Deferred on compute, not plumbing; details in
`evals/README.md`.

### Second dense channel for coupling

A fourth evaluation channel from a second embedding model, giving the coupling
estimator a redundant dense pair alongside the existing lexical pair. Also
tracked in `evals/README.md`.
