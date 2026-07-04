# Roadmap

A living list of planned and candidate changes. Items move out of this document
when they ship (into the changelog) or when they are rejected (with a note here
recording why, so the reasoning is not lost).

## Planned for 0.3

### Operator-declared base weights

A `base_weight` field on `ChannelConfig` (default 1.0), multiplied into the
channel's weight alongside the adaptive discrimination factor, so the final
per-query weight is `base_weight * g`.

Ruffle deliberately does not learn that one channel is globally better than
another: that is cross-channel, label-bound information outside what per-channel
statistics can identify. But an operator who has run a labeled evaluation holds
exactly that information, and today the API gives them nowhere to put it. The
field lets the operator declare a fixed cross-channel tilt while Ruffle keeps
adapting per query around it, the same division of labor as `GoodScore`: the
engine learns what it can from traffic, the operator declares what only labels
can establish.

The motivating case from the evaluation harness is fiqa under a strong dense
model, where the dense channel alone far outscores equal-weight fusion and the
label-fitted oracle puts all its weight on dense. No calibration-free method can
see that from the inside. With `base_weight`, the oracle's fixed weights become
a configuration any operator with an evaluation set can reach, and Ruffle's
per-query adjustment composes on top.

Scope notes: config, not persisted state, so no state format version bump. The
bindings read defaults from the engine, so they pick the field up without
drift. The evaluation harness should gain a condition that sets oracle-derived
base weights and measures the composed result against the oracle ceiling.

## Proposed for 0.4

### Feedback-learned channel weights

A cumulative version of `base_weight`: the engine consumes explicit relevance
judgments whenever they exist, in any quantity, and maintains the
cross-channel tilt itself as a bounded, shrunk, mergeable per-channel
statistic. Includes active grading guidance (the conflict diagnostic as an
acquisition function for which queries to grade) and explicitly excludes
implicit click feedback until presentation bias has its own evaluation. First
state format bump since v2. Full design in
[`proposals/feedback-learned-weights.md`](proposals/feedback-learned-weights.md);
adversarial review pending before an implementation issue is opened.

## Candidate, not yet committed

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
