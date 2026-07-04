# Proposal: feedback-learned channel weights

Status: draft, for adversarial review before an implementation issue is opened.

## Motivation

The evaluation harness's fitted-weights experiment established four facts.

1. On collections where one channel is globally stronger, fixed per-channel
   weights fitted on a small graded sample recover most of the label-fitted
   oracle's gain over plain RRF: all of it on fiqa from 16 graded queries, 93%
   of it on cqadupstack's pooled subforums, the oracle exactly on two of three
   quora draws at a budget of 100.
2. Ruffle cannot find those weights itself. Its statistics read each channel
   against that channel's own history, which is the only comparison available
   without labels; a global quality ordering between channels is label-bound
   information. This is a property of the estimation problem, not of the
   implementation.
3. Composing fitted weights with per-query adaptation (`base_weight * g`) is
   robust to fit error: on bad draws the composed result degrades toward plain
   warm Ruffle rather than following the bad fit down.
4. The robustness has one hard exception. A fitted weight of exactly zero
   silences a channel beyond what per-query evidence can revive, and small
   fits choose zeros eagerly. Declared weights need a positive floor.

`base_weight` ships the one-shot version of this: an operator fits weights
once and declares them. This proposal is the cumulative version: the engine
consumes relevance feedback whenever it exists, in any quantity, and maintains
the cross-channel tilt itself, with the same bounded, shrunk, mergeable
statistics it already uses everywhere else.

## The identifiability boundary

No extension of Ruffle's unlabeled statistics can approach the oracle. The
tempting shortcut, crediting channels that agree with the fused top-k
(pseudo-relevance feedback), is consensus reinforcement: it rewards whichever
channels already dominate the fused order, and the information it consumes is
already spent by the diagnostics. Approaching the oracle requires evidence
from outside the ranking system: explicit judgments, or user outcomes standing
in for them. The design question is how little of that evidence the engine can
make do with, and the fitted experiment's answer is: tens of queries, not
thousands.

This layer therefore changes the contract by addition, not revision. The core
claim stays "label-free adaptation within channels"; this adds "label-efficient
learning across channels" as a separate, optional layer that is exactly as
bounded as everything else in the engine.

## Design

### Weight factorization

The fused per-query weight becomes

```
weight_c = base_weight_c * learned_c * g_c
```

renormalized over the present channels, where `base_weight` is the operator's
static declaration (shipped in 0.3), `g` is the existing per-query adaptive
factor, and `learned_c` is the new feedback-learned multiplier. All three
factors are independent and compose multiplicatively; `learned_c` is `1.0`
until feedback arrives, so the layer is inert by default.

### The feedback event

The operator reports relevance when they have it:

```
fuser.feedback(ranks_by_channel, relevant_ids)
```

where `ranks_by_channel` carries, per registered channel, the ranked ids the
channel returned for some query (the same shape the fuse call consumes, scores
unnecessary), and `relevant_ids` is the set of ids judged relevant for that
query. The call is valid at any time, in any batch size, on any schedule:
after a one-off grading session, per query from an online judgment stream, or
never.

Per event, each channel receives a credit: the mean RRF contribution the
channel gave the relevant documents,

```
credit_c = mean over d in relevant_ids of 1 / (eta + rank_c(d))
```

with an absent document contributing zero. Credits are comparable across
channels because they share the query and the judgment set; a channel that
consistently ranks the relevant documents higher earns consistently higher
credit. The exact estimator is the primary open question for the evaluation
plan below; the candidates are this RRF-contribution credit, a pairwise
win-rate (the fraction of relevant documents a channel ranks above each other
channel), and the fitted-weights grid search run engine-side over a retained
window of events. The proposal commits to the interface and the statistical
treatment, not to the winner.

### The learned summary and its statistics

Each channel's state gains a `feedback` summary: a streaming mean and variance
of its credits with an effective count, the same `MeanVar` structure the
separation and reference baselines use. All existing semantics apply
unchanged:

- Streaming update: one feedback event merges as a count-1 summary.
- Merge: feedback summaries from different deployments pool by the existing
  merge algebra, so grading effort accumulates across sessions and machines.
- Decay: the summary decays on the existing schedule, so stale judgments fade.
- Tag gating: the summary lives under the channel's model-version tag and
  resets on a model change, which mechanizes the tuning guide's "revisit the
  declaration when the model changes".

The multiplier is a shrunk, bounded normalization of the credit means:

```
learned_c = clamp(shrink(credit_mean_c / cross-channel mean credit, n_c), floor, cap)
```

where `shrink(x, n) = (n * x + n0) / (n + n0)` pulls toward neutral `1.0`
under low evidence with pseudo-count `n0`, and the clamp enforces a positive
floor and a finite cap. The floor is the zero-absorption lesson made
structural: no amount of feedback can silence a channel, only an operator's
explicit `base_weight = 0` can. Defaults ship conservative (`n0` tens of
events, cap near the existing `g_upper_bound`, floor near `g_floor`).

### Active grading

The engine already computes a per-query conflict diagnostic: confident
channels disagreeing about the top of the ranking. Those are precisely the
queries where a judgment carries the most cross-channel information, so the
diagnostic doubles as an acquisition function. The operational loop is:
watch conflict in telemetry, grade a handful of the highest-conflict queries,
feed the judgments back. This needs no engine change beyond what exists; it
needs a documented procedure in the tuning guide and a harness experiment
measuring how much faster conflict-selected grading approaches the oracle
than random grading at equal budget.

### Implicit feedback is explicitly out of scope

Clicks and selections are the abundant feedback source, and they are biased by
presentation: users interact with what the fused order showed them, so a
channel that disagrees with the current consensus never has its candidates
displayed and can never earn credit. That is a rich-get-richer loop with the
same from-the-inside-invisible shape as the wrong-query failure mode, and no
amount of shrinkage removes the bias; it only slows it. A later revision can
evaluate skip-pair signals and propensity corrections against a simulated
click model in the harness. This proposal covers explicit judgments only, and
the API shape (ids judged relevant, source unspecified) leaves the door open
without promising anything.

## Compatibility

The feedback summary is the first addition to persisted state since the
format-v2 overhaul, so this is a state format version bump, with the merge,
decay, and tag semantics above specified as part of the format change. A state
without feedback summaries loads with every `learned_c` at `1.0`. The config
grows a `FeedbackConfig` (`n0`, `floor`, `cap`); registration and fuse paths
are untouched, and with no feedback events the engine's output is bit-identical
to today's.

## Evaluation plan

The harness gates the feature the same way it gated the retuned defaults.

1. Estimator bake-off: replay warmup-split qrels as feedback events and trace
   nDCG@10 against event count for each candidate estimator, on the reversed
   split for selection and the standard split for confirmation.
2. Trajectory: how many events to reach 50%, 80%, 95% of the oracle gap, per
   collection, against the fitted one-shot baselines at equal budget.
3. Robustness: noise injection (a fraction of judgments flipped or mislabeled)
   and the degraded-channel suite with feedback active, confirming the floor
   prevents feedback from finishing off a channel that adaptation would need
   later.
4. Interaction: feedback plus declared `base_weight`, confirming the factors
   compose rather than fight; feedback under coupling.

## Non-goals

No online learning-to-rank, no per-query or per-user personalization, no
implicit-feedback ingestion in this revision, and no change to the label-free
core: a deployment that never calls `feedback` gets today's engine exactly.
