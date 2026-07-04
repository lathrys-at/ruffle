# Proposal: feedback-learned channel weights (staged)

Status: draft. The work is staged so the proven, low-risk part ships first and the
part that would change the persisted state format is gated behind an experiment
that has not run.

## Motivation

The evaluation harness's fitted-weights experiment showed that a small graded
sample recovers most of the label-fitted ceiling on collections where one channel
is globally stronger. On fiqa, fixed weights fitted on about 16 graded queries
lift nDCG@10 from the RRF baseline of 0.3476 to 0.5209, roughly 90% of the way to
the 0.5406 oracle. On cqadupstack the pooled fit recovers a similar share, and on
quora the first-draw fit reached the oracle. Ruffle cannot find these weights on
its own: its statistics read each channel against that channel's own history,
which is the only comparison available without labels, and a global quality
ordering between channels is label-bound information. This is a property of the
estimation problem, not of the implementation.

`base_weight`, shipping in 0.3, is the one-shot form of this result: an operator
fits weights once against a labeled evaluation and declares them, and Ruffle's
per-query adaptation composes on top. This proposal is about the cumulative form:
consuming relevance feedback whenever it exists, in any quantity, and maintaining
the cross-channel tilt from it.

The demonstrated benefit needs no change to the engine or the persisted state. The
parts that would change them rest on an estimator that has not been chosen or
tested. So the work is staged. Stage 1 delivers the proven benefit as an offline
helper with no engine change. Stage 2 runs the estimator bake-off the persisted
design depends on. Stage 3 adds the cumulative, persisted layer only if an
estimator survives that bake-off, and only then touches the state format.

## What the evaluation established, with its limits

Four facts frame the design, stated with the numbers and the tails.

1. Small graded samples recover most of the fixed-weight ceiling on
   dominant-channel collections: about 90% on fiqa from roughly 16 queries, a
   similar share on cqadupstack, the oracle itself on the first quora draw.

2. Composing a static fit with per-query adaptation (`base_weight * g`) protects
   the aggregate on bad non-zero draws: when such a draw fits badly the composed
   mean returns toward plain warm Ruffle rather than following the fit down. A
   fitted zero is the exception in fact 4, where nothing revives the channel.

3. That protection is about the mean, not the tail. The composed fitted rows
   carry real per-query loss tails: the 5th-percentile delta against RRF reaches
   -0.2003 on quora and -0.3691 on cqadupstack, against plain warm Ruffle's
   -0.0443 and -0.0867 on the same collections. A feedback layer that tilts
   weights must be measured against warm Ruffle's tail, not only its mean, or it
   trades quiet per-query damage for an aggregate gain.

4. A fitted weight of exactly zero silences a channel beyond what per-query
   evidence can revive, and small fits choose zeros eagerly. The nfcorpus fitted
   row, where a 10-query fit collapsed onto a single channel and landed at
   0.3255, below the 0.3427 RRF baseline, is that failure in the table. Declared
   or learned weights need a positive floor.

## The identifiability boundary

No unlabeled credit signal we know of approaches the oracle. Crediting channels
that agree with the fused top-k (pseudo-relevance feedback) rewards whichever
channels already dominate the fused order, and the information it consumes is
already spent by the diagnostics. Approaching the oracle requires evidence from
outside the ranking system: explicit judgments, or user outcomes standing in for
them. The design question is how little of that evidence the engine needs, and the
fitted experiment's answer is tens of queries, not thousands.

This layer changes the contract by addition. The core claim stays "label-free
adaptation within channels"; this adds "label-efficient learning across channels"
as a separate, optional layer.

## Stage 1: offline fit helper

A helper takes graded queries and the per-channel runs for them and emits a
fitted `base_weight` per channel, floored at a small positive value. It reuses the
grid search the harness already runs for the fitted conditions. It writes channel
configuration, not state: the operator declares the emitted weights the same way
they would declare any `base_weight`, and Ruffle adapts per query around them.

This stage carries facts 1 and 2 above into a deployable tool with no engine
change, no new statistic, and no state format change. It is also the substrate
for Stage 2: the same helper, run against held-out judgments, is how the bake-off
measures each candidate estimator.

## Stage 2: estimator bake-off

The persisted design's central open question is which statistic to learn from
feedback, and the candidates do not reduce to one persisted shape. The
RRF-contribution credit is a per-channel marginal quantity. A pairwise win-rate
(how often a channel ranks relevant documents above another channel) is an
inherently paired quantity that needs a per-pair store. A windowed grid search is
neither. The choice determines the persisted structure, so it has to be made
before anything is persisted.

The bake-off replays warmup-split judgments as feedback and traces nDCG@10 against
event count for each candidate, on the reversed split for selection and the
standard split for confirmation. Each candidate is implemented as a throwaway
harness prototype, its update rule, event-clocked decay, and merge, and evaluated
through the helper; the engine and the persisted state are untouched. The named
conditions are:

- Redundant channel pair: a duplicated or highly correlated channel pair, to test
  whether a marginal per-channel credit over-tilts toward correlated channels
  where a joint or pairwise estimator would not. This is the sharpest test between
  the candidates, so it needs a ground truth: the label-fitted joint weight on
  that corpus. The selected estimator's recovered tilt must be no worse than the
  paired estimator's against that reference. The roadmap's second dense channel
  supplies the corpus.
- Conflict versus representative acquisition: fit on conflict-selected queries and
  test on a representative query sample, to measure whether conflict-guided
  grading approaches the oracle faster than random grading at equal budget, and
  whether selecting on conflict biases the learned tilt away from what
  representative traffic wants.
- Difficulty and depth: whether an absent document contributes zero credit
  conflates channel quality with list depth, and whether a channel absent from an
  event is correctly excluded rather than counted as zero.
- Sparse-event decay: behavior when feedback is rare and stale relative to query
  volume.
- Label noise: a fraction of judgments flipped or mislabeled, and whether the
  positive floor prevents feedback from finishing off a channel that adaptation
  would later need.
- Do-no-harm tail: the 5th-percentile per-query nDCG@10 delta against RRF, the
  statistic the harness already records. The gate requires a candidate's p5 to be
  no worse than warm Ruffle's on every evaluated collection, read against a
  minimum query count or a bootstrap interval so the decision reads signal rather
  than tail noise on the smaller splits. A candidate that improves the mean while
  deepening the tail past warm Ruffle does not pass.
- Graded relevance: whether the estimator uses graded judgments or only binary
  ones.
- Persistence versus re-fit: the real alternative to a persisted statistic is
  storing the raw judgments and re-running the Stage 1 helper on their pooled,
  cumulative, cross-deployment union. A streaming, mergeable estimator earns the
  format bump only if it matches that re-fit at materially lower storage and
  compute, or wins where re-fitting cannot pool, such as when raw judgments from
  separate deployments cannot be centralized but their summaries can merge.

An estimator proceeds to Stage 3 only if it beats the Stage 1 one-shot helper at
equal judgment budget, passes the do-no-harm tail gate, and clears the
persistence-versus-re-fit bar above.

## Stage 3: persisted feedback layer, if an estimator survives

In the roadmap this stage is the candidate tracked as "Persisted feedback-learned
weights", not committed until an estimator survives Stage 2. Only this stage
changes the engine and the state format, and only for an estimator that has passed
Stage 2. The pure primitive, its CLI, the events schema, and the tag gate below
are stable across the candidate estimators; the persisted structure, its
composition into weights, and several semantics are left to the estimator the
bake-off selects.

### The interface

The state-level primitive is pure: a state plus a batch of feedback events yields
a new state.

```
feedback(state, events) -> state
```

An event carries, per channel, the ranked ids the channel returned for one query,
the ids judged relevant for that query, and the model-version tag the channel's
outputs were graded under. `Fuser.feedback` is a thin live wrapper over the same
primitive. A `ruffle feedback --state in.json --events graded.jsonl --out out.json`
subcommand sits beside the existing `reconcile` and `rekey`, so central grading
can produce a new state that flows back through `reconcile` to deployments. The
events file schema is part of this stage and includes the per-channel tag.

The primitive gates on the model-version tag the same way merge and resume do. A
judgment graded on a channel's old model outputs must not fold into a state
carrying a new model tag, so `feedback` checks each event's tag against the
channel's tag in the input state and refuses on a mismatch, with the same
discipline as `RuffleState::merge`. A grading run with no input state cannot
synthesize a fingerprint and is not supported.

### The learned summary and its open choices

Each channel, or each channel pair, gains a feedback summary whose shape is the
estimator's natural one: a per-channel streaming summary for a marginal credit, a
per-pair store for a pairwise win-rate. The bake-off chooses. The semantics that
must be specified with that choice:

- Evidence weighting: an event backed by 50 judgments is not as reliable as one
  backed by 2, so an event folds in weighted by its judgment count, the way
  coupling weights each anchor reading by its overlap, not as a count-1 update.
- Cross-deployment merge: credits are comparable across channels only within one
  event's shared query and judgment set. A marginal per-channel summary pooled
  across deployments that graded different queries is no longer on a common
  footing, so either the persisted statistic is paired or relative and merges
  soundly, or cross-deployment feedback merge is scoped and documented rather than
  advertised. The bake-off's redundant-pair condition is what decides this.
- Decay cadence: feedback arrives on a grading schedule unrelated to query volume
  or anchor refreshes, so it decays once per feedback batch, not once per fuse.
  Decaying it inside the fuse path would fade judgments with query traffic.
- Tag gating: the summary lives under the channel's model-version tag and resets
  on a model change, the same as the other per-channel statistics.

### Weight composition

For the marginal-credit candidate the learned quantity is a per-channel scalar,
and the fused per-query weight is

```
weight_c = base_weight_c * clamp(learned_c * g_c, g_floor, g_upper_bound)
```

renormalized over the present channels, with `learned_c` at 1.0 until feedback
arrives. A pairwise or joint winner has no per-channel scalar: its learned
statistic is a matrix reduced to per-channel weights the way coupling reduces its
correlation matrix, and it composes through that reduction rather than as a
multiplier. Marginalizing a pairwise store into a scalar would reintroduce the
redundancy blindness that estimator exists to avoid, so the composition is part of
what the bake-off settles, not fixed here.

The floor and cap bound the adaptive composite `learned_c * g_c`, not each factor
alone. Ruffle already caps `g` at `g_upper_bound` to stop any one channel from
dominating; a separate cap on `learned_c` near the same value would stack two
ceilings and let a channel reach several times the intended limit, which
renormalization bounds in sum but not in ratio. The declared `base_weight` stays
outside that clamp: it is the operator's deliberate static tilt, uncapped, and a
declared `base_weight` of zero still silences a channel, the semantics it carries
in 0.3. Where coupling is enabled the learned factor composes with the coupling
reduction rather than duplicating it, and a pairwise feedback statistic in
particular must specify how it coexists with the redundancy discount instead of
double-counting redundancy.

### Compatibility and migration

This stage bumps both the state format version (a new field) and the statistic
version (a new learned quantity, whose meaning is fingerprinted and gates every
merge and resume). Because the version gate is strict equality in both directions,
an older state does not silently load at `learned_c` 1.0; it is refused unless it
is up-converted. So this stage ships a migration path, a state up-convert that
carries existing separation, reference, and coupling summaries forward and
initializes the feedback summary empty. Without it the bump would strand every
operator's accumulated baselines.

## Active grading

The engine already computes a per-query conflict diagnostic: confident channels
disagreeing about the top of the ranking. Those queries carry the most
cross-channel information, so the diagnostic can double as an acquisition
function: watch conflict in telemetry, grade a handful of the highest-conflict
queries, feed the judgments back. This guidance ships only with the Stage 2
acquisition result behind it, because conflict-guided grading is a hypothesis
until the conflict-versus-representative condition measures it, and because the
diagnostic reads (0,0) until at least two channels are both warm and performing at
or above their own norm on the query. The cold-start case, where the fitted
experiment showed the largest recovery, is therefore not where conflict guidance
applies; random or coverage grading covers it until baselines qualify.

The operating guide gains the retention loop this needs: at query time, retain the
query, its conflict value, and the per-channel ranked ids for queries above a
conflict threshold or in a top-N ring buffer, since a feedback event needs the
rank lists. Grade those offline and feed them through the CLI.

## Implicit feedback is out of scope

Clicks and selections are the abundant feedback source, and they are biased by
presentation: users interact with what the fused order showed them, so a channel
that disagrees with the current consensus never has its candidates displayed and
can never earn credit. That is a rich-get-richer loop that shrinkage slows but does
not remove. A later revision can evaluate skip-pair signals and propensity
corrections against a simulated click model in the harness. This proposal covers
explicit judgments only, and the interface (ids judged relevant, source
unspecified) leaves the door open without promising anything.

## Non-goals

No online learning-to-rank, no per-query or per-user personalization, no
implicit-feedback ingestion, and no change to the label-free core. A deployment
that never grades anything gets today's engine exactly. Stages 1 and 2 add no
persisted state at all; only Stage 3 does, and only for an estimator that has
earned it.
