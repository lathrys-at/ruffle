# Operating and Tuning Ruffle

This guide is for the operator running Ruffle in a real deployment: what to watch,
and when, why, and how to move a default.

Using a default configuration keeps Ruffle close to plain reciprocal-rank fusion,
so that it tilts weights only when evidence from the channels' own outputs support it.
Default values for parameters sit at the conservative end of their respective
ranges to preserve recall, and changing parameters to be more aggressive may trade
away recall safety.

## Data sources

The following data sources can be used to observe the behavior of Ruffle and guide
tuning decisions.

The first is per-query telemetry. Every [`Fused`] result carries the weights actually
used, the per-channel flags (`RanksOnlyDefaultWeighted`, `DegenerateSeparation`,
`NoReference`), the per-channel discrimination reads behind the weights
(`Fused::discrimination`: the combined `g`, the raw separation, the exported top-`m`
average), and the two agreement diagnostics (`confidence`, `conflict`). These are worth
logging, as nearly every symptom in this guide is a distribution over these fields.

The second is the persistent state. [`RuffleState`] serializes to JSON, and can be
inspected directly or through `ruffle reconcile --report` (the `cli` feature).
Per channel it holds the separation baseline (mean, variance, count) and the good-score
reference; per pair, the accumulated anchor redundancy (mean, variance, refresh count).

The third is divergence between snapshots. [`RuffleState::divergence`] is a
standardized distance between two states' per-channel baselines, covering both the
separation baseline and the native-units reference. You can use this to monitor
drift over time.

The fourth is comparison against plain-RRF. [`Fuser::fuse_stateless`] with an empty prior
reduces exactly to unweighted RRF, which makes plain RRF available as a baseline at any
time against which you can judge the output of stateful, weighted fusion.

## Adaptive weighting

Adaptive weighting pays off under two conditions (§9 and §10 of
[`derivation.md`](derivation.md)): the channels' informativeness varies from query to
query, or the channels are genuinely redundant. Where informativeness differences are
purely static and there is no redundancy, the per-query weighting contributes only
noise, and the defaults, which keep Ruffle near plain RRF, are the configuration to
run. Checking for these two conditions is worth doing before tuning anything.

Per-query variation shows up in the telemetry as spread in the standardized separation
across queries. For each channel, take the raw separation over monitored traffic and
standardize it against that channel's own baseline, or log `Fused::discrimination` and
look at the spread of `g` directly. If every channel's `g` sits in a tight band around
1.0 on almost every query, there is no per-query signal to exploit, and raising slopes
or bounds would mostly amplify noise.

Across-channel redundancy shows up on the anchor. After two or more refreshes, a pair
of channels' redundancy mean materially above zero (0.3 or more) with low variance
across refreshes says two channels share structure worth discounting. A mean near
zero, or a variance above the stability threshold, says the channels are independent,
which is what the default behavior of Ruffle already assumes.

If neither condition holds, stay at the defaults and revisit when the channel mix
changes. The defaults exist so that the no-signal case costs nothing over plain RRF.

## Channel priors

Declaring the good-score prior for a channel is worth it whenever you can. A declared
[`GoodScore`] establishes an initial baseline for the absolute-goodness statistic.
Without this prior, Ruffle has to learn the reference from traffic, and until enough
evidence accrues the channel runs on observed separation alone, which cannot tell the
difference between "nothing matches" and "everything matches equally well" (§4).
Declaring a good-score prior takes two numbers in the channel's native score units: the
top score of a typical, unremarkable query (`typical`), and the score a genuinely good
match reaches (`good`). A dozen or so manual queries against your own index is enough to
read them off. The pseudo-count `weight` sets how strongly Ruffle trusts your prior: at
`weight = n₀`, observed traffic has fully diluted the prior after a few multiples of n₀
queries. A small n₀ (2 to 5) suits a vaguely informed guess, while a large one (20 to 100)
suits evidence gathered from careful measurement. After enough traffic has accumulated,
consider comparing the inferred reference mean in the Ruffle state to the declared prior
`typical` value and make any necessary adjustments.

## Declared base weights

Ruffle never learns that one channel is globally better than another. Its statistics
read each channel against that channel's own history, which is the only comparison
possible without relevance labels; a global quality ordering between channels is
label-bound information. If you hold that information, from a labeled evaluation of
your own corpus or from domain knowledge, declare it as `base_weight` on the channel's
registration. The fused weight becomes `base_weight * g`, renormalized over the
channels present on the query, so the static tilt you declare and the per-query
adaptation compose rather than compete.

Only the ratios between channels matter; `(2, 1, 1)` and `(1, 0.5, 0.5)` are the same
declaration. The default `1.0` declares nothing. A `base_weight` of `0.0` silences a
channel's votes entirely while its baselines keep updating, which suits a channel you
want to observe in telemetry before letting it affect rankings. Base weights are
configuration, not persisted state: changing them takes effect on the next
construction and requires no state migration.

A reasonable procedure for setting them: run your channels over a set of queries with
graded relevance, grid-search fixed per-channel RRF weights on that set, and declare
the winner. Even a few dozen graded queries produce a usable tilt when one channel is
clearly stronger. Floor small fitted values at something positive (0.1, say) rather
than declaring zero: a zero excludes the channel from every query, beyond what
per-query evidence can revive, and a small fit chooses zeros too eagerly. Revisit the
declaration when a channel's model changes, since the tilt encodes a comparison
between specific model versions.

## Reading knobs

The following knobs control whether Ruffle's statistics are healthy for your
channels: whether separation is measurable, whether the good-score reference accrues,
and whether baselines warm up. They can be diagnosed from telemetry alone, with no
labels, and they are worth checking first, because a response knob tuned on top of
unhealthy readings amplifies their noise.

### `top_m` (default 5)

`top_m` is the fixed count behind the absolute-goodness statistic, the reference
refinement, and the diagnostics top-sets. A channel whose pools are usually shallower
than `top_m` never refines a learned reference, because a fixed-count average does not
exist below the fixed count (§4). The symptom is a `NoReference` flag that persists
long past cold start, and a reference count in the state stuck at zero (or at its
declared pseudo-count) while traffic flows. The pool-size distribution per channel
confirms the diagnosis. The preferred fix is declaring a good-score prior, which needs
no refinement to work; the alternative is lowering `top_m` toward your real pool
depth, which makes the statistic noisier per query (fewer scores averaged), only
partly offset by the pool-size shrinkage.

### `min_distinct_values` (default 8)

Below this many distinct score values the separation ratio is not computed at all. The
symptom of a too-high setting is a `DegenerateSeparation` flag on most of a channel's
queries when the channel plainly can rank; the classic case is an integer-count
channel whose scores take five or six values. The flag rate per channel, plus a
spot-check of a few flagged pools by eye, confirms it: if the pools show a real
top-versus-bulk shape over few distinct values, lowering toward 5 admits the reads.
The risk is meaningless ratios from genuinely degenerate pools, so going below 4 or 5
is not advisable, and the winsorize rate is worth watching after lowering.

### `denom_floor_frac` (default 0.75) and `winsor_z` (default 2.5)

Both protect the separation baseline from heavy-tailed reads. The floor widens a
near-tied bulk's denominator toward the inter-quartile gap; the winsor clamps a read
to the baseline mean ± `winsor_z` standard deviations before it enters the baseline.
The health check is the clamp rate: how often a logged raw separation falls outside
the baseline band. A few percent is normal. If reads
clamp persistently long after warm-up, the channel's separation distribution is
genuinely heavy-tailed; raising `winsor_z` (4 to 5) lets the baseline learn the true
spread, at the cost of slower recovery from a single wild query. If an integer-count
or otherwise tie-heavy channel produces separation reads that look muted relative to
what its pools show, the floor may be binding; comparing `q0.5 − q0.1` against
`denom_floor_frac · (q0.75 − q0.25)` on sample pools confirms it before you lower the
fraction.

### `top_eps` (default 0.10)

`top_eps` is the fraction of the pool treated as the extreme top in the separation
numerator. It should be small relative to the pool but comparable to the number of
genuinely relevant items a typical query has in the pool. If your queries are narrow,
with one or two relevant items in a deep pool, a 10% top dilutes the extreme top with
bulk and separation under-reads; lowering toward 0.05 sharpens it. If your queries
are broad and dozens of pool items are typically relevant, the relevant mass extends
past the top slice and raising toward 0.15 helps. Diagnosing this without judgments
is hard: on a handful of judged queries, you can check whether the channels with high
separation are the channels whose pools actually contain the relevant items.
If separation ranks the channels wrongly on most judged queries, `top_eps` is worth
revisiting before any response knob.

### `min_count_for_z` (default 5) and `shrink_pool_size` (default 20)

Both control how quickly Ruffle trusts new evidence. `min_count_for_z` is the baseline
count below which a standardized separation read is not trusted (and how quickly the
learned reference is trusted); `shrink_pool_size` is the pool size at which a query's
read receives full weight. The symptom of a mismatched `shrink_pool_size` is
structural: if your channels legitimately return 10 items, `pool_factor` caps at 0.5
forever and every weight is permanently pulled halfway to neutral. `shrink_pool_size`
belongs at or below your typical pool depth. `min_count_for_z` is in units of queries
observed and is reached quickly; raising it only makes sense against erratic weights
in the first day of a new channel's life, and a declared good-score prior is usually
the better fix for that.

## Response knobs

Response knobs change how hard the readings move the ranking, so changing one calls
for outcome evidence through a comparison on judged queries, however small. Fifty to two
hundred queries with binary or graded judgments is enough. The candidate configuration
should be compared against both the current one and plain RRF, and whatever metric you
are improving.

### `g_slope` (default 1.0)

`g_slope` is the logistic slope from a standardized reading to a weight factor: the main
aggressiveness dial for per-query adaptivity. Raising it makes sense only when per-query
variation is actually present (the first condition above), and only if the labeled
comparison shows the win over plain RRF growing as the slope rises, with recall@k flat.
The failure mode is the purely static case the derivation documents (§10): with no
per-query signal, a steeper slope converts read noise into ranking noise and can push
the fusion below plain RRF. If the current configuration is not beating plain RRF,
lowering the slope toward 0.5 moves the fusion back toward the plain-RRF baseline, and
that is the correct response.

### `g_upper_bound` (default 4.0) and `g_floor` (default 0.25)

`g_upper_bound` and `g_floor` bound a channel's weight. The neutral weight 1.0 does
not move with the cap, so tightening the cap is safe in the sense that at-norm
channels are unaffected. Raising the cap needs evidence that queries exist where one
channel deserves to dominate: on judged queries where a channel's reads are at their
strongest, giving it more room should improve nDCG without recall cost. Lowering the
floor is the most recall-dangerous change in the configuration. Before considering it,
it is worth measuring each channel's unique contribution on judged queries: the
relevant items that only that channel surfaced. A channel that reads poorly on average
but still supplies unique relevant items is exactly what the floor exists to protect.
If every channel's unique contribution at low readings is empty across your judgment
set, a lower floor gains precision; that situation is rare.

### `rrf_eta` (default 60)

`rrf_eta` is the standard RRF sharpness constant, orthogonal to Ruffle's weighting:
smaller values concentrate mass on the top ranks. 60 is the literature default and
interacts mildly with everything else (a smaller η makes any weight difference matter
more at the top). You can tune it the way you would tune plain RRF, on the labeled
set; tuning it before the weighting knobs keeps them calibrated against the η you will
actually run.

## Coupling

Coupling is off by default because independence is the only unconditionally
recall-safe assumption (§5.3). The following assumptions must hold before enabling it.

1. The anchor candidates are a random draw from the corpus, never any channel's top-k
   results or a union of result pools. Pooled candidates carry a selection bias
   (Berkson's paradox, §5.2) that reads as spurious anti-correlation, and the library
   cannot detect it from the ids. Anchor queries come from real traffic, and the
   anchor is sized so pairs clear `min_overlap` comfortably: a few hundred candidates
   per anchor query, with both-scored overlaps in the hundreds.
2. The anchor has been refreshed at least twice, over different query mixes. Stability
   across strata is a between-refresh property, and the `min_refreshes` gate
   (default 2) holds the discount back until then regardless.
3. The pair's redundancy mean in the state is 0.3 or more, with variance well under
   `stratum_stability_max_var` (default 0.25). A mean near zero means the gates would
   mostly suppress the discount anyway.

The default cap and shrinkage keep the discount mild: the ceiling is
`(1 − shrink_to_identity) · discount_cap = 0.25` of correlation, which moves a fully
redundant pair from weight 1.0 to about 0.92 each and an independent third channel up
to about 1.15. Raising `discount_cap` toward the measured correlation calls for a
labeled comparison at each step, with recall flat. The failure mode is query-dependent
redundancy (§5.3): a pooled discount over-discounts the queries where the channels are
actually independent; between-refresh variance creeping toward the stability threshold
is the signal to lower the cap.

`min_overlap` (default 30) and `min_reliability` (default 10) are statistical floors.
If pairs are not clearing them, the right response is a larger anchor, not lower
floors; a correlation estimated over twenty items is too noisy to act on.

## Decay and drift

Learned baselines in Ruffle do not decay by default. Enabling decay lets Ruffle adapt
over time to changing composition of a corpus or query patterns, but makes state merging
more difficult, where it becomes an approximate (§8) rather than exact operation. You
can determine whether you need to enable decay by snapshotting the Ruffle state
periodically and computing `divergence` between consecutive snapshots. Stable channels
have a divergence near zero. A channel whose divergence trends upward ("drifting") over
time may require enabling decay.

The decay `factor` controls how long Ruffle keeps memory. The effective memory depth
becomes `1 / (1 - factor)` observations, so the default of `0.98` holds a window of about
50 queries per channel. If you instead want to control decay manually, you can call
`RuffleState::decay`.

## Quick reference

| Knob | Default | Change when you observe | Check before changing | Risk if wrong |
|---|---|---|---|---|
| `top_m` | 5 | `NoReference` persists; reference count stuck | pool-size distribution | noisier D^abs |
| `min_distinct_values` | 8 | `DegenerateSeparation` on rankable integer channels | flag rate + pool inspection | meaningless ratios |
| `winsor_z` | 2.5 | clamp rate high after warm-up | raw reads vs baseline band | slow recovery from outliers |
| `denom_floor_frac` | 0.75 | muted separation on tie-heavy channels | quantile gaps on sample pools | blown-up ratios |
| `top_eps` | 0.10 | separation misranks channels on judged queries | judged spot-check | top mixed into bulk |
| `shrink_pool_size` | 20 | pools chronically below it; weights pinned near 1 | pool-size distribution | noise from thin pools |
| `min_count_for_z` | 5 | erratic first-day weights | early-traffic telemetry | slower adaptivity |
| `g_slope` | 1.0 | labeled win over RRF grows with slope | labeled comparison + recall check | noise amplification |
| `g_upper_bound` | 4.0 | strong-read queries want more dominance | labeled eval on strong-read slice | single-channel capture |
| `g_floor` | 0.25 | no unique relevant items at low readings | unique-contribution analysis | recall loss |
| `rrf_eta` | 60 | standard RRF tuning | labeled eval | top-rank over/under-emphasis |
| `coupling.enabled` | off | redundancy ≥ 0.3, stable, ≥ 2 refreshes | assumptions above | recall loss on independent queries |
| `discount_cap` | 0.5 | measured redundancy ≫ effective ceiling | labeled eval per step | over-discounting |
| `decay.enabled` | off | divergence trend between snapshots | divergence time series | forgetting real signal |

[`Fused`]: https://docs.rs/ruffle/latest/ruffle/struct.Fused.html
[`RuffleState`]: https://docs.rs/ruffle/latest/ruffle/struct.RuffleState.html
[`RuffleState::divergence`]: https://docs.rs/ruffle/latest/ruffle/struct.RuffleState.html#method.divergence
[`Fuser::fuse_stateless`]: https://docs.rs/ruffle/latest/ruffle/struct.Fuser.html#method.fuse_stateless
[`GoodScore`]: https://docs.rs/ruffle/latest/ruffle/struct.GoodScore.html
