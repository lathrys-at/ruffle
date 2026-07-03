# Evaluation harness

This directory measures Ruffle as a rank fusion engine on standard BEIR test
collections, against plain reciprocal-rank fusion and against each retrieval
channel on its own. It is a development harness, not part of any published
package; the numbers of record live in [`results/RESULTS.md`](results/RESULTS.md).

## Protocol

Three retrieval channels produce a top-100 run per query, on heterogeneous score
scales and with a deliberate redundancy between the two lexical channels:

- `bm25`: Lucene-style BM25 over word tokens ([bm25s](https://github.com/xhluca/bm25s)).
- `tfidf`: cosine over sublinear TF-IDF of character 3-5-grams (scikit-learn).
- `dense`: cosine over `sentence-transformers/all-MiniLM-L6-v2` embeddings.

The queries of each collection are shuffled with a fixed seed and split in half.
The first half warms Ruffle's baselines; every condition is scored on the second
half only, so the fused rankings under comparison come from identical channel
runs over identical queries. No relevance judgment is ever visible to the engine;
warming is unsupervised, and the split only keeps the evaluated queries distinct
from the ones the baselines were first formed on.

The conditions:

- `bm25`, `tfidf`, `dense`: each channel's own run, for context.
- `rrf`: plain unweighted RRF (eta = 60), implemented in the harness independently
  of the engine, with the engine's tie conventions (tied scores within a channel
  share their midrank; fused-score ties fall back to first-seen order) so
  agreement with the cold condition is checkable ranking for ranking.
- `borda`, `isr`: two further rank-based rules, Borda count over the truncated
  lists and inverse square rank, as completeness rows for RRF's own family.
- `combsum`, `combmnz`: the classic score-based fusions (Fox and Shaw), over
  min-max normalized scores. Both require bringing every channel's scores onto a
  shared scale first, which is exactly the per-channel calibration step Ruffle
  avoids; they are the contrast class.
- `rrf-oracle`: RRF with fixed per-channel weights grid-searched on the unit
  simplex (step 0.1) against the evaluation split's own relevance judgments. The
  labels choose the weights, so this row is not a competitor: it is a ceiling on
  what any fixed per-channel weighting could achieve with these runs, and the
  table reads as a bracket, the RRF floor to the oracle ceiling, with Ruffle's
  label-free weights in between.
- `ruffle-cold`: `Fuser.fuse_stateless` with an empty prior, per query. With no
  accumulated baselines and no declared references this reduces to unweighted
  RRF by construction; the condition verifies that reduction on real runs.
- `ruffle-warm`: a stateful `Fuser` replays the warmup queries, then fuses the
  evaluation queries with its accumulated per-channel baselines, the default
  configuration (redundancy discount off).
- `ruffle-warm-coupled`: as `ruffle-warm`, with `CouplingConfig(enabled=True)` and
  anchor refreshes interleaved through the warmup. Each anchor scores a seeded
  random draw of 256 corpus documents under every channel for one warmup query; a
  random draw rather than any channel's top-k, because a top-k pool is a selected
  sample that biases the correlation estimate.
- `ruffle-warm-aggressive`: the same estimators with the conservatism turned
  down and every evidence gate intact: discrimination reacts more sharply to a
  departure from a channel's own norm (`g_slope` 2.5, `g_floor` 0.1) and the
  redundancy discount, once gated in, may remove most of a duplicated signal
  (`discount_cap` 0.9, `shrink_to_identity` 0.2), with anchor refreshes as in
  the coupled condition. The profile prices the conservative defaults: what a
  harder tilt buys where one channel dominates, and what it costs on the
  degraded channels. No setting of these knobs learns that one channel is
  globally better than another; that is cross-channel, label-bound information
  outside the engine's contract.

Metrics are nDCG@10 (the BEIR standard), Recall@100, and MRR@10, via
[ir_measures](https://github.com/terrierteam/ir_measures). Each fused condition
carries a two-sided paired t-test on per-query nDCG@10 against the `rrf`
baseline, and the mean per-channel weights the engine actually used on the
evaluation queries.

## Targeted experiments

Two further protocols run per collection, reusing the cached runs.

The degraded-channel experiment adds a broken fourth channel derived from the
BM25 run and measures what it costs each fusion, in two failure modes chosen
because they sit on opposite sides of what label-free weighting can see.
`wrong-query` serves another query's BM25 results: internally healthy scores
over irrelevant content. Per-channel statistics read a channel against its own
norm, so this mode is designed to be invisible to them; the honest expectation
is that Ruffle matches RRF's damage rather than recovering it, with the conflict
diagnostic as the signal that moves. `flaky` serves the tail of the channel's
own results (ranks 51-100, true low scores) on a seeded half of the queries,
simulating intermittent retrieval failure; that failure is visible per query as
a departure from the channel's learned norm, which is what discrimination
weighting reads. Each mode reports three conditions (three-channel RRF, four-
channel RRF, four-channel Ruffle warm), the broken channel's mean weight, and
for the flaky mode that weight split across failed and healthy queries.

The learning curve warms a fresh stateful fuser on increasing prefixes of the
warmup split and scores the same evaluation split each time, tracing the climb
from the cold floor toward the fully warmed numbers. Warmup size zero is
online-from-cold: the fuser still adapts across the evaluation queries
themselves.

## Collections

Datasets load through [ir_datasets](https://ir-datasets.com/) and download on
first use into `cache/`:

| name | ir_datasets id | docs | test queries |
|---|---|---|---|
| scifact | `beir/scifact/test` | 5K | 300 |
| nfcorpus | `beir/nfcorpus/test` | 3.6K | 323 |
| fiqa | `beir/fiqa/test` | 57K | 648 |
| quora | `beir/quora/test` | 523K | 10,000 |
| trec-covid | `beir/trec-covid` | 171K | 50 |
| cqadupstack | `beir/cqadupstack/*` | 457K over 12 corpora | 13,145 |
| msmarco | `msmarco-passage` | 8.8M | 6,980 dev + 43 dl19 + 54 dl20 |

The default run covers the first four; quora is the statistical-power
collection, leaving 5,000 evaluation queries after the split. trec-covid has
too few queries for a meaningful warm/eval split and runs only when named
explicitly.

cqadupstack and msmarco run through dedicated runners, only when named
explicitly, and produce the main comparison only (the degraded and curve
experiments answer mechanism questions already covered on the standard
collections). cqadupstack follows the BEIR reporting convention: each subforum
is its own corpus with its own channels, warmup, and oracle, metrics are
macro-averaged over the twelve, and the paired test pools per-query values.
msmarco uses two channels (BM25 and dense, the canonical hybrid pair; a
character-ngram TF-IDF matrix is not workable at 8.8M passages), with corpus
embeddings in an on-disk float32 memmap and blockwise top-k scoring. Its
dev/small queries split into warmup and evaluation halves, and the TREC-DL
2019/2020 judged sets are evaluated by fusers resumed from the same dev-warmed
state snapshot, a transfer of warm baselines to a foreign query set over the
shared corpus.

## Running

The harness needs Python 3.10+ with `requirements.txt` installed and the `ruffle`
package importable (a wheel built from `bindings/python`, or `pip install
ruffle`). A full default run is:

```
python -m ruffle_evals
```

or for one collection at a chosen depth:

```
python -m ruffle_evals scifact --k 100
```

Channel runs and corpus embeddings are cached under `cache/` keyed by collection
and run depth, so re-runs only re-execute the fusion and the metrics. Each run
writes `results/<dataset>.json` (the main comparison: aggregate metrics, mean
weights, p-values, and the environment),
`results/<dataset>-degraded.json`, and `results/<dataset>-curve.json`, then
regenerates `results/RESULTS.md` from all result files present.

## Deferred

A fourth channel using a stronger embedding model (`BAAI/bge-small-en-v1.5`,
which wants a query instruction prefix at encode time) is noted but deferred
until after the larger-corpus collections (cqadupstack, MS MARCO / TREC-DL) are
in. As a replacement for MiniLM it would mostly shift collections into the
dense-dominant regime fiqa already covers; the interesting configuration is as
an addition, giving a redundant dense pair for the coupling estimator alongside
the existing lexical pair, and a strong/weak mix within one modality.
