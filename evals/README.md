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

Metrics are nDCG@10 (the BEIR standard), Recall@100, and MRR@10, via
[ir_measures](https://github.com/terrierteam/ir_measures). Each fused condition
carries a two-sided paired t-test on per-query nDCG@10 against the `rrf`
baseline, and the mean per-channel weights the engine actually used on the
evaluation queries.

## Collections

Datasets load through [ir_datasets](https://ir-datasets.com/) and download on
first use into `cache/`:

| name | ir_datasets id | docs | test queries |
|---|---|---|---|
| scifact | `beir/scifact/test` | 5K | 300 |
| nfcorpus | `beir/nfcorpus/test` | 3.6K | 323 |
| fiqa | `beir/fiqa/test` | 57K | 648 |
| trec-covid | `beir/trec-covid` | 171K | 50 |

The default run covers the first three. trec-covid has too few queries for a
meaningful warm/eval split and runs only when named explicitly.

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
writes `results/<dataset>.json` (aggregate metrics, per-query nDCG@10, mean
weights, and the environment) and regenerates `results/RESULTS.md` from all
result files present.
