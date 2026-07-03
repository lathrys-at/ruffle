"""BEIR test collections, loaded through ir_datasets.

Each collection yields the full document corpus, the test-split queries, and the
relevance judgments. Document text is the title and body joined with a space when
a title exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import ir_datasets

__all__ = ["DATASETS", "Dataset", "load"]

# name -> ir_datasets id of the split holding the test queries and qrels. All four
# download from public mirrors without a usage agreement. trec-covid has only 50
# queries, too few for a meaningful warm/eval split, so the default run set is the
# first three and trec-covid runs only when named explicitly.
DATASETS: dict[str, str] = {
    "scifact": "beir/scifact/test",
    "nfcorpus": "beir/nfcorpus/test",
    "fiqa": "beir/fiqa/test",
    "trec-covid": "beir/trec-covid",
}

DEFAULT_DATASETS = ("scifact", "nfcorpus", "fiqa")


@dataclass(frozen=True)
class Dataset:
    """One loaded collection: corpus text by doc id, query text by query id, and
    graded relevance judgments as ``qrels[qid][did]``."""

    name: str
    docs: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load(name: str) -> Dataset:
    """Loads a collection by harness name, downloading it on first use."""
    ds = ir_datasets.load(DATASETS[name])

    docs: dict[str, str] = {}
    for doc in ds.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = doc.text or ""
        docs[doc.doc_id] = f"{title} {text}".strip() if title else text

    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    return Dataset(name=name, docs=docs, queries=queries, qrels=qrels)
