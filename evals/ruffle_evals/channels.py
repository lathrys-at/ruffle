"""The three retrieval channels and their cached run files.

The channels are deliberately heterogeneous in score scale and deliberately
partially redundant: BM25 and the character-ngram TF-IDF are both lexical, so
their pair is where a learned redundancy discount has something to find, while
the dense channel carries the independent signal. All three are
higher-is-better.

Top-k runs are cached to disk as JSON keyed by query id, so fusion experiments
re-run without touching the models. Anchor construction needs scores for
arbitrary (query, document) pairs, which only the live models can produce, so the
`Channels` object also exposes full scoring.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import bm25s
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ruffle_evals import CACHE_DIR
from ruffle_evals.datasets import Dataset

__all__ = ["CHANNEL_KEYS", "Channels"]

CHANNEL_KEYS = ("bm25", "tfidf", "dense")

_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# One run entry: (doc_id, native_score), best first.
Run = dict[str, list[tuple[str, float]]]


class Channels:
    """The live channel models for one collection, with cached top-k runs.

    Construction indexes BM25 and TF-IDF and loads (or computes and caches) the
    dense corpus embeddings; the sentence-transformer model itself loads lazily,
    only when embeddings are absent from the cache or queries need encoding.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._doc_ids: list[str] = list(dataset.docs.keys())
        self._doc_index = {d: i for i, d in enumerate(self._doc_ids)}
        texts = [dataset.docs[d] for d in self._doc_ids]

        self._bm25_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(self._bm25_tokens, show_progress=False)

        # char_wb ngrams stay inside word boundaries, which keeps the vocabulary
        # bounded; the max_features cap holds the fiqa-scale matrix to a workable
        # size. norm="l2" (the default) makes every dot product a cosine.
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=200_000
        )
        self._tfidf_docs = self._tfidf.fit_transform(texts)

        self._query_emb_cache: dict[str, np.ndarray] = {}
        self._encoder = None
        self._dense_docs = self._corpus_embeddings(texts)

    # -- run files -----------------------------------------------------------

    def runs(self, k: int) -> dict[str, Run]:
        """The top-k run for every channel, computed once and cached on disk."""
        return {key: self._run(key, k) for key in CHANNEL_KEYS}

    def _run(self, key: str, k: int) -> Run:
        path = CACHE_DIR / "runs" / self._dataset.name / f"{key}-k{k}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            return {qid: [(d, float(s)) for d, s in items] for qid, items in raw.items()}
        run = self._compute_run(key, k)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run))
        return run

    def _compute_run(self, key: str, k: int) -> Run:
        qids = list(self._dataset.queries.keys())
        run: Run = {}
        for qid in qids:
            scores = self.full_scores(qid, key)
            top = np.argsort(-scores, kind="stable")[:k]
            # A zero lexical score means no ngram or term overlap at all; keeping
            # such documents would pad the run with arbitrary ties.
            run[qid] = [
                (self._doc_ids[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0
            ]
        return run

    # -- full scoring (anchors) ----------------------------------------------

    def full_scores(self, qid: str, key: str) -> np.ndarray:
        """One channel's native score for every corpus document, for one query."""
        query = self._dataset.queries[qid]
        if key == "bm25":
            # Token ids from a per-query tokenize are relative to the query's own
            # vocabulary, so full scoring goes through string tokens, which the
            # index maps against the corpus vocabulary.
            tokens = bm25s.tokenize(query, stopwords="en", show_progress=False, return_ids=False)
            if not tokens[0]:
                return np.zeros(len(self._doc_ids))
            return np.asarray(self._bm25.get_scores(tokens[0]), dtype=np.float64)
        if key == "tfidf":
            qvec = self._tfidf.transform([query])
            return np.asarray((self._tfidf_docs @ qvec.T).todense(), dtype=np.float64).ravel()
        if key == "dense":
            return self._dense_docs @ self._query_embedding(qid)
        raise KeyError(key)

    def score_lookup(self, qid: str, key: str) -> Callable[[str], float]:
        """A ``(doc_id) -> float`` scorer over one precomputed full-score vector."""
        scores = self.full_scores(qid, key)
        index = self._doc_index
        return lambda doc_id: float(scores[index[doc_id]])

    @property
    def doc_ids(self) -> Sequence[str]:
        return self._doc_ids

    # -- dense embeddings ------------------------------------------------------

    def _corpus_embeddings(self, texts: list[str]) -> np.ndarray:
        path = CACHE_DIR / "emb" / f"{self._dataset.name}-minilm.npy"
        if path.exists():
            return np.load(path)
        emb = self._encode(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    def _query_embedding(self, qid: str) -> np.ndarray:
        emb = self._query_emb_cache.get(qid)
        if emb is None:
            emb = self._encode([self._dataset.queries[qid]])[0]
            self._query_emb_cache[qid] = emb
        return emb

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(_DENSE_MODEL)
        return self._encoder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float64)
