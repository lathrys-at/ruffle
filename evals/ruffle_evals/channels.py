"""The retrieval channels and their cached run files.

The default channel set is deliberately heterogeneous in score scale and
deliberately partially redundant: BM25 and the character-ngram TF-IDF are both
lexical, so their pair is where a learned redundancy discount has something to
find, while the dense channel carries the independent signal. All channels are
higher-is-better. A corpus at the millions-of-documents scale runs with the
``bm25`` and ``dense`` subset (the canonical hybrid pair): a character-ngram
TF-IDF matrix is not workable at that size.

Top-k runs are cached to disk as JSON keyed by query id, so fusion experiments
re-run without touching the models. Anchor construction needs scores for
arbitrary (query, document) pairs, which only the live models can produce, so
the `Channels` object also exposes candidate-subset scoring.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import bm25s
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ruffle_evals import CACHE_DIR
from ruffle_evals.datasets import Dataset

__all__ = ["CHANNEL_KEYS", "DENSE_MODEL", "DENSE_SLUG", "Channels", "run_filename"]

CHANNEL_KEYS = ("bm25", "tfidf", "dense")

# The dense channel's model. gte-modernbert-base is a current (2025), ungated,
# Apache-2.0 retrieval model in the mid-50s BEIR range: strong enough that the
# dense channel carries modern signal, small enough (149M parameters) that the
# 8.8M-passage MS MARCO corpus embeds locally in about a day. It takes raw text
# with no instruction prefix; a model that needs one sets the two prompt
# constants. Sequences truncate at 512 tokens for throughput.
DENSE_MODEL = "Alibaba-NLP/gte-modernbert-base"
DENSE_SLUG = "gte-modernbert-base"
_DENSE_QUERY_PROMPT: str = ""
_DENSE_DOC_PROMPT: str = ""
_DENSE_MAX_SEQ = 512

_QUERY_CHUNK = 64


def run_filename(key: str, k: int) -> str:
    """The run cache filename for one channel: dense runs are keyed by the
    embedding model, so a model change can never serve another model's cache."""
    if key == "dense":
        return f"dense-{DENSE_SLUG}-k{k}.json"
    return f"{key}-k{k}.json"

# Above this corpus size the memmap/blockwise paths take over: embeddings live
# in an on-disk memmap rather than RAM, BM25 runs come from bm25s's own batched
# retrieve rather than full score vectors, and dense top-k is accumulated over
# document blocks.
_BIG_CORPUS = 1_000_000

_DOC_BLOCK = 400_000
_BIG_QUERY_CHUNK = 512
_ENCODE_BLOCK = 20_000

# One run entry: (doc_id, native_score), best first.
Run = dict[str, list[tuple[str, float]]]


class Channels:
    """The live channel models for one corpus, with cached top-k runs.

    Construction indexes the requested channels; the sentence-transformer model
    itself loads lazily, only when embeddings are absent from the cache or
    queries need encoding. The corpus text is released once the last model that
    needs it is built.
    """

    def __init__(
        self,
        name: str,
        doc_ids: list[str],
        texts: list[str],
        queries: dict[str, str],
        keys: Sequence[str] = CHANNEL_KEYS,
    ) -> None:
        self._name = name
        self._queries = queries
        self._keys = tuple(keys)
        self._doc_ids = doc_ids
        self._doc_index = {d: i for i, d in enumerate(doc_ids)}
        self._big = len(doc_ids) >= _BIG_CORPUS

        if "bm25" in self._keys:
            tokens = bm25s.tokenize(texts, stopwords="en", show_progress=self._big)
            self._bm25 = bm25s.BM25()
            self._bm25.index(tokens, show_progress=self._big)
            del tokens

        if "tfidf" in self._keys:
            # char_wb ngrams stay inside word boundaries, which keeps the
            # vocabulary bounded; the max_features cap holds the corpus-scale
            # matrix to a workable size. norm="l2" (the default) makes every dot
            # product a cosine.
            self._tfidf = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=200_000
            )
            self._tfidf_docs = self._tfidf.fit_transform(texts)

        self._query_emb_cache: dict[str, np.ndarray] = {}
        self._encoder = None
        if "dense" in self._keys:
            self._dense_docs = self._corpus_embeddings(texts)

    @classmethod
    def for_dataset(cls, dataset: Dataset, keys: Sequence[str] = CHANNEL_KEYS) -> Channels:
        doc_ids = list(dataset.docs.keys())
        texts = [dataset.docs[d] for d in doc_ids]
        return cls(dataset.name, doc_ids, texts, dataset.queries, keys)

    # -- run files -----------------------------------------------------------

    def runs(self, k: int) -> dict[str, Run]:
        """The top-k run for every channel, computed once and cached on disk."""
        return {key: self._run(key, k) for key in self._keys}

    def _run(self, key: str, k: int) -> Run:
        path = CACHE_DIR / "runs" / self._name / run_filename(key, k)
        if path.exists():
            raw = json.loads(path.read_text())
            return {qid: [(d, float(s)) for d, s in items] for qid, items in raw.items()}
        run = self._compute_run(key, k)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run))
        return run

    def _compute_run(self, key: str, k: int) -> Run:
        qids = list(self._queries.keys())
        if self._big and key == "bm25":
            return self._bm25_run_batched(qids, k)
        if self._big and key == "dense":
            return self._dense_run_blockwise(qids, k)
        run: Run = {}
        for start in range(0, len(qids), _QUERY_CHUNK):
            chunk = qids[start : start + _QUERY_CHUNK]
            sims = self._score_chunk(chunk, key)
            for row, qid in enumerate(chunk):
                run[qid] = self._topk_row(sims[row], k)
        return run

    def _score_chunk(self, qids: list[str], key: str) -> np.ndarray:
        """Native scores for a chunk of queries over the whole corpus, one row per
        query. Chunking bounds the dense (queries x docs) block that materializes."""
        queries = [self._queries[qid] for qid in qids]
        if key == "bm25":
            return np.vstack([self._bm25_full(q) for q in queries])
        if key == "tfidf":
            qvecs = self._tfidf.transform(queries)
            return np.asarray((qvecs @ self._tfidf_docs.T).todense(), dtype=np.float64)
        if key == "dense":
            emb = np.vstack([self._query_embedding(qid) for qid in qids])
            return np.asarray(emb @ self._dense_docs.T, dtype=np.float64)
        raise KeyError(key)

    def _topk_row(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        # argpartition instead of a full sort: the corpus can be half a million
        # documents and only the top k matter.
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        # A zero lexical score means no ngram or term overlap at all; keeping such
        # documents would pad the run with arbitrary ties.
        return [(self._doc_ids[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0]

    def _bm25_run_batched(self, qids: list[str], k: int) -> Run:
        """BM25 runs through bm25s's own batched retrieve, which never
        materializes full score vectors."""
        tokens = bm25s.tokenize(
            [self._queries[qid] for qid in qids],
            stopwords="en",
            show_progress=False,
            return_ids=False,
        )
        run: Run = {}
        for start in range(0, len(qids), _BIG_QUERY_CHUNK):
            chunk = qids[start : start + _BIG_QUERY_CHUNK]
            chunk_tokens = [tokens[i] if tokens[i] else [""] for i in range(start, start + len(chunk))]
            indices, scores = self._bm25.retrieve(
                chunk_tokens, k=min(k, len(self._doc_ids)), show_progress=False, n_threads=-1
            )
            for row, qid in enumerate(chunk):
                run[qid] = [
                    (self._doc_ids[int(i)], float(s))
                    for i, s in zip(indices[row], scores[row])
                    if s > 0
                ]
        return run

    def _dense_run_blockwise(self, qids: list[str], k: int) -> Run:
        """Dense top-k accumulated over document blocks, so the similarity matrix
        never exceeds (query chunk x doc block)."""
        run: Run = {}
        n_docs = len(self._doc_ids)
        for q_start in range(0, len(qids), _BIG_QUERY_CHUNK):
            chunk = qids[q_start : q_start + _BIG_QUERY_CHUNK]
            q_emb = np.vstack([self._query_embedding(qid) for qid in chunk]).astype(np.float32)
            best_scores = np.full((len(chunk), k), -np.inf, dtype=np.float32)
            best_idx = np.zeros((len(chunk), k), dtype=np.int64)
            for d_start in range(0, n_docs, _DOC_BLOCK):
                block = np.asarray(self._dense_docs[d_start : d_start + _DOC_BLOCK])
                sims = q_emb @ block.T
                take = min(k, sims.shape[1])
                part = np.argpartition(-sims, take - 1, axis=1)[:, :take]
                part_scores = np.take_along_axis(sims, part, axis=1)
                merged_scores = np.concatenate([best_scores, part_scores], axis=1)
                merged_idx = np.concatenate([best_idx, part + d_start], axis=1)
                keep = np.argpartition(-merged_scores, k - 1, axis=1)[:, :k]
                best_scores = np.take_along_axis(merged_scores, keep, axis=1)
                best_idx = np.take_along_axis(merged_idx, keep, axis=1)
            order = np.argsort(-best_scores, axis=1, kind="stable")
            best_scores = np.take_along_axis(best_scores, order, axis=1)
            best_idx = np.take_along_axis(best_idx, order, axis=1)
            for row, qid in enumerate(chunk):
                run[qid] = [
                    (self._doc_ids[int(i)], float(s))
                    for i, s in zip(best_idx[row], best_scores[row])
                    if np.isfinite(s) and s > 0
                ]
        return run

    # -- candidate scoring (anchors) -------------------------------------------

    def score_candidates(self, qid: str, doc_ids: Sequence[str], key: str) -> list[float]:
        """One channel's native scores for a candidate subset, for one query.
        Random access, so an anchor draw never needs a full corpus pass."""
        rows = [self._doc_index[d] for d in doc_ids]
        if key == "bm25":
            full = self._bm25_full(self._queries[qid])
            return [float(full[r]) for r in rows]
        if key == "tfidf":
            qvec = self._tfidf.transform([self._queries[qid]])
            sub = self._tfidf_docs[rows]
            return [float(v) for v in np.asarray((sub @ qvec.T).todense()).ravel()]
        if key == "dense":
            sub = np.asarray(self._dense_docs[rows], dtype=np.float64)
            return [float(v) for v in sub @ self._query_embedding(qid).astype(np.float64)]
        raise KeyError(key)

    def _bm25_full(self, query: str) -> np.ndarray:
        # Token ids from a per-query tokenize are relative to the query's own
        # vocabulary, so full scoring goes through string tokens, which the index
        # maps against the corpus vocabulary.
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False, return_ids=False)
        if not tokens[0]:
            return np.zeros(len(self._doc_ids))
        return np.asarray(self._bm25.get_scores(tokens[0]), dtype=np.float64)

    @property
    def doc_ids(self) -> Sequence[str]:
        return self._doc_ids

    # -- dense embeddings ------------------------------------------------------

    def _corpus_embeddings(self, texts: list[str]) -> np.ndarray:
        path = CACHE_DIR / "emb" / f"{self._name}-{DENSE_SLUG}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r") if self._big else np.load(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._big:
            emb = self._encode(texts)
            np.save(path, emb)
            return emb
        # Corpus-scale embeddings stream straight into an on-disk memmap; the
        # temp-name rename makes a crashed run recompute rather than load a
        # half-written cache.
        tmp = path.with_suffix(".npy.tmp")
        dim = self._encode(["probe"]).shape[1]
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(texts), dim))
        for start in range(0, len(texts), _ENCODE_BLOCK):
            out[start : start + _ENCODE_BLOCK] = self._encode(texts[start : start + _ENCODE_BLOCK])
            print(f"[{self._name}] embedded {min(start + _ENCODE_BLOCK, len(texts))}/{len(texts)}", flush=True)
        out.flush()
        del out
        tmp.rename(path)
        return np.load(path, mmap_mode="r")

    def _query_embedding(self, qid: str) -> np.ndarray:
        emb = self._query_emb_cache.get(qid)
        if emb is None:
            emb = self._encode([self._queries[qid]], prompt=_DENSE_QUERY_PROMPT)[0]
            self._query_emb_cache[qid] = emb
        return emb

    def _encode(self, texts: list[str], prompt: str = _DENSE_DOC_PROMPT) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(DENSE_MODEL)
            self._encoder.max_seq_length = _DENSE_MAX_SEQ
        if prompt:
            texts = [prompt + t for t in texts]
        # float32 keeps a corpus-scale embedding cache at millions of rows
        # manageable; scores are widened to float64 at scoring time.
        return self._encoder.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
