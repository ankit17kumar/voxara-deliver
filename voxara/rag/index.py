"""Hybrid retrieval index: BM25 (lexical) + dense vectors, fused with RRF.

The dense side defaults to hashed character n-gram TF-IDF, which needs no
model download, handles Hinglish spelling drift, and embeds a query in well
under a millisecond. Pass `embed_fn` to swap in a real multilingual
embedding model (e.g. a Sarvam, Cohere or bge-m3 endpoint) without changing
anything else.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Callable, Sequence

import numpy as np

from .chunker import Chunk
from .text import char_ngrams, normalise

DIM = 2048


def hashed_ngram_embed(texts: Sequence[str], idf: dict[str, float] | None = None) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, t in enumerate(texts):
        for g, c in Counter(char_ngrams(t)).items():
            w = (1 + math.log(c)) * (idf.get(g, 1.0) if idf else 1.0)
            out[row, zlib.crc32(g.encode()) % DIM] += w  # stable across processes
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
                 k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # BM25 statistics
        self.doc_toks = [normalise(c.text) for c in chunks]
        self.doc_len = np.array([len(t) for t in self.doc_toks], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if chunks else 1.0
        df: Counter = Counter()
        for toks in self.doc_toks:
            df.update(set(toks))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tf = [Counter(t) for t in self.doc_toks]
        # Dense statistics
        if embed_fn is None:
            gdf: Counter = Counter()
            for c in chunks:
                gdf.update(set(char_ngrams(c.text)))
            self._gidf = {g: math.log(1 + n / d) for g, d in gdf.items()}
            self.embed_fn = lambda xs: hashed_ngram_embed(xs, self._gidf)
        else:
            self.embed_fn = embed_fn
        self.vectors = self.embed_fn([c.text for c in chunks]) if chunks else np.zeros((0, DIM))

    def _mask(self, workflow: str | None, lang: str | None) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        if workflow:
            m &= np.array([c.meta.get("workflow") in (workflow, "all") for c in self.chunks])
        if lang:
            m &= np.array([c.meta.get("lang", "en") in (lang, "en", "hi-en") for c in self.chunks])
        return m

    def bm25(self, query: str) -> np.ndarray:
        q = normalise(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
            scores[i] = s
        return scores

    def dense(self, query: str) -> np.ndarray:
        return (self.vectors @ self.embed_fn([query])[0]).astype(np.float32)

    def search(self, query: str, k: int = 3, workflow: str | None = None,
               lang: str | None = None, rrf_k: int = 60) -> list[tuple[Chunk, float, dict]]:
        mask = self._mask(workflow, lang)
        lex, den = self.bm25(query), self.dense(query)
        lex[~mask] = -1
        den[~mask] = -1
        fused = np.zeros(len(self.chunks), dtype=np.float32)
        for scores in (lex, den):
            order = np.argsort(-scores)
            for rank, idx in enumerate(order):
                if scores[idx] > 0:
                    fused[idx] += 1.0 / (rrf_k + rank + 1)
        top = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i]),
                 {"bm25": float(lex[i]), "dense": float(den[i])})
                for i in top if fused[i] > 0]
