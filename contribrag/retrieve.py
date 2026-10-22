"""Hybrid retrieval: vector search + BM25, fused with reciprocal rank fusion.
A light symbol/path boost helps 'where is X defined' questions land exactly."""
from __future__ import annotations

import re  # noqa: F401 (kept for potential regex helpers)

from .embeddings import get_embedder, tokenize
from .models import Chunk
from .store import Store


def _tokens(text: str) -> list[str]:
    return tokenize(text)


class Retriever:
    def __init__(self, store: Store, embedder=None):
        self.store = store
        self.embedder = embedder or get_embedder()
        self._bm25 = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_repo: str | None = None

    def _ensure_bm25(self, repo: str | None):
        if self._bm25 is not None and self._bm25_repo == repo:
            return
        from rank_bm25 import BM25Okapi
        self._bm25_chunks = self.store.all_chunks(repo)
        corpus = [_tokens(c.text + " " + c.symbol + " " + c.path)
                  for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_repo = repo

    def search(self, query: str, repo: str | None = None,
               k: int = 8, kinds: tuple[str, ...] | None = None) -> list[tuple[Chunk, float]]:
        vec = self.embedder.embed([query])[0]
        dense = self.store.search(vec, limit=max(30, k * 4), repo=repo)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for rank, (ch, _) in enumerate(dense):
            chunks[ch.id] = ch
            scores[ch.id] = scores.get(ch.id, 0.0) + 1.0 / (60 + rank)
        self._ensure_bm25(repo)
        if self._bm25 is not None:
            bm = self._bm25.get_scores(_tokens(query))
            order = sorted(range(len(bm)), key=lambda i: -bm[i])[:30]
            for rank, i in enumerate(order):
                ch = self._bm25_chunks[i]
                chunks[ch.id] = ch
                scores[ch.id] = scores.get(ch.id, 0.0) + 1.0 / (60 + rank)
        # Symbol/path boost: exact identifier mentions should win.
        q = query.lower()
        for cid, ch in chunks.items():
            if ch.symbol and ch.symbol.split(".")[-1].lower() in q:
                scores[cid] += 0.02
            if ch.path.lower() in q:
                scores[cid] += 0.02
        ranked = sorted(chunks.values(), key=lambda c: -scores[c.id])
        if kinds:
            ranked = [c for c in ranked if c.kind in kinds]
        return [(c, scores[c.id]) for c in ranked[:k]]
