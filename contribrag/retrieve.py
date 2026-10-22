"""Hybrid retrieval: vector search + BM25 fused with reciprocal rank fusion,
then an optional reranking stage (cross-encoder when fastembed is present,
lexical fallback otherwise). Supports multi-query search for HyDE expansion.
"""
from __future__ import annotations

from .embeddings import get_embedder, get_reranker, tokenize
from .models import Chunk
from .store import Store


def _tokens(text: str) -> list[str]:
    return tokenize(text)


class Retriever:
    def __init__(self, store: Store, embedder=None, reranker=None, use_rerank: bool = True):
        self.store = store
        self.embedder = embedder or get_embedder()
        self.use_rerank = use_rerank
        self._reranker = reranker
        self._bm25 = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_repo: str | None = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def _ensure_bm25(self, repo: str | None):
        if self._bm25 is not None and self._bm25_repo == repo:
            return
        from rank_bm25 import BM25Okapi
        self._bm25_chunks = self.store.all_chunks(repo)
        corpus = [_tokens(c.embed_text()) for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_repo = repo

    def _fuse(self, query: str, repo: str | None) -> list[tuple[Chunk, float]]:
        vec = self.embedder.embed([query])[0]
        dense = self.store.search(vec, limit=40, repo=repo)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for rank, (ch, _) in enumerate(dense):
            chunks[ch.id] = ch
            scores[ch.id] = scores.get(ch.id, 0.0) + 1.0 / (60 + rank)
        self._ensure_bm25(repo)
        if self._bm25 is not None:
            bm = self._bm25.get_scores(_tokens(query))
            order = sorted(range(len(bm)), key=lambda i: -bm[i])[:40]
            for rank, i in enumerate(order):
                ch = self._bm25_chunks[i]
                chunks[ch.id] = ch
                scores[ch.id] = scores.get(ch.id, 0.0) + 1.0 / (60 + rank)
        q = query.lower()
        for cid, ch in chunks.items():
            if ch.symbol and ch.symbol.split(".")[-1].lower() in q:
                scores[cid] += 0.02
            if ch.path.lower() in q:
                scores[cid] += 0.02
        return sorted(((c, scores[c.id]) for c in chunks.values()),
                      key=lambda t: -t[1])

    def search(self, query: str, repo: str | None = None, k: int = 8,
               kinds: tuple[str, ...] | None = None) -> list[tuple[Chunk, float]]:
        return self.search_multi([query], repo=repo, k=k, kinds=kinds)

    def search_multi(self, queries: list[str], repo: str | None = None, k: int = 8,
                     kinds: tuple[str, ...] | None = None) -> list[tuple[Chunk, float]]:
        merged: dict[str, tuple[Chunk, float]] = {}
        for q in queries:
            for ch, sc in self._fuse(q, repo):
                if ch.id not in merged or sc > merged[ch.id][1]:
                    merged[ch.id] = (ch, sc)
        ranked = sorted(merged.values(), key=lambda t: -t[1])
        if kinds:
            ranked = [t for t in ranked if t[0].kind in kinds]
        candidates = ranked[:30]
        if self.use_rerank and len(candidates) > k:
            # Blend, never replace: the fusion score carries the symbol/path
            # boost that nails "where is X defined", while the cross-encoder
            # carries semantic relevance. Pure rerank ordering measurably
            # hurts definition queries (see eval ablation in README).
            rs = self.reranker.rerank(queries[0],
                                      [c.embed_text() for c, _ in candidates])
            fs = [sc for _, sc in candidates]

            def norm(vals):
                lo, hi = min(vals), max(vals)
                return [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in vals]

            blended = [(candidates[i][0], 0.5 * a + 0.5 * b)
                       for i, (a, b) in enumerate(zip(norm(fs), norm(rs)))]
            ranked = sorted(blended, key=lambda t: -t[1])
        else:
            ranked = candidates
        return ranked[:k]
