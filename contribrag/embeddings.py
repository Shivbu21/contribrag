"""Embeddings: fastembed (neural, local) when installed; otherwise a
deterministic hashing embedder so the full pipeline runs offline with zero
heavy dependencies. The interface is the same either way."""
from __future__ import annotations

import hashlib
import math
import re

from .config import settings

_TOKEN = re.compile(r"[a-z0-9_]+")


def _stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lowercase tokens with snake_case split and naive stemming — shared by
    the hashing embedder and BM25 so 'passwords hashed' matches hash_password."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        for part in tok.split("_"):
            if part:
                out.append(_stem(part))
    return out


class HashingEmbedder:
    """Bag-of-words hashing vectors, L2-normalised. Not semantic — a
    dependable offline fallback and the embedder used in tests/CI."""

    name = "hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
                # a second hash for bigrams-ish spread
                vec[(h >> 8) % self.dim] += 0.5
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FastEmbedder:
    name = "fastembed"

    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model)
        self.dim = settings.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


def get_embedder():
    try:
        return FastEmbedder(settings.embed_model)
    except Exception:
        return HashingEmbedder(settings.embed_dim)
