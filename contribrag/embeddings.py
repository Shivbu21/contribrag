"""Embeddings: fastembed (neural, local) when installed; otherwise a
deterministic hashing embedder so the full pipeline runs offline with zero
heavy dependencies. The interface is the same either way."""
from __future__ import annotations

import hashlib
import math
import os
import re
from contextlib import contextmanager

from .config import settings


@contextmanager
def _sanitized_proxy_env():
    """Some environments ship no_proxy entries like '[::1]' that httpx cannot
    parse, which breaks model downloads. Drop bracketed entries just for the
    download/init call; inference afterwards is fully local."""
    saved = {}
    for key in ("no_proxy", "NO_PROXY"):
        if key in os.environ:
            saved[key] = os.environ[key]
            os.environ[key] = ",".join(
                e for e in os.environ[key].split(",") if "[" not in e)
    try:
        yield
    finally:
        for key, val in saved.items():
            os.environ[key] = val

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
        with _sanitized_proxy_env():
            self._model = TextEmbedding(model_name=model)
        self.dim = settings.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        # parallel=0 + small slices: fastembed's parallel workers each load a
        # copy of the model, which OOMs small machines on big repos.
        out = []
        for i in range(0, len(texts), 64):
            out.extend(list(map(float, v)) for v in
                       self._model.embed(texts[i:i + 64], parallel=0))
        return out


def get_embedder():
    try:
        return FastEmbedder(settings.embed_model)
    except Exception:
        return HashingEmbedder(settings.embed_dim)


# --- Reranking ---------------------------------------------------------------

class CrossEncoderReranker:
    """Neural cross-encoder reranker via fastembed (local ONNX)."""

    name = "cross-encoder"

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        with _sanitized_proxy_env():
            self._model = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


class LexicalReranker:
    """Offline fallback reranker: stemmed token overlap, symbol-aware."""

    name = "lexical"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        out = []
        for t in texts:
            toks = tokenize(t)
            if not toks:
                out.append(0.0)
                continue
            overlap = sum(1 for tok in toks if tok in q)
            out.append(overlap / (len(toks) ** 0.5))
        return out


def get_reranker():
    try:
        return CrossEncoderReranker(settings.rerank_model)
    except Exception:
        return LexicalReranker()
