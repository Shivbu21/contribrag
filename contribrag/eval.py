"""Eval harness: golden questions with known-correct file/symbol targets.

Metrics:
- hit@1 / hit@k — the expected chunk is the top hit / within top-k retrieved
- citation validity — share of answers whose citations all verify

Supports ablation: --embedder hashing|fastembed, --rerank / --no-rerank.
Run: python -m contribrag.eval eval/golden_sample.json [--embedder fastembed] [--no-rerank]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .embeddings import HashingEmbedder, get_embedder
from .pipeline import index_repo
from .qa import answer_question
from .retrieve import Retriever
from .store import Store


def run_eval(golden_path: str, embedder=None, use_rerank: bool = True,
             with_issues: bool = False, store_path: str | None = None,
             index_only: bool = False) -> dict:
    spec = json.loads(Path(golden_path).read_text())
    embedder = embedder or HashingEmbedder()
    if store_path:
        store = Store(Path(store_path))
        if index_only:
            return {"indexed": index_repo(spec["source"], with_issues=with_issues,
                                          store=store, embedder=embedder)}
        repos = {c.repo for c in store.all_chunks()}
        repo = spec.get("repo") or (sorted(repos)[0] if repos else None)
        info = {"repo": repo, "chunks": store.count(repo)}
    else:
        tmp = Path(tempfile.mkdtemp(prefix="contribrag-eval-"))
        store = Store(tmp / "qdrant")
        info = index_repo(spec["source"], with_issues=with_issues, store=store,
                          embedder=embedder)
        repo = info["repo"]
    retriever = Retriever(store, embedder, use_rerank=use_rerank)
    rows, hit1, hitk, valid = [], 0, 0, 0
    for item in spec["questions"]:
        hits = retriever.search(item["q"], repo=repo, k=5)
        chunks = [c for c, _ in hits]
        def matches(c):
            return c.path == item["expect_path"] and item["expect_symbol"] in c.symbol
        top = bool(chunks) and matches(chunks[0])
        anyk = any(matches(c) for c in chunks)
        ans = answer_question(retriever, item["q"], repo=repo)
        ok = bool(ans["verification"]["ok"])
        hit1 += top; hitk += anyk; valid += ok
        rows.append({"q": item["q"], "hit@1": top, "hit@5": anyk,
                     "citations_valid": ok,
                     "top_hit": chunks[0].citation if chunks else "—"})
    n = len(spec["questions"]) or 1
    return {"repo": repo, "chunks": info["chunks"], "questions": n,
            "embedder": getattr(embedder, "name", "?"), "rerank": use_rerank,
            "reranker": getattr(retriever.reranker, "name", "?") if use_rerank else None,
            "hit@1": round(hit1 / n, 3), "hit@5": round(hitk / n, 3),
            "citation_validity": round(valid / n, 3), "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--embedder", choices=["hashing", "fastembed"], default="hashing")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--store", help="use/create this store dir instead of a temp one")
    ap.add_argument("--index-only", action="store_true",
                    help="only index into --store (phase 1 of a two-phase eval)")
    a = ap.parse_args()
    emb = HashingEmbedder() if a.embedder == "hashing" else get_embedder()
    report = run_eval(a.golden, embedder=emb, use_rerank=not a.no_rerank,
                      store_path=a.store, index_only=a.index_only)
    print(json.dumps(report, indent=2))
