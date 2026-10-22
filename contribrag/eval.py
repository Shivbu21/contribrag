"""Eval harness: golden questions with known-correct file/symbol targets.

Metrics:
- hit@1 / hit@k — the expected chunk is the top hit / within top-k retrieved
- citation validity — share of answers whose citations all verify
Run: python -m contribrag.eval eval/golden_sample.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from .embeddings import HashingEmbedder
from .pipeline import index_repo
from .qa import answer_question
from .retrieve import Retriever
from .store import Store


def run_eval(golden_path: str) -> dict:
    spec = json.loads(Path(golden_path).read_text())
    tmp = Path(tempfile.mkdtemp(prefix="contribrag-eval-"))
    store = Store(tmp / "qdrant")
    embedder = HashingEmbedder()
    info = index_repo(spec["source"], with_issues=False, store=store, embedder=embedder)
    retriever = Retriever(store, embedder)
    repo = info["repo"]
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
            "hit@1": round(hit1 / n, 3), "hit@5": round(hitk / n, 3),
            "citation_validity": round(valid / n, 3), "rows": rows}


if __name__ == "__main__":
    report = run_eval(sys.argv[1] if len(sys.argv) > 1 else "eval/golden_sample.json")
    print(json.dumps(report, indent=2))
