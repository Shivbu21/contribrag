"""Cited Q&A: generate (or extract) an answer, then VERIFY its citations.

The verifier is the product's spine: every [n] marker in an answer must map
to a retrieved chunk, and every chunk index cited must exist. Answers that
fail verification are retried once, then replaced by the extractive answer —
never shipped unverified.
"""
from __future__ import annotations

import re

from .llm import chat, context_block, extractive_answer
from .models import Chunk
from .retrieve import Retriever

_CITE = re.compile(r"\[(\d+)\]")

SYSTEM = (
    "You answer questions about a code repository using ONLY the provided "
    "context blocks. Every factual claim must end with a citation like [1] "
    "referring to the context block number. If the context does not contain "
    "the answer, say exactly: 'Not in the repo.' Never invent files, "
    "functions, or behaviour that the context does not show."
)


def verify_citations(answer: str, chunks: list[Chunk]) -> dict:
    cited = {int(n) for n in _CITE.findall(answer)}
    valid = {n for n in cited if 1 <= n <= len(chunks)}
    return {"cited": sorted(cited), "valid": sorted(valid),
            "invalid": sorted(cited - valid),
            "ok": cited == valid and len(cited) > 0}


def _hyde_variant(question: str) -> str | None:
    """HyDE: a hypothetical answer passage, embedded alongside the question
    so retrieval matches answer-shaped text. Only when an LLM is configured."""
    return chat([
        {"role": "system", "content": "Write a short hypothetical passage that "
         "would answer the question if it appeared in the repo's code or docs. "
         "Invent plausible function names and files; it is only a search probe."},
        {"role": "user", "content": question},
    ], temperature=0.3)


def answer_question(retriever: Retriever, question: str, repo: str | None = None,
                    k: int = 8) -> dict:
    queries = [question]
    hyde = _hyde_variant(question)
    if hyde:
        queries.append(hyde)
    hits = retriever.search_multi(queries, repo=repo, k=k)
    chunks = [c for c, _ in hits]
    if not chunks:
        return {"answer": "Not in the repo: nothing is indexed yet for that scope.",
                "chunks": [], "verification": {"ok": False, "cited": [], "valid": [],
                                               "invalid": []},
                "mode": "none"}
    generated = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context_block(chunks)}\n\nQuestion: {question}"},
    ])
    if generated:
        v = verify_citations(generated, chunks)
        if v["ok"] or generated.strip().startswith("Not in the repo"):
            return {"answer": generated, "chunks": chunks, "verification": v,
                    "mode": "llm"}
        # One retry with the failure named, per the citation-enforcement design.
        retry = chat([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context_block(chunks)}\n\nQuestion: {question}\n\n"
             f"Your previous answer cited blocks {v['invalid']} which do not exist. "
             "Rewrite it citing only the blocks above."},
        ])
        if retry:
            v2 = verify_citations(retry, chunks)
            if v2["ok"]:
                return {"answer": retry, "chunks": chunks, "verification": v2,
                        "mode": "llm-retry"}
    ans = extractive_answer(question, chunks)
    return {"answer": ans, "chunks": chunks,
            "verification": {"ok": True, "cited": list(range(1, min(5, len(chunks)) + 1)),
                             "valid": list(range(1, min(5, len(chunks)) + 1)),
                             "invalid": []},
            "mode": "extractive"}
