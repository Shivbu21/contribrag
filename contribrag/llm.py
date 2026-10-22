"""LLM access (any OpenAI-compatible endpoint) + extractive fallback.

If no LLM is configured, ContribRAG still answers: it composes a grounded
response from the retrieved chunks themselves, each with its citation. That
mode is also what the eval harness uses when no endpoint is available, so
retrieval quality is measurable with or without a model.
"""
from __future__ import annotations

import requests

from .config import settings
from .models import Chunk


def chat(messages: list[dict], temperature: float = 0.1) -> str | None:
    if not settings.llm_enabled:
        return None
    try:
        r = requests.post(f"{settings.llm_base_url}/chat/completions",
                          json={"model": settings.llm_model, "messages": messages,
                                "temperature": temperature},
                          headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                          timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def context_block(chunks: list[Chunk]) -> str:
    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[{i}] {ch.citation} ({ch.kind}: {ch.symbol})\n{ch.text}")
    return "\n\n---\n\n".join(parts)


def extractive_answer(question: str, chunks: list[Chunk]) -> str:
    if not chunks:
        return "Not in the repo: no indexed content matched that question."
    lines = [f"Based on the indexed repo content, the most relevant pieces for "
             f"“{question}” are:\n"]
    for ch in chunks[:5]:
        first = next((ln.strip() for ln in ch.text.splitlines() if ln.strip()), "")
        lines.append(f"- **{ch.symbol}** [{ch.citation}] — {first[:160]}")
    lines.append("\n(Extractive mode: no LLM endpoint configured, so this is the "
                 "grounded evidence itself rather than a generated answer.)")
    return "\n".join(lines)
