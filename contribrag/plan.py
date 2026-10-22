"""PR plan generator: for a chosen issue, a grounded plan — files to read
first, likely change points, tests to run — every step backed by retrieved
chunks with citations. Works in extractive mode too (plan skeleton from
evidence), and upgrades to prose when an LLM is configured."""
from __future__ import annotations

from .llm import chat, context_block
from .models import Issue
from .retrieve import Retriever


def build_plan(retriever: Retriever, issue: Issue, repo: str | None = None) -> dict:
    query = f"{issue.title}\n{issue.body[:1200]}"
    hits = retriever.search(query, repo=repo, k=10, kinds=("code", "doc"))
    chunks = [c for c, _ in hits]
    code = [c for c in chunks if c.kind == "code"]
    docs = [c for c in chunks if c.kind == "doc"]
    tests = [c for c in code if "test" in c.path.lower()]
    steps = [
        {"step": "Read the issue fully", "detail": issue.title,
         "cite": f"issue #{issue.number}"},
    ]
    for ch in code[:4]:
        steps.append({"step": f"Read {ch.symbol}", "detail": ch.citation, "cite": ch.citation})
    for ch in docs[:2]:
        steps.append({"step": f"Check docs: {ch.symbol}", "detail": ch.citation, "cite": ch.citation})
    steps.append({"step": "Reproduce / write a failing test",
                  "detail": "Confirm the behaviour the issue describes before changing code.",
                  "cite": tests[0].citation if tests else "no test files retrieved"})
    steps.append({"step": "Make the smallest change that fixes it",
                  "detail": "Touch only the functions implicated above; keep the diff reviewable.",
                  "cite": code[0].citation if code else "—"})
    steps.append({"step": "Run the relevant tests",
                  "detail": (f"Start with {tests[0].path}, then the module's test suite."
                             if tests else "Locate the test suite (tests/ or *_test.py) and run it."),
                  "cite": tests[0].citation if tests else "—"})
    plan = {"issue": {"number": issue.number, "title": issue.title, "url": issue.url},
            "steps": steps, "evidence": [c.citation for c in chunks]}
    prose = chat([
        {"role": "system", "content": "You write short, concrete contribution plans. "
         "Use ONLY the provided context; cite blocks like [1]. No invented files."},
        {"role": "user", "content": f"Issue #{issue.number}: {issue.title}\n{issue.body[:1500]}\n\n"
         f"Context:\n{context_block(chunks)}\n\nWrite a 5-step plan: what to read, "
         "where the change likely goes, which tests to run, and one risk to watch."},
    ])
    if prose:
        plan["prose"] = prose
    return plan
