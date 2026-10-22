# ContribRAG

**Ask a repo. Then contribute to it.**

Paste a GitHub URL → ContribRAG ingests the code, docs and issues → you ask
questions and get answers with **file:line citations that are verified before
you see them** → Contribute mode matches you to good-first-issues and drafts a
grounded PR plan.

Built for new open-source contributors, and for anyone who has ever opened a
large repo and thought *where do I even start?*

## Why it's different

- **AST chunking, not text windows.** Code is chunked with tree-sitter at
  function/class level, so citations point at real symbols with exact line
  spans (`src/click/core.py:412-489`), not arbitrary 500-word slices.
- **Citations are enforced, not hoped for.** Every generated answer passes a
  verifier: each `[n]` marker must map to a retrieved chunk. Fail → one retry
  → fall back to the grounded extractive answer. Unverified text never ships.
- **Contribute mode.** Open issues are embedded and ranked against *your*
  skills, beginner-friendly labels first, with a "why this fits you" for each.
- **PR plans from evidence.** For a chosen issue: what to read first, where
  the change likely goes, which tests to run — every step cited.
- **Measurable.** An eval harness with golden questions reports hit@1, hit@k
  and citation validity per repo. Retrieval quality is a number, not a vibe.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Index a repo (clones it, chunks code+docs, pulls open issues)
.venv/bin/python -m contribrag.cli ingest https://github.com/pallets/click

# Ask it questions — answers carry verified citations
.venv/bin/python -m contribrag.cli ask "How does click parse command options?"

# Contribute mode: get matched to issues
.venv/bin/python -m contribrag.cli match https://github.com/pallets/click \
    --skills "python, cli, testing"

# A grounded PR plan for one issue
.venv/bin/python -m contribrag.cli plan https://github.com/pallets/click 1234

# API + UI
.venv/bin/uvicorn contribrag.api:app --port 8000
.venv/bin/streamlit run app.py
```

## Modes

| Mode | Needs | Behaviour |
|---|---|---|
| Extractive (default) | nothing | Answers composed from retrieved chunks with citations. Fully offline. |
| LLM | `CONTRIBRAG_LLM_BASE_URL` (any OpenAI-compatible endpoint, e.g. Ollama) | Generated prose answers, still citation-verified. |
| Neural embeddings | `pip install fastembed` | bge-small embeddings replace the built-in hashing fallback. |

## Eval

```bash
.venv/bin/python -m contribrag.eval eval/golden_sample.json
# → hit@1, hit@5, citation_validity per golden set (fixture repo included)
```

Current fixture numbers (offline hashing embedder): **hit@1 0.80, hit@5 1.00,
citation validity 1.00** over 5 golden questions.

## Architecture

```
GitHub URL ─► ingest (git clone / local copy, file caps are reported, not silent)
           ─► chunking (tree-sitter AST for .py/.js/.ts, heading chunks for docs,
                        issues/PRs as chunks)
           ─► index (Qdrant local mode + embeddings)
           ─► hybrid retrieval (dense + BM25, reciprocal-rank fusion, symbol boost)
           ─► LangGraph agent ─► ask | setup | match-issues | plan
           ─► citation verifier (retry once → extractive fallback)
```

## Honest limits (v1)

- Ingest caps: 3,000 files / 400 KB per file by default; caps are reported in
  the ingest stats, never silently applied. Huge monorepos need the caps raised.
- Private repos: set `GITHUB_TOKEN` (used for the API and token clone).
- The citation verifier checks that cited blocks *exist and were retrieved*;
  judging whether a chunk fully supports a claim is the eval harness's job
  (citation precision on golden sets), not the runtime verifier's.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q   # 5 tests: chunking, citations, matcher, e2e
```

## License

MIT
