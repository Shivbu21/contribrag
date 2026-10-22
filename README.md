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
| LLM | `CONTRIBRAG_LLM_BASE_URL` (any OpenAI-compatible endpoint, e.g. Ollama) | Generated prose answers, still citation-verified; HyDE expansion on. |
| Embeddings | `fastembed` installed (default in requirements) | bge-small neural embeddings + cross-encoder rerank; without it, hashing embeddings + lexical rerank keep everything working offline. |

## Eval

```bash
.venv/bin/python -m contribrag.eval eval/golden_sample.json [--embedder fastembed] [--no-rerank]
```

Ablation (hit@1 / hit@5, citation validity 1.00 in every configuration):

| Config | Fixture (5 q) | pallets/click (7 q, 2,846 chunks) |
|---|---|---|
| Hashing embeddings, no rerank | 0.80 / 1.00 | 0.57 / 0.57 |
| Hashing + pure cross-encoder rerank | 1.00 / 1.00 | **0.29** / 1.00 |
| Hashing + **blended** rerank (shipped) | 1.00 / 1.00 | **0.71 / 1.00** |
| Neural (bge-small) + blended rerank | 1.00 / 1.00 | 0.57 / 1.00 |

The middle row is the interesting one: a web-trained cross-encoder, used as
the *only* ordering, promotes chunks that *discuss* a symbol over the chunk
that *defines* it. ContribRAG therefore **blends** the rerank score with the
fusion score (which carries a symbol/path boost) instead of replacing it.
Second lesson, same table: on definition-heavy questions the lexical side of
the hybrid beats neural embeddings at hit@1 — which is exactly why the
shipped pipeline fuses dense + BM25 + symbol signals instead of betting on
one model.

## Architecture

```
GitHub URL ─► ingest (git clone / local copy, file caps are reported, not silent)
           ─► chunking (tree-sitter AST for .py/.js/.ts, heading chunks for docs,
                        issues + merged PRs as chunks)
           ─► contextual enrichment (path :: symbol header embedded with each chunk)
           ─► index (Qdrant local mode; re-runs are incremental — stale chunks pruned)
           ─► hybrid retrieval (dense + BM25, reciprocal-rank fusion, symbol boost)
           ─► blended reranking (cross-encoder score ⋈ fusion score)
           ─► LangGraph agent ─► ask | setup | architecture | match-issues | plan
           ─► citation verifier (retry once → extractive fallback)
```

**v0.2 additions**

- **Neural embeddings** (fastembed bge-small) with the hashing embedder as an
  offline fallback; **cross-encoder reranking, blended** (see Eval).
- **HyDE query expansion** when an LLM endpoint is configured: a hypothetical
  answer passage is embedded alongside the question for retrieval.
- **Architecture route** — "explain the architecture" returns a structural map
  built from the index itself (modules, entry points, import edges).
- **PR memory** — merged PRs are indexed with their files-changed lists, so
  "how was X fixed before?" has something real to answer from.
- **MCP server** — `python -m contribrag.mcp_server` exposes `ask_repo`,
  `match_issues` and `repo_stats` to any MCP client over stdio.
- **Incremental re-index** — unchanged chunks keep their ids; chunks from
  deleted/renamed files are pruned and the count reported.

## Honest limits (v1)

- Ingest caps: 3,000 files / 400 KB per file by default; caps are reported in
  the ingest stats, never silently applied. Huge monorepos need the caps raised.
- Private repos: set `GITHUB_TOKEN` (used for the API and token clone).
- The citation verifier checks that cited blocks *exist and were retrieved*;
  judging whether a chunk fully supports a claim is the eval harness's job
  (citation precision on golden sets), not the runtime verifier's.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q   # 10 tests: chunking, citations, matcher, rerank, prune, archmap, MCP, e2e
```

## License

MIT
