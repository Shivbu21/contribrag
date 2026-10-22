"""Indexing pipeline: obtain repo → chunk files (+ issues) → embed → store."""
from __future__ import annotations

from pathlib import Path

from . import github_api
from .chunking import chunk_file
from .config import settings
from .embeddings import get_embedder
from .ingest import obtain_repo, walk_files
from .models import Chunk
from .store import Store


def index_repo(source: str, with_issues: bool = True, store: Store | None = None,
               embedder=None) -> dict:
    store = store or Store()
    embedder = embedder or get_embedder()
    root = settings.data_dir / "repos"
    root.mkdir(parents=True, exist_ok=True)
    repo_path, label = obtain_repo(source, root)
    files, stats = walk_files(repo_path)
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_file(label, repo_path, f))
    issue_count = 0
    if with_issues:
        issues = github_api.fetch_issues(source)
        for iss in issues:
            chunks.append(iss.to_chunk(label))
            issue_count += 1
    if chunks:
        store.ensure(len(embedder.embed(["dim probe"])[0]))
        batch = 128
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            store.upsert(part, embedder.embed([c.text for c in part]))
    return {"repo": label, "files": len(files), "chunks": len(chunks),
            "issues": issue_count, "embedder": getattr(embedder, "name", "?"),
            "stats": stats}
