"""Runtime configuration, all overridable by environment variables."""
from __future__ import annotations

import os
from pathlib import Path


class Settings:
    # Storage
    data_dir: Path = Path(os.environ.get("CONTRIBRAG_DATA", ".contribrag"))
    # Embeddings: "fastembed" if installed, else deterministic hashing fallback
    embed_model: str = os.environ.get("CONTRIBRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = 384  # hashing embedder dimension (fastembed bge-small is also 384)
    # LLM (any OpenAI-compatible endpoint). If unset, extractive mode is used:
    # answers are composed from retrieved chunks with citations, no generation.
    llm_base_url: str | None = os.environ.get("CONTRIBRAG_LLM_BASE_URL")  # e.g. http://localhost:11434/v1
    llm_api_key: str = os.environ.get("CONTRIBRAG_LLM_API_KEY", "not-needed")
    llm_model: str = os.environ.get("CONTRIBRAG_LLM_MODEL", "qwen2.5:7b")
    # GitHub
    github_token: str | None = os.environ.get("GITHUB_TOKEN")
    # Ingest limits (v1 honesty: huge monorepos are capped, and we say so)
    max_files: int = int(os.environ.get("CONTRIBRAG_MAX_FILES", "3000"))
    max_file_bytes: int = int(os.environ.get("CONTRIBRAG_MAX_FILE_BYTES", "400000"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url)


settings = Settings()
