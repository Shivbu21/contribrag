"""Repository ingestion: clone a GitHub repo or take a local path, walk files."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import settings

CODE_EXTS = {".py": "python", ".js": "javascript", ".ts": "javascript",
             ".jsx": "javascript", ".tsx": "javascript"}
DOC_EXTS = {".md", ".rst", ".txt"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
             "build", ".next", ".idea", ".tox", "vendor", ".contribrag"}
SKIP_SUFFIXES = (".min.js", ".map", ".lock")
GITHUB_RE = re.compile(r"(?:github\.com[:/])([\w.-]+)/([\w.-]+?)(?:\.git)?$")


def parse_github_url(url: str) -> tuple[str, str] | None:
    m = GITHUB_RE.search(url.strip())
    return (m.group(1), m.group(2)) if m else None


def obtain_repo(source: str, dest_root: Path) -> tuple[Path, str]:
    """Return (local_path, repo_label). Clones GitHub URLs (shallow); copies local dirs."""
    parsed = parse_github_url(source)
    if parsed:
        owner, name = parsed
        label = f"{owner}/{name}"
        dest = dest_root / owner / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/{owner}/{name}.git"
            if settings.github_token:  # private repos via token auth
                url = f"https://x-access-token:{settings.github_token}@github.com/{owner}/{name}.git"
            r = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(
                    f"git clone failed for {label}. If the repo is private, set "
                    f"GITHUB_TOKEN (token clone) or configure git credentials. "
                    f"git said: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown error'}")
        return dest, label
    src = Path(source).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory or GitHub URL: {source}")
    label = src.name
    dest = dest_root / "local" / label
    if dest.resolve() != src:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*SKIP_DIRS))
    return dest, label


def walk_files(root: Path) -> tuple[list[Path], dict]:
    """Collect ingestible files; return (files, stats) — stats make caps visible."""
    files: list[Path] = []
    stats = {"skipped_large": 0, "skipped_binary": 0, "capped": False}
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file() or path.name.endswith(SKIP_SUFFIXES):
            continue
        ext = path.suffix.lower()
        if ext not in CODE_EXTS and ext not in DOC_EXTS:
            continue
        if path.stat().st_size > settings.max_file_bytes:
            stats["skipped_large"] += 1
            continue
        try:
            path.read_bytes()[:512].decode("utf-8")
        except (UnicodeDecodeError, OSError):
            stats["skipped_binary"] += 1
            continue
        files.append(path)
        if len(files) >= settings.max_files:
            stats["capped"] = True
            break
    return files, stats
