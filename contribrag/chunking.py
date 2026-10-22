"""Chunking: tree-sitter AST chunks for code, heading/paragraph chunks for docs.

Code chunks are function/class level so citations land on real symbols with
exact line spans — the core differentiator vs naive fixed-window splitting.
Falls back to line-window chunking if tree-sitter is unavailable.
"""
from __future__ import annotations

from pathlib import Path

from .models import Chunk

MAX_CHUNK_CHARS = 6000
OVERLAP_LINES = 10

try:
    import tree_sitter_python as tsp
    import tree_sitter_javascript as tsj
    from tree_sitter import Language, Parser
    _PARSERS = {
        "python": Parser(Language(tsp.language())),
        "javascript": Parser(Language(tsj.language())),
    }
except Exception:  # pragma: no cover - exercised only without the deps
    _PARSERS = {}


def _window_chunks(repo: str, path: str, kind: str, language: str,
                   lines: list[str], symbol_prefix: str = "") -> list[Chunk]:
    chunks, i, n = [], 0, len(lines)
    win = 60
    while i < n:
        j = min(n, i + win)
        text = "\n".join(lines[i:j]).strip()
        if text:
            chunks.append(Chunk(repo=repo, path=path, kind=kind,
                                symbol=symbol_prefix or path,
                                start_line=i + 1, end_line=j,
                                text=text[:MAX_CHUNK_CHARS], language=language))
        i = j - OVERLAP_LINES if j < n else j
    return chunks


def _ast_chunks(repo: str, path: str, language: str, source: bytes,
                lines: list[str]) -> list[Chunk] | None:
    parser = _PARSERS.get(language)
    if parser is None:
        return None
    tree = parser.parse(source)
    chunks: list[Chunk] = []
    defs = {"function_definition", "class_definition", "function_declaration",
            "class_declaration", "method_definition", "arrow_function"}
    covered: set[int] = set()

    def visit(node, parents: list[str]):
        if node.type in defs:
            name_node = node.child_by_field_name("name")
            name = (name_node.text.decode() if name_node else "anonymous")
            symbol = ".".join(parents + [name]) if parents else name
            start, end = node.start_point[0] + 1, node.end_point[0] + 1
            text = "\n".join(lines[start - 1:end])
            if len(text) > MAX_CHUNK_CHARS:  # huge body: window it, keep symbol
                sub = _window_chunks(repo, path, "code", language,
                                     lines[start - 1:end], symbol_prefix=symbol)
                for c in sub:
                    c.start_line += start - 1
                    c.end_line += start - 1
                chunks.extend(sub)
            else:
                chunks.append(Chunk(repo=repo, path=path, kind="code", symbol=symbol,
                                    start_line=start, end_line=end,
                                    text=text, language=language))
            covered.update(range(start, end + 1))
            for child in node.children:
                visit(child, parents + [name])
            return
        for child in node.children:
            visit(child, parents)

    visit(tree.root_node, [])
    # Module-level leftovers (imports, constants, script code) in windows,
    # skipping lines already covered by a definition chunk.
    leftover = [(i + 1, ln) for i, ln in enumerate(lines) if (i + 1) not in covered]
    runs: list[list[tuple[int, str]]] = []
    for item in leftover:
        if runs and item[0] == runs[-1][-1][0] + 1:
            runs[-1].append(item)
        elif item[1].strip():
            runs.append([item])
    for run in runs:
        text = "\n".join(ln for _, ln in run).strip()
        if len(text) >= 40:
            chunks.append(Chunk(repo=repo, path=path, kind="code",
                                symbol=f"{path} (module level)",
                                start_line=run[0][0], end_line=run[-1][0],
                                text=text[:MAX_CHUNK_CHARS], language=language))
    return chunks


def chunk_file(repo: str, root: Path, file: Path) -> list[Chunk]:
    rel = str(file.relative_to(root))
    raw = file.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    ext = file.suffix.lower()
    from .ingest import CODE_EXTS, DOC_EXTS
    if ext in CODE_EXTS:
        ast = _ast_chunks(repo, rel, CODE_EXTS[ext], raw, lines)
        if ast:
            return ast
        return _window_chunks(repo, rel, "code", CODE_EXTS[ext], lines)
    if ext in DOC_EXTS:
        return _doc_chunks(repo, rel, lines)
    return []


def _doc_chunks(repo: str, path: str, lines: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading, buf, start = path, [], 1
    def flush(end: int):
        text = "\n".join(buf).strip()
        if len(text) >= 40:
            chunks.append(Chunk(repo=repo, path=path, kind="doc", symbol=heading,
                                start_line=start, end_line=end,
                                text=text[:MAX_CHUNK_CHARS], language="markdown"))
    for i, ln in enumerate(lines, start=1):
        if ln.startswith("#") and buf:
            flush(i - 1)
            heading, buf, start = ln.lstrip("# ").strip() or path, [ln], i
        else:
            if ln.startswith("#") and not buf:
                heading, start = ln.lstrip("# ").strip() or path, i
            buf.append(ln)
    flush(len(lines))
    return chunks
