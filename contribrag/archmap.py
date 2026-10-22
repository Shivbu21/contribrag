"""Architecture map: a structural digest of an indexed repo built purely
from chunk metadata (paths, symbols) + import lines inside code chunks.
Powers the 'explain the architecture' route without inventing anything."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .models import Chunk
from .store import Store

_IMPORT = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", re.M)


def build_map(chunks: list[Chunk]) -> dict:
    files: dict[str, dict] = {}
    kinds = Counter()
    for ch in chunks:
        kinds[ch.kind] += 1
        if ch.kind != "code":
            continue
        f = files.setdefault(ch.path, {"symbols": [], "imports": set(), "lines": 0})
        if ch.symbol and not ch.symbol.endswith("(module level)"):
            f["symbols"].append(ch.symbol)
        f["lines"] = max(f["lines"], ch.end_line)
        for m in _IMPORT.findall(ch.text):
            f["imports"].add(m.split(".")[0])
    dirs = Counter()
    for path in files:
        dirs[path.split("/")[0] if "/" in path else "(root)"] += 1
    entry = [p for p in files if any(k in p.rsplit("/", 1)[-1]
             for k in ("__main__", "main.", "cli.", "app.", "__init__"))]
    return {"files": {p: {"symbols": v["symbols"][:12],
                          "imports": sorted(v["imports"])[:12],
                          "lines": v["lines"]} for p, v in files.items()},
            "kinds": dict(kinds), "dirs": dict(dirs.most_common(10)),
            "entry_points": entry[:8],
            "top_files": sorted(files, key=lambda p: -len(files[p]["symbols"]))[:8]}


def summary_text(repo: str, amap: dict) -> str:
    lines = [f"Architecture map for {repo} "
             f"({amap['kinds'].get('code', 0)} code chunks, "
             f"{amap['kinds'].get('doc', 0)} doc chunks, "
             f"{amap['kinds'].get('issue', 0)} issues, "
             f"{amap['kinds'].get('pr', 0)} PRs indexed).", ""]
    lines.append("Top-level areas (by file count):")
    for d, n in amap["dirs"].items():
        lines.append(f"- {d}: {n} files")
    if amap["entry_points"]:
        lines.append("\nEntry points: " + ", ".join(amap["entry_points"]))
    lines.append("\nHeaviest modules (most symbols):")
    for p in amap["top_files"]:
        info = amap["files"][p]
        syms = ", ".join(info["symbols"][:5])
        lines.append(f"- {p} (~{info['lines']} lines): {syms}")
    lines.append("\n(Structural digest from the index itself — ask a specific "
                 "question for cited detail on any module.)")
    return "\n".join(lines)


def architecture_answer(store: Store, repo: str | None) -> dict:
    chunks = store.all_chunks(repo)
    if not chunks:
        return {"answer": "Nothing is indexed yet for that scope.",
                "chunks": [], "mode": "archmap",
                "verification": {"ok": False, "cited": [], "valid": [], "invalid": []}}
    label = repo or chunks[0].repo
    amap = build_map(chunks)
    docs = [c for c in chunks if c.kind == "doc" and "readme" in c.path.lower()][:3]
    return {"answer": summary_text(label, amap), "chunks": docs,
            "mode": "archmap", "map": amap,
            "verification": {"ok": True, "cited": [], "valid": [], "invalid": []}}
