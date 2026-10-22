"""Minimal MCP server (stdio, JSON-RPC) exposing ContribRAG as tools.

Run: python -m contribrag.mcp_server   (point any MCP client at it)
Tools: ask_repo(question, repo?), match_issues(source, skills), repo_stats()
Implemented directly on the MCP wire format — no extra dependency.
"""
from __future__ import annotations

import json
import sys

from .agent import ContribAgent
from .store import Store

PROTOCOL = "2024-11-05"
TOOLS = [
    {"name": "ask_repo",
     "description": "Ask a question about an indexed repository; the answer carries verified file:line citations.",
     "inputSchema": {"type": "object", "properties": {
         "question": {"type": "string"},
         "repo": {"type": "string", "description": "owner/name label, optional"}},
         "required": ["question"]}},
    {"name": "match_issues",
     "description": "Rank a GitHub repo's open issues for a contributor with the given skills (contribute mode).",
     "inputSchema": {"type": "object", "properties": {
         "source": {"type": "string"},
         "skills": {"type": "string"}},
         "required": ["source"]}},
    {"name": "repo_stats",
     "description": "Chunk counts per indexed repository.",
     "inputSchema": {"type": "object", "properties": {}}},
]

_agent: ContribAgent | None = None


def agent() -> ContribAgent:
    global _agent
    if _agent is None:
        _agent = ContribAgent()
    return _agent


def call_tool(name: str, args: dict) -> str:
    if name == "ask_repo":
        out = agent().run(question=args["question"], repo=args.get("repo"))
        srcs = "\n".join(f"[{c.citation}] {c.symbol}" for c in out.get("chunks", []))
        return f"{out['answer']}\n\nSources:\n{srcs}"
    if name == "match_issues":
        out = agent().run(question="which issue should I contribute to",
                          source=args["source"],
                          skills=args.get("skills", "python, llm, rag"))
        return "\n".join(f"#{m['number']} {m['title']} (score {m['score']}) — {m['why']}"
                         for m in out.get("matches", [])) or "No open issues matched."
    if name == "repo_stats":
        chunks = Store().all_chunks()
        by: dict[str, int] = {}
        for c in chunks:
            by[c.repo] = by.get(c.repo, 0) + 1
        return json.dumps({"total_chunks": len(chunks), "by_repo": by}, indent=2)
    raise ValueError(f"unknown tool {name}")


def handle(msg: dict) -> dict | None:
    method = msg.get("method", "")
    mid = msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "contribrag", "version": "0.2.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        try:
            text = call_tool(params.get("name", ""), params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as e:  # surface as a tool error, never crash the server
            result = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None  # notification


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line))
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": f"parse error: {e}"}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
