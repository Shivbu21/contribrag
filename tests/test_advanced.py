import json
import tempfile
from pathlib import Path

from contribrag import mcp_server
from contribrag.agent import ContribAgent
from contribrag.archmap import architecture_answer, build_map
from contribrag.embeddings import HashingEmbedder, LexicalReranker
from contribrag.pipeline import index_repo
from contribrag.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"


def _fixture_store():
    store = Store(Path(tempfile.mkdtemp()) / "q")
    emb = HashingEmbedder()
    info = index_repo(str(FIXTURE), with_issues=False, store=store, embedder=emb)
    return store, emb, info


def test_embed_text_carries_symbol_header():
    store, emb, info = _fixture_store()
    ch = next(c for c in store.all_chunks(info["repo"]) if c.symbol == "divide")
    assert ch.embed_text().startswith("src/calc.py :: divide")


def test_lexical_reranker_prefers_relevant():
    r = LexicalReranker()
    scores = r.rerank("how are passwords hashed",
                      ["src/calc.py :: divide\ndivision by zero",
                       "src/auth.py :: hash_password\nhash a password with sha256"])
    assert scores[1] > scores[0]


def test_prune_removes_stale_chunks():
    store, emb, info = _fixture_store()
    repo = info["repo"]
    allc = store.all_chunks(repo)
    keep = {c.id for c in allc if c.path != "src/auth.py"}
    removed = store.prune(repo, keep)
    assert removed == len(allc) - len(keep)
    assert all(c.path != "src/auth.py" for c in store.all_chunks(repo))


def test_archmap_and_agent_route():
    store, emb, info = _fixture_store()
    amap = build_map(store.all_chunks(info["repo"]))
    assert "src/calc.py" in amap["files"]
    assert "divide" in amap["files"]["src/calc.py"]["symbols"]
    out = architecture_answer(store, info["repo"])
    assert out["mode"] == "archmap" and "calc" in out["answer"]
    agent = ContribAgent(store=store, embedder=emb)
    routed = agent.run(question="explain the architecture of this repo",
                       repo=info["repo"])
    assert routed["mode"] == "archmap"


def test_mcp_protocol():
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["name"] == "contribrag"
    tools = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in tools["result"]["tools"]}
    assert names == {"ask_repo", "match_issues", "repo_stats"}
    call = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                              "params": {"name": "nope", "arguments": {}}})
    assert call["result"]["isError"] is True
