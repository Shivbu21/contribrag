import tempfile
from pathlib import Path

from contribrag.embeddings import HashingEmbedder
from contribrag.pipeline import index_repo
from contribrag.qa import answer_question
from contribrag.retrieve import Retriever
from contribrag.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"


def test_end_to_end_ask():
    store = Store(Path(tempfile.mkdtemp()) / "q")
    emb = HashingEmbedder()
    info = index_repo(str(FIXTURE), with_issues=False, store=store, embedder=emb)
    assert info["chunks"] >= 6
    retriever = Retriever(store, emb)
    out = answer_question(retriever, "Where is division by zero handled?",
                          repo=info["repo"])
    assert out["chunks"], "expected retrieval hits"
    top = out["chunks"][0]
    assert top.path == "src/calc.py" and "divide" in top.symbol
    assert out["verification"]["ok"]
