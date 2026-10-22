from pathlib import Path

from contribrag.chunking import chunk_file
from contribrag.models import Issue
from contribrag.qa import verify_citations
from contribrag import github_api
from contribrag.embeddings import HashingEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"


def test_ast_chunks_have_symbols_and_lines():
    chunks = chunk_file("sample", FIXTURE, FIXTURE / "src" / "calc.py")
    symbols = {c.symbol for c in chunks}
    assert "divide" in symbols and "add" in symbols
    assert "Calculator" in symbols and "Calculator.multiply" in symbols
    div = next(c for c in chunks if c.symbol == "divide")
    lines = (FIXTURE / "src" / "calc.py").read_text().splitlines()
    assert "def divide" in lines[div.start_line - 1]
    assert div.citation == f"src/calc.py:{div.start_line}-{div.end_line}"


def test_doc_chunks_use_headings():
    chunks = chunk_file("sample", FIXTURE, FIXTURE / "README.md")
    assert any(c.symbol == "Setup" for c in chunks)


def test_verify_citations():
    from contribrag.models import Chunk
    chunks = [Chunk("r", "a.py", "code", "f", 1, 3, "x"),
              Chunk("r", "b.py", "code", "g", 1, 3, "y")]
    assert verify_citations("uses f [1] and g [2]", chunks)["ok"]
    bad = verify_citations("invents [7]", chunks)
    assert not bad["ok"] and bad["invalid"] == [7]
    assert not verify_citations("no citations at all", chunks)["ok"]


def test_rank_issues_prefers_beginner_labels():
    issues = [
        Issue(1, "Refactor the whole core engine", "deep internals work", []),
        Issue(2, "Fix typo in README quickstart", "docs typo", ["good first issue"]),
        Issue(3, "Add python quickstart example", "python docs example", ["help wanted"]),
    ]
    ranked = github_api.rank_issues(issues, HashingEmbedder(), "python documentation")
    assert ranked[0]["issue"].number in (2, 3)
