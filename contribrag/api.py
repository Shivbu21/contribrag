"""FastAPI service: uvicorn contribrag.api:app --port 8000"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import ContribAgent
from .pipeline import index_repo

app = FastAPI(title="ContribRAG", version="0.1.0")
_agent: ContribAgent | None = None


def agent() -> ContribAgent:
    global _agent
    if _agent is None:
        _agent = ContribAgent()
    return _agent


class IngestReq(BaseModel):
    source: str
    with_issues: bool = True


class AskReq(BaseModel):
    question: str
    repo: str | None = None


class MatchReq(BaseModel):
    source: str
    skills: str = "python, llm, rag, agents"


class PlanReq(BaseModel):
    source: str
    issue: int
    repo: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
def ingest(req: IngestReq):
    return index_repo(req.source, with_issues=req.with_issues)


@app.post("/ask")
def ask(req: AskReq):
    out = agent().run(question=req.question, repo=req.repo)
    return {"answer": out["answer"], "mode": out["mode"],
            "verification": out["verification"],
            "sources": [c.citation for c in out["chunks"]]}


@app.post("/match")
def match(req: MatchReq):
    return agent().run(question="which issue should I contribute to",
                       source=req.source, skills=req.skills)


@app.post("/plan")
def plan(req: PlanReq):
    return agent().run(source=req.source, issue_number=req.issue, repo=req.repo)
