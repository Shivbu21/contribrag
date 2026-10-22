"""LangGraph agent: routes a request to ask / match-issues / plan / setup.

State graph: route → (retrieve → answer) | match | plan | setup → end.
Each node is a plain function over the shared services, so the graph stays
testable without a running LLM.
"""
from __future__ import annotations

from typing import TypedDict

from . import github_api
from .archmap import architecture_answer
from .embeddings import get_embedder
from .plan import build_plan
from .qa import answer_question
from .retrieve import Retriever
from .store import Store


class AgentState(TypedDict, total=False):
    intent: str
    question: str
    repo: str
    source: str          # original repo URL/path, for live issue fetch
    skills: str
    issue_number: int
    result: dict


class ContribAgent:
    def __init__(self, store: Store | None = None, embedder=None):
        self.store = store or Store()
        self.embedder = embedder or get_embedder()
        self.retriever = Retriever(self.store, self.embedder)
        self.graph = self._build_graph()

    # -- nodes -------------------------------------------------------------
    def _route(self, s: AgentState) -> AgentState:
        q = (s.get("question") or "").lower()
        if s.get("intent"):
            return s
        if s.get("issue_number"):
            s["intent"] = "plan"
        elif any(w in q for w in ("architecture", "structure of", "overview",
                                  "explain this repo", "how is this organized",
                                  "how is this organised", "repo map")):
            s["intent"] = "architecture"
        elif any(w in q for w in ("good first issue", "which issue", "contribute",
                                  "issue should i", "issues for me", "match me")):
            s["intent"] = "match"
        elif any(w in q for w in ("setup", "set up", "install", "run locally",
                                  "how do i run")):
            s["intent"] = "setup"
        else:
            s["intent"] = "ask"
        return s

    def _ask(self, s: AgentState) -> AgentState:
        s["result"] = answer_question(self.retriever, s["question"], repo=s.get("repo"))
        return s

    def _architecture(self, s: AgentState) -> AgentState:
        s["result"] = architecture_answer(self.store, s.get("repo"))
        return s

    def _setup(self, s: AgentState) -> AgentState:
        q = ("How is this project set up and run locally? installation steps "
             "requirements README " + (s.get("question") or ""))
        res = answer_question(self.retriever, q, repo=s.get("repo"), k=10)
        res = {**res, "mode": f"setup/{res['mode']}"}
        s["result"] = res
        return s

    def _match(self, s: AgentState) -> AgentState:
        issues = github_api.fetch_issues(s.get("source") or "")
        ranked = github_api.rank_issues(issues, self.embedder,
                                        s.get("skills") or "python, llm, rag")
        s["result"] = {"matches": [
            {"number": r["issue"].number, "title": r["issue"].title,
             "url": r["issue"].url, "labels": r["issue"].labels,
             "score": r["score"], "why": r["why"]} for r in ranked]}
        return s

    def _plan(self, s: AgentState) -> AgentState:
        issues = github_api.fetch_issues(s.get("source") or "", state="all")
        iss = next((i for i in issues if i.number == s.get("issue_number")), None)
        if iss is None:
            s["result"] = {"error": f"Issue #{s.get('issue_number')} not found"}
        else:
            s["result"] = build_plan(self.retriever, iss, repo=s.get("repo"))
        return s

    # -- graph ---------------------------------------------------------------
    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
            g = StateGraph(AgentState)
            g.add_node("route", self._route)
            g.add_node("ask", self._ask)
            g.add_node("architecture", self._architecture)
            g.add_node("setup", self._setup)
            g.add_node("match", self._match)
            g.add_node("plan", self._plan)
            g.set_entry_point("route")
            g.add_conditional_edges("route",
                                    lambda s: s["intent"],
                                    {"ask": "ask", "setup": "setup",
                                     "match": "match", "plan": "plan",
                                     "architecture": "architecture"})
            for n in ("ask", "setup", "match", "plan", "architecture"):
                g.add_edge(n, END)
            return g.compile()
        except Exception:
            return None  # direct-dispatch fallback in run()

    def run(self, **kwargs) -> dict:
        state: AgentState = dict(kwargs)  # type: ignore[assignment]
        if self.graph is not None:
            out = self.graph.invoke(state)
            return out.get("result", {})
        state = self._route(state)
        node = {"ask": self._ask, "setup": self._setup,
                "match": self._match, "plan": self._plan,
                "architecture": self._architecture}[state["intent"]]
        return node(state)["result"]
