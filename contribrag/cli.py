"""CLI: python -m contribrag.cli <command>

  ingest <github-url|local-path>     clone/copy, chunk, index (+ issues)
  ask "<question>" [--repo label]    cited Q&A over the index
  match <github-url> --skills "..."  contribute-mode issue ranking
  plan <github-url> <issue-number>   grounded PR plan for one issue
  stats                              indexed chunk counts
"""
from __future__ import annotations

import argparse
import json

from . import github_api
from .agent import ContribAgent
from .pipeline import index_repo
from .store import Store


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="contribrag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("source")
    p.add_argument("--no-issues", action="store_true")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--repo")
    p = sub.add_parser("match"); p.add_argument("source")
    p.add_argument("--skills", default="python, llm, rag, agents")
    p = sub.add_parser("plan"); p.add_argument("source"); p.add_argument("issue", type=int)
    p.add_argument("--repo")
    sub.add_parser("stats")

    args = ap.parse_args(argv)
    if args.cmd == "ingest":
        print(json.dumps(index_repo(args.source, with_issues=not args.no_issues), indent=2))
        return
    if args.cmd == "stats":
        store = Store()
        chunks = store.all_chunks()
        by_repo: dict[str, int] = {}
        for c in chunks:
            by_repo[c.repo] = by_repo.get(c.repo, 0) + 1
        print(json.dumps({"total_chunks": len(chunks), "by_repo": by_repo}, indent=2))
        return
    agent = ContribAgent()
    if args.cmd == "ask":
        out = agent.run(question=args.question, repo=args.repo)
        print(out["answer"])
        print("\nSources:")
        for c in out["chunks"]:
            print(f"  [{c.citation}] {c.symbol}")
    elif args.cmd == "match":
        out = agent.run(question="which issue should I contribute to",
                        source=args.source, skills=args.skills)
        for m in out["matches"]:
            print(f"#{m['number']} {m['title']}  (score {m['score']})")
            print(f"    why: {m['why']}\n    {m['url']}")
    elif args.cmd == "plan":
        out = agent.run(source=args.source, issue_number=args.issue, repo=args.repo)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
