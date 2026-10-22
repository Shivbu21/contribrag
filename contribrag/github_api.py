"""GitHub API access (issues + PRs) and the contribute-mode issue matcher."""
from __future__ import annotations

import requests

from .config import settings
from .ingest import parse_github_url
from .models import Issue

API = "https://api.github.com"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def fetch_issues(source: str, state: str = "open", limit: int = 100) -> list[Issue]:
    parsed = parse_github_url(source)
    if not parsed:
        return []
    owner, name = parsed
    out: list[Issue] = []
    page = 1
    while len(out) < limit:
        r = requests.get(f"{API}/repos/{owner}/{name}/issues",
                         params={"state": state, "per_page": 100, "page": page},
                         headers=_headers(), timeout=30)
        if r.status_code != 200:
            break
        items = r.json()
        if not items:
            break
        for it in items:
            if "pull_request" in it:  # issues endpoint includes PRs; skip them here
                continue
            out.append(Issue(number=it["number"], title=it["title"],
                             body=(it.get("body") or "")[:4000],
                             labels=[lb["name"] for lb in it.get("labels", [])],
                             url=it["html_url"], state=it["state"]))
        page += 1
    return out[:limit]


def fetch_prs(source: str, limit: int = 50) -> list[dict]:
    parsed = parse_github_url(source)
    if not parsed:
        return []
    owner, name = parsed
    r = requests.get(f"{API}/repos/{owner}/{name}/pulls",
                     params={"state": "closed", "per_page": limit, "sort": "updated"},
                     headers=_headers(), timeout=30)
    if r.status_code != 200:
        return []
    return [{"number": p["number"], "title": p["title"], "url": p["html_url"],
             "merged": bool(p.get("merged_at"))} for p in r.json()]


# --- Contribute mode -------------------------------------------------------

def rank_issues(issues: list[Issue], embedder, skills: str,
                explored_paths: set[str] | None = None, top: int = 5) -> list[dict]:
    """Rank open issues for this contributor: semantic fit to their skills,
    beginner-friendly labels first, bonus when the issue mentions code areas
    they have already explored in Q&A."""
    if not issues:
        return []
    profile = f"Contributor skills and interests: {skills}"
    pv = embedder.embed([profile])[0]
    ivs = embedder.embed([f"{i.title}\n{i.body[:1500]}" for i in issues])

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    ranked = []
    for iss, vec in zip(issues, ivs):
        score = cos(pv, vec)
        why = []
        if iss.is_beginner_friendly:
            score += 0.15
            why.append("labelled beginner-friendly")
        hits = [p for p in (explored_paths or set()) if p.split("/")[-1].split(".")[0]
                and p.split("/")[-1].split(".")[0] in (iss.title + iss.body)]
        if hits:
            score += 0.05 * len(hits)
            why.append(f"touches code you explored ({', '.join(sorted(set(hits))[:2])})")
        why.append("semantic fit to your skills" if not why else "plus semantic fit")
        ranked.append({"issue": iss, "score": round(score, 3), "why": "; ".join(why)})
    ranked.sort(key=lambda r: -r["score"])
    # Beginner-friendly first within a small score band reads better for newcomers.
    return ranked[:top]
