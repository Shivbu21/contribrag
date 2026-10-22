"""Core data models."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    repo: str            # owner/name or local label
    path: str            # path inside repo
    kind: str            # "code" | "doc" | "issue" | "pr"
    symbol: str          # function/class name, doc heading, or issue title
    start_line: int      # 1-based
    end_line: int        # 1-based, inclusive
    text: str
    language: str = ""

    @property
    def id(self) -> str:
        raw = f"{self.repo}|{self.path}|{self.start_line}|{self.end_line}|{self.symbol}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    @property
    def citation(self) -> str:
        if self.kind in ("issue", "pr"):
            return f"{self.symbol} ({self.path})"
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def embed_text(self) -> str:
        """Contextual enrichment: the text actually embedded/searched carries
        its path + symbol header, so 'where is X defined' queries match the
        chunk that defines X, not just chunks that mention it."""
        return f"{self.path} :: {self.symbol}\n{self.text}"

    def to_payload(self) -> dict:
        return {
            "repo": self.repo, "path": self.path, "kind": self.kind,
            "symbol": self.symbol, "start_line": self.start_line,
            "end_line": self.end_line, "text": self.text,
            "language": self.language, "chunk_id": self.id,
        }

    @staticmethod
    def from_payload(p: dict) -> "Chunk":
        return Chunk(repo=p["repo"], path=p["path"], kind=p["kind"],
                     symbol=p["symbol"], start_line=p["start_line"],
                     end_line=p["end_line"], text=p["text"],
                     language=p.get("language", ""))


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    url: str = ""
    state: str = "open"

    @property
    def is_beginner_friendly(self) -> bool:
        tags = {t.lower() for t in self.labels}
        return bool(tags & {"good first issue", "help wanted", "beginner",
                            "good-first-issue", "first-timers-only", "easy"})

    def to_chunk(self, repo: str) -> Chunk:
        text = f"Issue #{self.number}: {self.title}\nLabels: {', '.join(self.labels)}\n\n{self.body}"
        return Chunk(repo=repo, path=f"issue/#{self.number}", kind="issue",
                     symbol=f"Issue #{self.number}: {self.title}",
                     start_line=1, end_line=max(1, text.count("\n") + 1), text=text)
