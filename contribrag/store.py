"""Vector store: Qdrant in local (embedded) mode — no server needed."""
from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings
from .models import Chunk

COLLECTION = "contribrag_chunks"


class Store:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or (settings.data_dir / "qdrant"))
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    def ensure(self, dim: int):
        names = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in names:
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]):
        points = []
        for ch, vec in zip(chunks, vectors):
            pid = uuid.uuid5(uuid.NAMESPACE_URL, ch.id).hex
            points.append(PointStruct(id=pid, vector=vec, payload=ch.to_payload()))
        for i in range(0, len(points), 256):
            self.client.upsert(COLLECTION, points[i:i + 256])

    def search(self, vector: list[float], limit: int = 30,
               repo: str | None = None) -> list[tuple[Chunk, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        flt = None
        if repo:
            flt = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])
        hits = self.client.query_points(COLLECTION, query=vector, limit=limit,
                                        query_filter=flt).points
        return [(Chunk.from_payload(h.payload), float(h.score)) for h in hits]

    def all_chunks(self, repo: str | None = None) -> list[Chunk]:
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(COLLECTION, limit=512, offset=offset,
                                                with_payload=True)
            for p in points:
                ch = Chunk.from_payload(p.payload)
                if repo is None or ch.repo == repo:
                    out.append(ch)
            if offset is None:
                return out

    def count(self, repo: str | None = None) -> int:
        return len(self.all_chunks(repo))

    def prune(self, repo: str, keep_chunk_ids: set[str]) -> int:
        """Delete this repo's points whose chunk id is not in keep_chunk_ids.
        Point ids are uuid5(chunk_id), so stale points are computable."""
        from qdrant_client.models import PointIdsList
        stale = []
        for ch in self.all_chunks(repo):
            if ch.id not in keep_chunk_ids:
                stale.append(uuid.uuid5(uuid.NAMESPACE_URL, ch.id).hex)
        for i in range(0, len(stale), 256):
            self.client.delete(COLLECTION,
                               points_selector=PointIdsList(points=stale[i:i + 256]))
        return len(stale)
