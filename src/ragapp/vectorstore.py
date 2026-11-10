"""Thin Qdrant wrapper: one collection of chunk vectors with text + source payloads."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import Settings
from .embeddings import Embedder


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    score: float = 0.0

    @property
    def point_id(self) -> str:
        # Deterministic id => re-ingesting the same file overwrites, never duplicates.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.source}#{self.chunk_index}"))


def build_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if settings.qdrant_path == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(path=settings.qdrant_path)


class VectorStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            size = self.client.get_collection(self.collection).config.params.vectors.size
            if size != self.embedder.dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has {size}-dim vectors but embedder "
                    f"'{self.embedder.name}' produces {self.embedder.dim}. "
                    "Re-run `ragapp ingest --recreate`."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            self.client.upsert(
                self.collection,
                points=[
                    PointStruct(
                        id=c.point_id,
                        vector=v,
                        payload={"text": c.text, "source": c.source, "chunk_index": c.chunk_index},
                    )
                    for c, v in zip(batch, vectors)
                ],
            )
        return len(chunks)

    def search(self, query: str, top_k: int) -> list[Chunk]:
        vector = self.embedder.embed([query])[0]
        hits = self.client.query_points(
            self.collection, query=vector, limit=top_k, with_payload=True
        ).points
        return [
            Chunk(
                text=h.payload["text"],
                source=h.payload["source"],
                chunk_index=h.payload["chunk_index"],
                score=float(h.score),
            )
            for h in hits
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count
