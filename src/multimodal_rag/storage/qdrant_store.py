from __future__ import annotations

import re

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from multimodal_rag.config import Settings
from multimodal_rag.models import Chunk, RetrievalHit
from multimodal_rag.storage.base import VectorStore


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())


class QdrantStore(VectorStore):
    def __init__(self, settings: Settings):
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        self._prefix = _safe_name(settings.qdrant_collection_prefix)

    def _collection_name(self, collection: str, modality: str) -> str:
        return f"{self._prefix}_{_safe_name(collection)}_{_safe_name(modality)}"

    def _ensure_collection(self, name: str, dim: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            return

        info = self.client.get_collection(name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != dim:
            raise ValueError(
                f"Qdrant collection dimension mismatch for {name}. "
                f"Existing={existing_dim}, incoming={dim}. Use a new collection."
            )

    def upsert(
        self,
        collection: str,
        modality: str,
        vectors: list[list[float]],
        chunks: list[Chunk],
    ) -> int:
        if not vectors:
            return 0
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        dim = len(vectors[0])
        name = self._collection_name(collection, modality)
        self._ensure_collection(name, dim)

        points = []
        for vector, chunk in zip(vectors, chunks, strict=False):
            points.append(
                qm.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=chunk.to_payload(),
                )
            )

        self.client.upsert(collection_name=name, points=points, wait=True)
        return len(points)

    def query(
        self,
        collection: str,
        modality: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        name = self._collection_name(collection, modality)
        if not self.client.collection_exists(name):
            return []

        results = self.client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits: list[RetrievalHit] = []
        for result in results:
            payload = result.payload or {}
            chunk = Chunk.from_payload(payload)
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(result.score),
                    backend="qdrant",
                )
            )
        return hits

    def delete_by_source(
        self,
        collection: str,
        modality: str,
        source_paths: list[str],
    ) -> int:
        if not source_paths:
            return 0

        name = self._collection_name(collection, modality)
        if not self.client.collection_exists(name):
            return 0

        source_set = set(source_paths)
        point_ids: list[str | int] = []
        offset: str | int | None = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=name,
                with_payload=True,
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            if not records:
                break

            for record in records:
                payload = record.payload or {}
                source_path = str(payload.get("source_path", ""))
                if source_path in source_set:
                    point_ids.append(record.id)

            if next_offset is None:
                break
            offset = next_offset

        if not point_ids:
            return 0

        self.client.delete(
            collection_name=name,
            points_selector=qm.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(point_ids)

    def delete_collection(self, collection: str) -> int:
        removed = 0
        for modality in ("text", "table", "image"):
            name = self._collection_name(collection, modality)
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
                removed += 1
        return removed
