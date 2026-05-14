from __future__ import annotations

from abc import ABC, abstractmethod

from multimodal_rag.models import Chunk, RetrievalHit


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        collection: str,
        modality: str,
        vectors: list[list[float]],
        chunks: list[Chunk],
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        collection: str,
        modality: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        raise NotImplementedError

    @abstractmethod
    def delete_by_source(
        self,
        collection: str,
        modality: str,
        source_paths: list[str],
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self, collection: str) -> int:
        raise NotImplementedError
