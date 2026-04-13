from __future__ import annotations

from multimodal_rag.config import Settings
from multimodal_rag.storage.base import VectorStore
from multimodal_rag.storage.faiss_store import FaissStore
from multimodal_rag.storage.qdrant_store import QdrantStore


def create_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "qdrant":
        return QdrantStore(settings)
    return FaissStore(settings.storage_dir / "faiss")
