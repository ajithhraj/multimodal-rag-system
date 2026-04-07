from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import orjson

from multimodal_rag.models import Chunk, RetrievalHit
from multimodal_rag.storage.base import VectorStore


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FaissStore(VectorStore):
    """FAISS-backed store with a NumPy cosine fallback when FAISS is unavailable."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except Exception:
            self._faiss = None

    def _modality_dir(self, collection: str, modality: str) -> Path:
        path = self.base_dir / _safe_name(collection) / _safe_name(modality)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _metadata_path(modality_dir: Path) -> Path:
        return modality_dir / "chunks.json"

    @staticmethod
    def _faiss_path(modality_dir: Path) -> Path:
        return modality_dir / "index.faiss"

    @staticmethod
    def _vectors_path(modality_dir: Path) -> Path:
        return modality_dir / "vectors.npy"

    def _load_chunks(self, modality_dir: Path) -> list[Chunk]:
        metadata_path = self._metadata_path(modality_dir)
        if not metadata_path.exists():
            return []
        payload = orjson.loads(metadata_path.read_bytes())
        return [Chunk.from_payload(item) for item in payload]

    def _save_chunks(self, modality_dir: Path, chunks: list[Chunk]) -> None:
        payload = [chunk.to_payload() for chunk in chunks]
        self._metadata_path(modality_dir).write_bytes(orjson.dumps(payload))

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

        modality_dir = self._modality_dir(collection, modality)
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix = _normalize(matrix)

        existing_chunks = self._load_chunks(modality_dir)
        existing_chunks.extend(chunks)
        self._save_chunks(modality_dir, existing_chunks)

        if self._faiss:
            index_path = self._faiss_path(modality_dir)
            if index_path.exists():
                index = self._faiss.read_index(str(index_path))
                if index.d != matrix.shape[1]:
                    raise ValueError(
                        f"Embedding dimension mismatch. Existing={index.d}, incoming={matrix.shape[1]}"
                    )
            else:
                index = self._faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss.write_index(index, str(index_path))
            return len(chunks)

        vectors_path = self._vectors_path(modality_dir)
        if vectors_path.exists():
            existing_matrix = np.load(vectors_path)
            if existing_matrix.shape[1] != matrix.shape[1]:
                raise ValueError(
                    "Embedding dimension mismatch for NumPy fallback index: "
                    f"existing={existing_matrix.shape[1]}, incoming={matrix.shape[1]}"
                )
            matrix = np.vstack([existing_matrix, matrix])
        np.save(vectors_path, matrix)
        return len(chunks)

    def query(
        self,
        collection: str,
        modality: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        modality_dir = self._modality_dir(collection, modality)
        chunks = self._load_chunks(modality_dir)
        if not chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        query = _normalize(query)

        hits: list[RetrievalHit] = []
        if self._faiss:
            index_path = self._faiss_path(modality_dir)
            if not index_path.exists():
                return []
            index = self._faiss.read_index(str(index_path))
            scores, ids = index.search(query, top_k)
            for score, idx in zip(scores[0], ids[0], strict=False):
                if idx < 0 or idx >= len(chunks):
                    continue
                hits.append(RetrievalHit(chunk=chunks[idx], score=float(score), backend="faiss"))
            return hits

        vectors_path = self._vectors_path(modality_dir)
        if not vectors_path.exists():
            return []
        matrix = np.load(vectors_path)
        if matrix.size == 0:
            return []
        if matrix.shape[1] != query.shape[1]:
            return []

        sims = np.dot(matrix, query.T).reshape(-1)
        idxs = np.argsort(-sims)[:top_k]
        for idx in idxs:
            hits.append(
                RetrievalHit(chunk=chunks[int(idx)], score=float(sims[int(idx)]), backend="numpy-fallback")
            )
        return hits
