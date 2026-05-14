from __future__ import annotations

import re
import shutil
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

    @staticmethod
    def _remove_if_exists(path: Path) -> None:
        if path.exists():
            path.unlink()

    def _load_chunks(self, modality_dir: Path) -> list[Chunk]:
        metadata_path = self._metadata_path(modality_dir)
        if not metadata_path.exists():
            return []
        payload = orjson.loads(metadata_path.read_bytes())
        return [Chunk.from_payload(item) for item in payload]

    def _save_chunks(self, modality_dir: Path, chunks: list[Chunk]) -> None:
        payload = [chunk.to_payload() for chunk in chunks]
        self._metadata_path(modality_dir).write_bytes(orjson.dumps(payload))

    def _reconstruct_from_faiss(self, modality_dir: Path, expected_rows: int) -> np.ndarray:
        if not self._faiss or expected_rows <= 0:
            return np.empty((0, 0), dtype=np.float32)
        index_path = self._faiss_path(modality_dir)
        if not index_path.exists():
            return np.empty((0, 0), dtype=np.float32)
        index = self._faiss.read_index(str(index_path))
        if index.ntotal <= 0:
            return np.empty((0, 0), dtype=np.float32)
        row_count = min(index.ntotal, expected_rows)
        reconstructed = index.reconstruct_n(0, row_count)
        return np.asarray(reconstructed, dtype=np.float32)

    def _load_state(self, modality_dir: Path) -> tuple[list[Chunk], np.ndarray]:
        chunks = self._load_chunks(modality_dir)
        if not chunks:
            return [], np.empty((0, 0), dtype=np.float32)

        vectors_path = self._vectors_path(modality_dir)
        if vectors_path.exists():
            matrix = np.asarray(np.load(vectors_path), dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
        else:
            matrix = self._reconstruct_from_faiss(modality_dir, expected_rows=len(chunks))

        if matrix.size == 0:
            return [], np.empty((0, 0), dtype=np.float32)

        row_count = min(len(chunks), matrix.shape[0])
        if row_count <= 0:
            return [], np.empty((0, 0), dtype=np.float32)
        return chunks[:row_count], matrix[:row_count]

    def _save_state(self, modality_dir: Path, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if not chunks or matrix.size == 0:
            self._remove_if_exists(self._metadata_path(modality_dir))
            self._remove_if_exists(self._vectors_path(modality_dir))
            self._remove_if_exists(self._faiss_path(modality_dir))
            return

        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(chunks):
            raise ValueError("chunks and vectors row count mismatch")

        self._save_chunks(modality_dir, chunks)
        np.save(self._vectors_path(modality_dir), matrix)

        if not self._faiss:
            return

        index = self._faiss.IndexFlatIP(matrix.shape[1])
        index.add(np.ascontiguousarray(matrix))
        self._faiss.write_index(index, str(self._faiss_path(modality_dir)))

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
        existing_chunks, existing_matrix = self._load_state(modality_dir)
        incoming = _normalize(np.asarray(vectors, dtype=np.float32))

        if existing_matrix.size > 0 and existing_matrix.shape[1] != incoming.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"existing={existing_matrix.shape[1]}, incoming={incoming.shape[1]}"
            )

        entries: dict[str, tuple[Chunk, np.ndarray]] = {}
        for idx, chunk in enumerate(existing_chunks):
            entries[chunk.chunk_id] = (chunk, existing_matrix[idx])
        for vector, chunk in zip(incoming, chunks, strict=False):
            entries[chunk.chunk_id] = (chunk, vector)

        merged_chunks = [item[0] for item in entries.values()]
        merged_matrix = np.vstack([item[1] for item in entries.values()]).astype(np.float32)

        self._save_state(modality_dir, merged_chunks, merged_matrix)
        return len(chunks)

    def query(
        self,
        collection: str,
        modality: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        modality_dir = self._modality_dir(collection, modality)
        chunks, matrix = self._load_state(modality_dir)
        if not chunks or matrix.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        query = _normalize(query)
        if matrix.shape[1] != query.shape[1]:
            return []

        hits: list[RetrievalHit] = []
        max_results = min(top_k, len(chunks))
        if max_results <= 0:
            return []

        if self._faiss:
            index_path = self._faiss_path(modality_dir)
            if index_path.exists():
                index = self._faiss.read_index(str(index_path))
                if index.d == query.shape[1] and index.ntotal == len(chunks):
                    scores, ids = index.search(query, max_results)
                    for score, idx in zip(scores[0], ids[0], strict=False):
                        if idx < 0 or idx >= len(chunks):
                            continue
                        hits.append(RetrievalHit(chunk=chunks[idx], score=float(score), backend="faiss"))
                    return hits

        sims = np.dot(matrix, query.T).reshape(-1)
        idxs = np.argsort(-sims)[:max_results]
        for idx in idxs:
            hits.append(
                RetrievalHit(chunk=chunks[int(idx)], score=float(sims[int(idx)]), backend="numpy-fallback")
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

        modality_dir = self._modality_dir(collection, modality)
        chunks, matrix = self._load_state(modality_dir)
        if not chunks:
            return 0

        source_set = set(source_paths)
        kept_chunks: list[Chunk] = []
        kept_vectors: list[np.ndarray] = []
        removed = 0
        for idx, chunk in enumerate(chunks):
            if chunk.source_path in source_set:
                removed += 1
                continue
            kept_chunks.append(chunk)
            kept_vectors.append(matrix[idx])

        if removed <= 0:
            return 0

        if kept_vectors:
            kept_matrix = np.vstack(kept_vectors).astype(np.float32)
        else:
            kept_matrix = np.empty((0, matrix.shape[1]), dtype=np.float32)
        self._save_state(modality_dir, kept_chunks, kept_matrix)
        return removed

    def delete_collection(self, collection: str) -> int:
        collection_dir = self.base_dir / _safe_name(collection)
        if not collection_dir.exists():
            return 0
        shutil.rmtree(collection_dir)
        return 1
