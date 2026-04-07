from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.embedding.providers import TextEmbedder, VisionEmbedder
from multimodal_rag.generation.synthesizer import AnswerSynthesizer
from multimodal_rag.ingestion.loader import discover_files, ingest_files
from multimodal_rag.models import Chunk, Modality, QueryAnswer, RetrievalHit
from multimodal_rag.storage.base import VectorStore
from multimodal_rag.storage.factory import create_vector_store


def _normalize_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    if not hits:
        return []
    scores = [hit.score for hit in hits]
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-9:
        return [RetrievalHit(chunk=hit.chunk, score=1.0, backend=hit.backend) for hit in hits]

    normalized: list[RetrievalHit] = []
    for hit in hits:
        score = (hit.score - low) / (high - low)
        normalized.append(RetrievalHit(chunk=hit.chunk, score=float(score), backend=hit.backend))
    return normalized


class MultimodalRAG:
    def __init__(self, settings: Settings, store: VectorStore | None = None):
        self.settings = settings
        self.store = store or create_vector_store(settings)
        self.text_embedder = TextEmbedder(settings)
        self.vision_embedder = VisionEmbedder()
        self.synthesizer = AnswerSynthesizer(settings)

    @classmethod
    def from_settings(cls) -> "MultimodalRAG":
        return cls(get_settings())

    def _resolve_collection(self, collection: str | None) -> str:
        return collection or self.settings.collection

    def _group_by_modality(self, chunks: Iterable[Chunk]) -> dict[Modality, list[Chunk]]:
        grouped: dict[Modality, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.modality].append(chunk)
        return grouped

    def ingest_paths(self, raw_paths: list[Path], collection: str | None = None) -> dict[str, int]:
        target_collection = self._resolve_collection(collection)
        files: list[Path] = []
        for raw in raw_paths:
            files.extend(discover_files(raw))
        files = sorted({path.resolve() for path in files})

        chunks = ingest_files(files, self.settings)
        grouped = self._group_by_modality(chunks)

        counts = {"files": len(files), "chunks": len(chunks), "text": 0, "table": 0, "image": 0}

        text_chunks = grouped.get(Modality.TEXT, [])
        if text_chunks:
            vectors = self.text_embedder.embed_documents([chunk.content for chunk in text_chunks])
            counts["text"] = self.store.upsert(
                target_collection,
                Modality.TEXT.value,
                vectors,
                text_chunks,
            )

        table_chunks = grouped.get(Modality.TABLE, [])
        if table_chunks:
            vectors = self.text_embedder.embed_documents([chunk.content for chunk in table_chunks])
            counts["table"] = self.store.upsert(
                target_collection,
                Modality.TABLE.value,
                vectors,
                table_chunks,
            )

        image_chunks = grouped.get(Modality.IMAGE, [])
        if image_chunks:
            image_paths = [Path(chunk.metadata.get("image_path", chunk.source_path)) for chunk in image_chunks]
            image_text = [chunk.content for chunk in image_chunks]
            vectors = self.vision_embedder.embed_images(image_paths, image_text)
            counts["image"] = self.store.upsert(
                target_collection,
                Modality.IMAGE.value,
                vectors,
                image_chunks,
            )

        return counts

    @staticmethod
    def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        best: dict[str, RetrievalHit] = {}
        for hit in hits:
            key = hit.chunk.chunk_id
            prior = best.get(key)
            if prior is None or hit.score > prior.score:
                best[key] = hit
        return list(best.values())

    def query(self, question: str, collection: str | None = None, top_k: int | None = None) -> QueryAnswer:
        target_collection = self._resolve_collection(collection)
        per_modality_k = top_k or self.settings.retrieval_top_k_per_modality

        text_query_vec = self.text_embedder.embed_query(question)
        vision_query_vec = self.vision_embedder.embed_query(question)

        text_hits = self.store.query(
            target_collection,
            Modality.TEXT.value,
            text_query_vec,
            per_modality_k,
        )
        table_hits = self.store.query(
            target_collection,
            Modality.TABLE.value,
            text_query_vec,
            per_modality_k,
        )
        image_hits = self.store.query(
            target_collection,
            Modality.IMAGE.value,
            vision_query_vec,
            per_modality_k,
        )

        fused = (
            _normalize_hits(text_hits)
            + _normalize_hits(table_hits)
            + _normalize_hits(image_hits)
        )
        fused = self._dedupe_hits(fused)
        fused.sort(key=lambda item: item.score, reverse=True)
        final_hits = fused[: self.settings.max_context_chunks]

        answer = self.synthesizer.generate(question, final_hits)
        return QueryAnswer(answer=answer, hits=final_hits)
