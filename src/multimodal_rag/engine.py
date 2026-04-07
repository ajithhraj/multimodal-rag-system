from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.embedding.providers import TextEmbedder, VisionEmbedder
from multimodal_rag.generation.synthesizer import AnswerSynthesizer
from multimodal_rag.ingestion.loader import discover_files, ingest_files
from multimodal_rag.models import Citation, Chunk, Modality, QueryAnswer, RetrievalHit
from multimodal_rag.retrieval import CrossEncoderReranker, LexicalIndex, reciprocal_rank_fusion
from multimodal_rag.storage.base import VectorStore
from multimodal_rag.storage.factory import create_vector_store


class MultimodalRAG:
    def __init__(self, settings: Settings, store: VectorStore | None = None):
        self.settings = settings
        self.store = store or create_vector_store(settings)
        self.text_embedder = TextEmbedder(settings)
        self.vision_embedder = VisionEmbedder()
        self.synthesizer = AnswerSynthesizer(settings)
        self.lexical_index = LexicalIndex(settings.storage_dir / "lexical")
        self.reranker = CrossEncoderReranker(
            enabled=settings.retrieval_enable_reranker,
            model_name=settings.retrieval_reranker_model,
        )

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

        # Sparse lexical index covers all textual content, including image captions.
        self.lexical_index.upsert(target_collection, chunks)

        return counts

    @staticmethod
    def _build_citations(hits: list[RetrievalHit]) -> list[Citation]:
        citations: list[Citation] = []
        for hit in hits:
            page_number = hit.chunk.metadata.get("page_number")
            page_value: int | None = None
            if isinstance(page_number, int):
                page_value = page_number
            elif isinstance(page_number, str) and page_number.isdigit():
                page_value = int(page_number)
            excerpt = hit.chunk.content.replace("\n", " ").strip()[:220] or None
            citations.append(
                Citation(
                    chunk_id=hit.chunk.chunk_id,
                    source_path=hit.chunk.source_path,
                    modality=hit.chunk.modality,
                    page_number=page_value,
                    excerpt=excerpt,
                )
            )
        return citations

    def _build_vision_query_vector(self, question: str, query_image_path: Path | None) -> list[float]:
        if query_image_path and query_image_path.exists():
            vectors = self.vision_embedder.embed_images(
                [query_image_path],
                [question or query_image_path.name],
            )
            if vectors:
                return vectors[0]
        return self.vision_embedder.embed_query(question)

    def query(
        self,
        question: str,
        collection: str | None = None,
        top_k: int | None = None,
        query_image_path: Path | None = None,
    ) -> QueryAnswer:
        target_collection = self._resolve_collection(collection)
        per_modality_k = top_k or self.settings.retrieval_top_k_per_modality

        text_query_vec = self.text_embedder.embed_query(question)
        vision_query_vec = self._build_vision_query_vector(question, query_image_path)

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
        lexical_hits = self.lexical_index.search(
            target_collection,
            question,
            top_k=self.settings.retrieval_top_k_lexical,
        )

        fused = reciprocal_rank_fusion(
            [text_hits, table_hits, image_hits, lexical_hits],
            k=self.settings.retrieval_rrf_k,
        )
        rerank_pool = fused[: self.settings.retrieval_rerank_candidates]
        final_hits = self.reranker.rerank(question, rerank_pool, top_k=self.settings.max_context_chunks)

        answer = self.synthesizer.generate(question, final_hits)
        citations = self._build_citations(final_hits)
        return QueryAnswer(answer=answer, hits=final_hits, citations=citations)
