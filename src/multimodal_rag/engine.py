from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Iterable, Literal

import orjson

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.embedding.providers import TextEmbedder, VisionEmbedder
from multimodal_rag.generation.synthesizer import AnswerSynthesizer
from multimodal_rag.ingestion.loader import discover_files, ingest_files
from multimodal_rag.models import Citation, Chunk, Modality, QueryAnswer, RetrievalHit
from multimodal_rag.retrieval import CrossEncoderReranker, LexicalIndex, reciprocal_rank_fusion
from multimodal_rag.storage.base import VectorStore
from multimodal_rag.storage.factory import create_vector_store

RetrievalMode = Literal["dense_only", "hybrid", "hybrid_rerank"]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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

    def _resolve_tenant(self, tenant_id: str | None) -> str:
        raw = tenant_id or self.settings.default_tenant
        return self.settings.normalize_tenant_id(raw)

    @staticmethod
    def _safe_collection_name(raw: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip().lower())
        cleaned = cleaned.strip("-_")
        return cleaned or "default"

    def _scoped_collection(self, collection: str | None, tenant_id: str | None) -> str:
        tenant = self._resolve_tenant(tenant_id)
        base = self._safe_collection_name(self._resolve_collection(collection))
        return f"tenant-{tenant}__{base}"

    def _manifest_path(self, scoped_collection: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", scoped_collection.strip().lower())
        return self.settings.storage_dir / "manifests" / f"{safe}.json"

    def _load_manifest(self, scoped_collection: str) -> dict[str, str]:
        path = self._manifest_path(scoped_collection)
        if not path.exists():
            return {}
        payload = orjson.loads(path.read_bytes())
        if not isinstance(payload, dict):
            return {}
        manifest: dict[str, str] = {}
        for key, value in payload.items():
            manifest[str(key)] = str(value)
        return manifest

    def _save_manifest(self, scoped_collection: str, manifest: dict[str, str]) -> None:
        path = self._manifest_path(scoped_collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _group_by_modality(self, chunks: Iterable[Chunk]) -> dict[Modality, list[Chunk]]:
        grouped: dict[Modality, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.modality].append(chunk)
        return grouped

    def _default_retrieval_mode(self) -> RetrievalMode:
        if self.settings.retrieval_enable_reranker:
            return "hybrid_rerank"
        return "hybrid"

    @staticmethod
    def _content_token_set(text: str) -> set[str]:
        return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _diversify_hits(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []

        max_context_chunks = self.settings.max_context_chunks
        if not self.settings.retrieval_enable_result_diversity:
            return hits[:max_context_chunks]

        per_source_cap = self.settings.retrieval_max_chunks_per_source
        duplicate_threshold = self.settings.retrieval_duplicate_similarity_threshold

        source_counts: dict[str, int] = {}
        kept: list[RetrievalHit] = []
        kept_token_sets: list[set[str]] = []

        for hit in hits:
            source = hit.chunk.source_path
            source_count = source_counts.get(source, 0)
            if source_count >= per_source_cap:
                continue

            candidate_tokens = self._content_token_set(hit.chunk.content)
            is_duplicate = False
            if candidate_tokens:
                for existing_tokens in kept_token_sets:
                    if self._jaccard_similarity(candidate_tokens, existing_tokens) >= duplicate_threshold:
                        is_duplicate = True
                        break
            if is_duplicate:
                continue

            kept.append(hit)
            kept_token_sets.append(candidate_tokens)
            source_counts[source] = source_count + 1
            if len(kept) >= max_context_chunks:
                break

        if kept:
            return kept
        return hits[:max_context_chunks]

    @staticmethod
    def _resolve_retrieval_mode(retrieval_mode: str | None, default_mode: RetrievalMode) -> RetrievalMode:
        allowed: set[str] = {"dense_only", "hybrid", "hybrid_rerank"}
        mode = retrieval_mode or default_mode
        if mode not in allowed:
            raise ValueError(f"Unsupported retrieval_mode '{mode}'. Allowed: dense_only, hybrid, hybrid_rerank")
        return mode  # type: ignore[return-value]

    def ingest_paths(
        self,
        raw_paths: list[Path],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, int]:
        target_collection = self._scoped_collection(collection, tenant_id)
        manifest = self._load_manifest(target_collection)
        files: list[Path] = []
        for raw in raw_paths:
            files.extend(discover_files(raw))
        files = sorted({path.resolve() for path in files})

        changed_files: list[Path] = []
        changed_source_paths: list[str] = []
        updated_manifest = dict(manifest)
        for path in files:
            source_path = str(path)
            fingerprint = self._file_fingerprint(path)
            if (
                self.settings.ingestion_skip_unchanged_files
                and manifest.get(source_path) == fingerprint
            ):
                continue
            changed_files.append(path)
            changed_source_paths.append(source_path)
            updated_manifest[source_path] = fingerprint

        # Idempotent source refresh: remove previous chunks for these files first.
        for modality in (Modality.TEXT, Modality.TABLE, Modality.IMAGE):
            self.store.delete_by_source(
                target_collection,
                modality.value,
                changed_source_paths,
            )
        self.lexical_index.delete_by_source(target_collection, changed_source_paths)

        chunks = ingest_files(changed_files, self.settings) if changed_files else []
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
        self._save_manifest(target_collection, updated_manifest)

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
        retrieval_mode: str | None = None,
        tenant_id: str | None = None,
    ) -> QueryAnswer:
        target_collection = self._scoped_collection(collection, tenant_id)
        per_modality_k = top_k or self.settings.retrieval_top_k_per_modality
        mode = self._resolve_retrieval_mode(retrieval_mode, self._default_retrieval_mode())

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

        dense_fused = reciprocal_rank_fusion(
            [text_hits, table_hits, image_hits],
            k=self.settings.retrieval_rrf_k,
            weights=[
                self.settings.retrieval_rrf_weight_text,
                self.settings.retrieval_rrf_weight_table,
                self.settings.retrieval_rrf_weight_image,
            ],
        )

        if mode == "dense_only":
            final_hits = self._diversify_hits(dense_fused)
        else:
            lexical_hits = self.lexical_index.search(
                target_collection,
                question,
                top_k=self.settings.retrieval_top_k_lexical,
            )
            hybrid_fused = reciprocal_rank_fusion(
                [text_hits, table_hits, image_hits, lexical_hits],
                k=self.settings.retrieval_rrf_k,
                weights=[
                    self.settings.retrieval_rrf_weight_text,
                    self.settings.retrieval_rrf_weight_table,
                    self.settings.retrieval_rrf_weight_image,
                    self.settings.retrieval_rrf_weight_lexical,
                ],
            )
            if mode == "hybrid":
                final_hits = self._diversify_hits(hybrid_fused)
            else:
                rerank_pool = hybrid_fused[: self.settings.retrieval_rerank_candidates]
                reranked = self.reranker.rerank(
                    question,
                    rerank_pool,
                    top_k=self.settings.max_context_chunks,
                )
                final_hits = self._diversify_hits(reranked)

        answer = self.synthesizer.generate(question, final_hits)
        citations = self._build_citations(final_hits)
        return QueryAnswer(
            answer=answer,
            hits=final_hits,
            citations=citations,
            retrieval_mode=mode,
        )
