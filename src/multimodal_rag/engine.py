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
LEADING_PROMPT_RE = re.compile(
    r"^(what|which|who|when|where|why|how)\s+(is|are|was|were|do|does|did|the|a|an)?\s*",
    flags=re.IGNORECASE,
)


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

    @staticmethod
    def _quality_stats(hits: list[RetrievalHit]) -> dict[str, float | int]:
        unique_sources = len({hit.chunk.source_path for hit in hits})
        unique_modalities = len({hit.chunk.modality.value for hit in hits})
        top_score = hits[0].score if hits else 0.0
        return {
            "hit_count": len(hits),
            "unique_sources": unique_sources,
            "unique_modalities": unique_modalities,
            "top_score": float(top_score),
        }

    def _quality_tuple(self, hits: list[RetrievalHit]) -> tuple[int, int, int, float]:
        stats = self._quality_stats(hits)
        return (
            int(stats["hit_count"]),
            int(stats["unique_sources"]),
            int(stats["unique_modalities"]),
            float(stats["top_score"]),
        )

    def _needs_auto_correction(self, hits: list[RetrievalHit]) -> bool:
        stats = self._quality_stats(hits)
        return (
            int(stats["hit_count"]) < self.settings.retrieval_auto_correct_min_hits
            or int(stats["unique_sources"]) < self.settings.retrieval_auto_correct_min_unique_sources
            or int(stats["unique_modalities"]) < self.settings.retrieval_auto_correct_min_unique_modalities
        )

    def _retrieve_hits(
        self,
        target_collection: str,
        question: str,
        text_query_vec: list[float],
        vision_query_vec: list[float],
        per_modality_k: int,
        mode: RetrievalMode,
        lexical_top_k: int,
    ) -> list[RetrievalHit]:
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
            return self._diversify_hits(dense_fused)

        lexical_hits = self.lexical_index.search(
            target_collection,
            question,
            top_k=lexical_top_k,
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
            return self._diversify_hits(hybrid_fused)

        rerank_pool = hybrid_fused[: self.settings.retrieval_rerank_candidates]
        reranked = self.reranker.rerank(
            question,
            rerank_pool,
            top_k=self.settings.max_context_chunks,
        )
        return self._diversify_hits(reranked)

    @staticmethod
    def _normalize_query_variant(raw: str) -> str:
        text = raw.strip().strip("?.!,:;")
        text = LEADING_PROMPT_RE.sub("", text).strip()
        return text

    def _expand_query_variants(self, question: str) -> list[str]:
        base = question.strip()
        if not base:
            return [question]

        variants = [base]
        if not self.settings.retrieval_query_expansion_enabled:
            return variants

        normalized = self._normalize_query_variant(base)
        if normalized and normalized.lower() != base.lower():
            variants.append(normalized)

        split_parts = re.split(r"\b(?:and|or|vs|versus)\b|,|;|/|&", normalized, flags=re.IGNORECASE)
        for part in split_parts:
            candidate = self._normalize_query_variant(part)
            if len(candidate.split()) < 2:
                continue
            variants.append(candidate)

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            key = variant.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(variant.strip())
            if len(deduped) >= self.settings.retrieval_query_expansion_max_variants:
                break

        return deduped or [base]

    def _retrieve_hits_with_variants(
        self,
        target_collection: str,
        question: str,
        query_image_path: Path | None,
        primary_vision_query_vec: list[float] | None,
        per_modality_k: int,
        mode: RetrievalMode,
        lexical_top_k: int,
    ) -> tuple[list[RetrievalHit], list[str]]:
        variants = self._expand_query_variants(question)
        if not variants:
            variants = [question]

        variant_hits: list[list[RetrievalHit]] = []
        variant_weights: list[float] = []
        for index, variant in enumerate(variants):
            text_query_vec = self.text_embedder.embed_query(variant)
            if index == 0:
                vision_query_vec = primary_vision_query_vec
                if vision_query_vec is None:
                    vision_query_vec = self._build_vision_query_vector(variant, query_image_path)
            else:
                vision_query_vec = self.vision_embedder.embed_query(variant)

            hits = self._retrieve_hits(
                target_collection=target_collection,
                question=variant,
                text_query_vec=text_query_vec,
                vision_query_vec=vision_query_vec,
                per_modality_k=per_modality_k,
                mode=mode,
                lexical_top_k=lexical_top_k,
            )
            variant_hits.append(hits)
            variant_weights.append(1.0 if index == 0 else self.settings.retrieval_query_expansion_weight)

        if len(variant_hits) == 1:
            return variant_hits[0], variants

        merged = reciprocal_rank_fusion(
            variant_hits,
            k=self.settings.retrieval_rrf_k,
            weights=variant_weights,
        )
        return self._diversify_hits(merged), variants

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
        initial_mode = self._resolve_retrieval_mode(retrieval_mode, self._default_retrieval_mode())

        primary_vision_query_vec = self._build_vision_query_vector(question, query_image_path)

        initial_hits, query_variants = self._retrieve_hits_with_variants(
            target_collection=target_collection,
            question=question,
            query_image_path=query_image_path,
            primary_vision_query_vec=primary_vision_query_vec,
            per_modality_k=per_modality_k,
            mode=initial_mode,
            lexical_top_k=self.settings.retrieval_top_k_lexical,
        )

        final_hits = initial_hits
        final_mode = initial_mode
        corrected = False

        can_auto_correct = self.settings.retrieval_auto_correct_enabled and retrieval_mode is None
        if can_auto_correct and self._needs_auto_correction(initial_hits):
            corrected_mode = self.settings.retrieval_auto_correct_target_mode
            corrected_per_modality_k = min(
                50,
                max(
                    per_modality_k,
                    int(round(per_modality_k * self.settings.retrieval_auto_correct_top_k_multiplier)),
                ),
            )
            corrected_lexical_top_k = min(
                200,
                max(
                    self.settings.retrieval_top_k_lexical,
                    int(
                        round(
                            self.settings.retrieval_top_k_lexical
                            * self.settings.retrieval_auto_correct_lexical_multiplier
                        )
                    ),
                ),
            )
            corrected_hits, _ = self._retrieve_hits_with_variants(
                target_collection=target_collection,
                question=question,
                query_image_path=query_image_path,
                primary_vision_query_vec=primary_vision_query_vec,
                per_modality_k=corrected_per_modality_k,
                mode=corrected_mode,
                lexical_top_k=corrected_lexical_top_k,
            )
            if self._quality_tuple(corrected_hits) > self._quality_tuple(initial_hits):
                final_hits = corrected_hits
                final_mode = corrected_mode
                corrected = True

        answer = self.synthesizer.generate(question, final_hits)
        citations = self._build_citations(final_hits)
        grounded = len(citations) >= self.settings.response_min_citations
        if self.settings.response_require_citations and not grounded:
            answer = self.settings.response_ungrounded_fallback_text
        return QueryAnswer(
            answer=answer,
            hits=final_hits,
            citations=citations,
            retrieval_mode=final_mode,
            corrected=corrected,
            grounded=grounded,
            retrieval_diagnostics={
                "initial_mode": initial_mode,
                "final_mode": final_mode,
                "initial_quality": self._quality_stats(initial_hits),
                "final_quality": self._quality_stats(final_hits),
                "auto_correction_enabled": can_auto_correct,
                "query_variants": query_variants,
            },
        )
