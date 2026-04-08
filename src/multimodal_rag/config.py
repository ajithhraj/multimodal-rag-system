from functools import lru_cache
from pathlib import Path
import re
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MMRAG_",
        extra="ignore",
    )

    app_name: str = "Multimodal RAG System"
    storage_dir: Path = Path(".rag_store")
    collection: str = "default"
    default_tenant: str = "public"
    vector_backend: Literal["faiss", "qdrant"] = "faiss"

    chunk_size: int = 900
    chunk_overlap: int = 140
    adaptive_chunking_enabled: bool = True
    adaptive_chunking_min_size: int = Field(default=280, ge=64, le=4096)
    adaptive_chunking_table_factor: float = Field(default=0.58, ge=0.2, le=1.0)
    adaptive_chunking_procedural_factor: float = Field(default=0.78, ge=0.2, le=1.2)
    adaptive_chunking_narrative_factor: float = Field(default=1.0, ge=0.4, le=1.5)
    adaptive_chunking_overlap_factor: float = Field(default=0.8, ge=0.2, le=1.2)
    max_context_chunks: int = 8
    ingestion_skip_unchanged_files: bool = True

    orchestrator: Literal["langchain", "llamaindex"] = "langchain"

    openai_api_key: str | None = None
    chat_model: str = "gpt-4.1-mini"
    vision_model: str = "gpt-4.1-mini"
    text_embedding_model: str = "text-embedding-3-small"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path = Path(".rag_store/qdrant")
    qdrant_collection_prefix: str = "mmrag"

    retrieval_top_k_per_modality: int = Field(default=4, ge=1, le=50)
    retrieval_top_k_lexical: int = Field(default=12, ge=1, le=200)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=500)
    retrieval_rrf_weight_text: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieval_rrf_weight_table: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieval_rrf_weight_image: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieval_rrf_weight_lexical: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieval_enable_reranker: bool = False
    retrieval_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_rerank_candidates: int = Field(default=20, ge=1, le=200)
    retrieval_enable_result_diversity: bool = True
    retrieval_max_chunks_per_source: int = Field(default=3, ge=1, le=20)
    retrieval_duplicate_similarity_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    retrieval_auto_correct_enabled: bool = True
    retrieval_auto_correct_min_hits: int = Field(default=3, ge=1, le=20)
    retrieval_auto_correct_min_unique_sources: int = Field(default=2, ge=1, le=20)
    retrieval_auto_correct_min_unique_modalities: int = Field(default=2, ge=1, le=3)
    retrieval_auto_correct_target_mode: Literal["hybrid", "hybrid_rerank"] = "hybrid_rerank"
    retrieval_auto_correct_top_k_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    retrieval_auto_correct_lexical_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)

    auth_enabled: bool = False
    auth_api_key_header: str = "X-API-Key"
    auth_tenant_header: str = "X-Tenant-ID"
    # Format: tenant_a:key_a,tenant_b:key_b
    auth_tenant_api_keys: str | None = None

    @staticmethod
    def normalize_tenant_id(raw: str) -> str:
        tenant = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip().lower())
        tenant = tenant.strip("-_")
        return tenant or "public"

    def parse_tenant_key_map(self) -> dict[str, str]:
        value = (self.auth_tenant_api_keys or "").strip()
        if not value:
            return {}

        mapping: dict[str, str] = {}
        for item in value.split(","):
            pair = item.strip()
            if not pair:
                continue
            if ":" not in pair:
                continue
            tenant_raw, key_raw = pair.split(":", 1)
            tenant = self.normalize_tenant_id(tenant_raw)
            key = key_raw.strip()
            if not key:
                continue
            mapping[tenant] = key
        return mapping


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
