from functools import lru_cache
from pathlib import Path
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
    retrieval_enable_reranker: bool = False
    retrieval_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_rerank_candidates: int = Field(default=20, ge=1, le=200)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
