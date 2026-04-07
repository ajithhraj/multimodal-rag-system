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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
