from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings

from multimodal_rag.config import Settings
from multimodal_rag.embedding.hash_embedder import HashEmbedder

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, settings: Settings):
        self._fallback = HashEmbedder(dimensions=384)
        self._openai = None
        if settings.openai_api_key:
            self._openai = OpenAIEmbeddings(
                model=settings.text_embedding_model,
                api_key=settings.openai_api_key,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._openai:
            try:
                return self._openai.embed_documents(texts)
            except Exception as exc:  # pragma: no cover - network/model branch
                logger.warning("OpenAI embeddings failed, using hash fallback: %s", exc)
        return self._fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._openai:
            try:
                return self._openai.embed_query(text)
            except Exception as exc:  # pragma: no cover - network/model branch
                logger.warning("OpenAI query embedding failed, using hash fallback: %s", exc)
        return self._fallback.embed_query(text)


class VisionEmbedder:
    """Image-aware embedder.

    Uses CLIP from sentence-transformers when available.
    Falls back to deterministic text hashing if CLIP is unavailable.
    """

    def __init__(self):
        self._fallback = HashEmbedder(dimensions=384)
        self._clip = None
        try:
            from sentence_transformers import SentenceTransformer

            self._clip = SentenceTransformer("clip-ViT-B-32")
        except Exception:
            self._clip = None

    def embed_images(self, image_paths: list[Path], fallback_texts: list[str]) -> list[list[float]]:
        if not image_paths:
            return []
        if self._clip:
            try:
                from PIL import Image

                images = [Image.open(path).convert("RGB") for path in image_paths]
                vectors = self._clip.encode(images, convert_to_numpy=True, normalize_embeddings=True)
                return vectors.astype(np.float32).tolist()
            except Exception as exc:  # pragma: no cover - model runtime branch
                logger.warning("CLIP image encoding failed, using text fallback: %s", exc)
        return self._fallback.embed_documents(fallback_texts)

    def embed_query(self, text: str) -> list[float]:
        if self._clip:
            try:
                vector = self._clip.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return vector.astype(np.float32).tolist()
            except Exception as exc:  # pragma: no cover - model runtime branch
                logger.warning("CLIP query encoding failed, using hash fallback: %s", exc)
        return self._fallback.embed_query(text)
