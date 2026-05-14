from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from multimodal_rag.config import Settings
from multimodal_rag.embedding.hash_embedder import HashEmbedder
from multimodal_rag.ingestion.vision import VisionCaptioner, run_ocr

logger = logging.getLogger(__name__)

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency branch
    OpenAIEmbeddings = None  # type: ignore[assignment]


class TextEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._fallback = HashEmbedder(dimensions=384)
        self._strict = settings.strict_api_only_mode()
        self._openai = None
        if settings.has_openai_api_key():
            if OpenAIEmbeddings is None:
                if self._strict:
                    raise RuntimeError(
                        "langchain_openai is required in API-only mode for OpenAI embeddings."
                    )
                logger.warning("langchain_openai not available, using local hash embeddings.")
            else:
                self._openai = OpenAIEmbeddings(
                    model=settings.text_embedding_model,
                    api_key=settings.openai_api_key,
                )
        elif self._strict:
            raise RuntimeError("OpenAI API key is required for embeddings in API-only mode.")

    @property
    def uses_openai(self) -> bool:
        return self._openai is not None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._openai:
            try:
                return self._openai.embed_documents(texts)
            except Exception as exc:  # pragma: no cover - network/model branch
                if self._strict:
                    raise RuntimeError(f"OpenAI document embeddings failed: {exc}") from exc
                logger.warning("OpenAI embeddings failed, using hash fallback: %s", exc)
        if self._strict:
            raise RuntimeError("OpenAI embeddings are unavailable in API-only mode.")
        return self._fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._openai:
            try:
                return self._openai.embed_query(text)
            except Exception as exc:  # pragma: no cover - network/model branch
                if self._strict:
                    raise RuntimeError(f"OpenAI query embedding failed: {exc}") from exc
                logger.warning("OpenAI query embedding failed, using hash fallback: %s", exc)
        if self._strict:
            raise RuntimeError("OpenAI query embeddings are unavailable in API-only mode.")
        return self._fallback.embed_query(text)


class VisionEmbedder:
    """Image-aware embedder.

    OpenAI mode:
    - indexes images using their caption/OCR text via the text embedding model
    - can caption a query image, then embed that caption for retrieval

    Local mode:
    - uses CLIP from sentence-transformers when available
    - falls back to deterministic text hashing if CLIP is unavailable
    """

    def __init__(self, settings: Settings, text_embedder: TextEmbedder):
        self.settings = settings
        self._text_embedder = text_embedder
        self._fallback = HashEmbedder(dimensions=384)
        self._strict = settings.strict_api_only_mode()
        self._captioner = VisionCaptioner(settings)
        self._clip = None
        self._openai_mode = text_embedder.uses_openai

        if not self._openai_mode:
            try:
                from sentence_transformers import SentenceTransformer

                self._clip = SentenceTransformer("clip-ViT-B-32")
            except Exception as exc:
                if self._strict:
                    raise RuntimeError(
                        "Local CLIP fallback is disabled in API-only mode and OpenAI image embeddings are unavailable."
                    ) from exc
                self._clip = None

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return " ".join((value or "").split()).strip()

    def _build_query_text(self, text: str, image_path: Path | None = None) -> str:
        parts: list[str] = []
        clean_text = self._clean_text(text)
        if clean_text:
            parts.append(clean_text)
        if image_path is not None and image_path.exists():
            caption = self._captioner.caption(image_path)
            if caption:
                parts.append(f"Image description: {caption}")
            if not self._strict:
                ocr_text = run_ocr(image_path)
                if ocr_text:
                    parts.append(f"OCR text: {ocr_text}")
            if len(parts) == (1 if clean_text else 0):
                parts.append(f"Image file name: {image_path.name}")
        return "\n\n".join(parts) if parts else "image retrieval query"

    def embed_images(self, image_paths: list[Path], fallback_texts: list[str]) -> list[list[float]]:
        if not image_paths:
            return []
        if self._openai_mode:
            return self._text_embedder.embed_documents(fallback_texts)
        if self._clip:
            try:
                from PIL import Image

                images = [Image.open(path).convert("RGB") for path in image_paths]
                vectors = self._clip.encode(images, convert_to_numpy=True, normalize_embeddings=True)
                return vectors.astype(np.float32).tolist()
            except Exception as exc:  # pragma: no cover - model runtime branch
                if self._strict:
                    raise RuntimeError(f"Local CLIP image encoding failed in API-only mode: {exc}") from exc
                logger.warning("CLIP image encoding failed, using text fallback: %s", exc)
        if self._strict:
            raise RuntimeError("OpenAI-backed image embeddings are unavailable in API-only mode.")
        return self._fallback.embed_documents(fallback_texts)

    def embed_query(self, text: str, image_path: Path | None = None) -> list[float]:
        if self._openai_mode:
            return self._text_embedder.embed_query(self._build_query_text(text, image_path=image_path))
        if self._clip:
            try:
                vector = self._clip.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return vector.astype(np.float32).tolist()
            except Exception as exc:  # pragma: no cover - model runtime branch
                if self._strict:
                    raise RuntimeError(f"Local CLIP query encoding failed in API-only mode: {exc}") from exc
                logger.warning("CLIP query encoding failed, using hash fallback: %s", exc)
        if self._strict:
            raise RuntimeError("Image query embeddings are unavailable in API-only mode.")
        return self._fallback.embed_query(text)
