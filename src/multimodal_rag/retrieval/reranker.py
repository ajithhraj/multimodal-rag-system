from __future__ import annotations

import logging

from multimodal_rag.models import RetrievalHit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Optional cross-encoder reranker.

    Keeps the pipeline usable even when sentence-transformers is unavailable.
    """

    def __init__(self, enabled: bool, model_name: str):
        self._enabled = enabled
        self._model = None
        if not enabled:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except Exception as exc:  # pragma: no cover - dependency/runtime branch
            logger.warning("Cross-encoder reranker unavailable, using fused ranking only: %s", exc)
            self._model = None

    @property
    def is_enabled(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []

        if not self._model:
            sorted_hits = sorted(hits, key=lambda item: item.score, reverse=True)
            return sorted_hits[:top_k]

        pairs = [(query, hit.chunk.content) for hit in hits]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:  # pragma: no cover - model runtime branch
            logger.warning("Cross-encoder scoring failed, using fused ranking only: %s", exc)
            sorted_hits = sorted(hits, key=lambda item: item.score, reverse=True)
            return sorted_hits[:top_k]

        reranked = sorted(
            (
                RetrievalHit(
                    chunk=hit.chunk,
                    score=float(score),
                    backend="cross-encoder",
                )
                for hit, score in zip(hits, scores, strict=False)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return reranked[:top_k]
