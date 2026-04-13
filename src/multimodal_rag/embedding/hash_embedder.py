from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np


class HashEmbedder:
    """Deterministic local embedding fallback that needs no network/model downloads."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
