from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import orjson

from multimodal_rag.models import Chunk, RetrievalHit

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


@dataclass(slots=True)
class _LexicalState:
    chunks: list[Chunk]
    tokenized: list[list[str]]
    bm25: object | None = None


class LexicalIndex:
    """Persistent lexical index used for sparse retrieval (BM25)."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[float, _LexicalState]] = {}
        self._bm25_class = self._resolve_bm25_class()

    @staticmethod
    def _resolve_bm25_class():
        try:
            from rank_bm25 import BM25Okapi

            return BM25Okapi
        except Exception:
            return None

    def _index_path(self, collection: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", collection.strip().lower())
        return self.base_dir / f"{safe}.json"

    def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        path = self._index_path(collection)
        existing: dict[str, dict] = {}
        if path.exists():
            payload = orjson.loads(path.read_bytes())
            existing = {str(item["chunk_id"]): item for item in payload}

        for chunk in chunks:
            if not chunk.content.strip():
                continue
            existing[chunk.chunk_id] = chunk.to_payload()

        payload = list(existing.values())
        path.write_bytes(orjson.dumps(payload))
        self._cache.pop(collection, None)
        return len(chunks)

    def delete_by_source(self, collection: str, source_paths: list[str]) -> int:
        if not source_paths:
            return 0

        path = self._index_path(collection)
        if not path.exists():
            return 0

        payload = orjson.loads(path.read_bytes())
        source_set = set(source_paths)
        kept = [item for item in payload if str(item.get("source_path", "")) not in source_set]
        removed = len(payload) - len(kept)
        if removed <= 0:
            return 0

        path.write_bytes(orjson.dumps(kept))
        self._cache.pop(collection, None)
        return removed

    def _load_state(self, collection: str) -> _LexicalState:
        path = self._index_path(collection)
        if not path.exists():
            return _LexicalState(chunks=[], tokenized=[], bm25=None)

        mtime = path.stat().st_mtime
        cached = self._cache.get(collection)
        if cached and cached[0] == mtime:
            return cached[1]

        payload = orjson.loads(path.read_bytes())
        chunks = [Chunk.from_payload(item) for item in payload]
        tokenized = [_tokenize(chunk.content) for chunk in chunks]
        bm25 = None
        if self._bm25_class and tokenized:
            bm25 = self._bm25_class(tokenized)

        state = _LexicalState(chunks=chunks, tokenized=tokenized, bm25=bm25)
        self._cache[collection] = (mtime, state)
        return state

    @staticmethod
    def _fallback_scores(query_tokens: list[str], tokenized_docs: list[list[str]]) -> list[float]:
        if not query_tokens:
            return [0.0 for _ in tokenized_docs]
        query_set = set(query_tokens)
        scores: list[float] = []
        for doc_tokens in tokenized_docs:
            if not doc_tokens:
                scores.append(0.0)
                continue
            overlap = 0
            for token in doc_tokens:
                if token in query_set:
                    overlap += 1
            scores.append(overlap / len(doc_tokens))
        return scores

    def search(self, collection: str, query: str, top_k: int) -> list[RetrievalHit]:
        state = self._load_state(collection)
        if not state.chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        if state.bm25 is not None:
            raw_scores = state.bm25.get_scores(query_tokens)  # type: ignore[call-arg]
            scores = [float(score) for score in raw_scores]
        else:
            scores = self._fallback_scores(query_tokens, state.tokenized)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[RetrievalHit] = []
        for idx in order[:top_k]:
            score = scores[idx]
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    chunk=state.chunks[idx],
                    score=score,
                    backend="bm25" if state.bm25 is not None else "lexical-fallback",
                )
            )
        return hits


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalHit]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalHit]:
    """Merge ranked lists with RRF: score(d)=sum(1/(k+rank_i(d)))."""
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError("weights length must match result_lists length")

    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, Chunk] = {}

    for list_index, results in enumerate(result_lists):
        weight = 1.0 if weights is None else float(weights[list_index])
        if weight <= 0.0:
            continue
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (weight / (k + rank))
            chunk_lookup[chunk_id] = hit.chunk

    ordered = sorted(fused_scores, key=lambda item: fused_scores[item], reverse=True)
    return [
        RetrievalHit(chunk=chunk_lookup[chunk_id], score=fused_scores[chunk_id], backend="rrf")
        for chunk_id in ordered
    ]
