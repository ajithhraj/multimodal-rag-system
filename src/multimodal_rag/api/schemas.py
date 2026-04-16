from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestPathsRequest(BaseModel):
    paths: list[str]
    collection: str | None = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    text: int
    table: int
    image: int


IngestJobStatus = Literal["pending", "running", "done", "error"]


class IngestJobItem(BaseModel):
    job_id: str
    status: IngestJobStatus
    created_at: datetime
    updated_at: datetime
    file_count: int | None = None
    result: IngestResponse | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    collection: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_mode: Literal["dense_only", "hybrid", "hybrid_rerank"] | None = None


class SourceItem(BaseModel):
    chunk_id: str
    source_path: str
    modality: str
    score: float


class CitationItem(BaseModel):
    chunk_id: str
    source_path: str
    modality: str
    page_number: int | None = None
    excerpt: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    citations: list[CitationItem]
    retrieval_mode: str | None = None
    corrected: bool = False
    grounded: bool = True
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
