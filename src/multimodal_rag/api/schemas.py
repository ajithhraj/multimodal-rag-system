from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestPathsRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    collection: str | None = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    text: int
    table: int
    image: int


IngestJobState = Literal["queued", "running", "completed", "failed", "cancelled"]


class IngestJobResponse(BaseModel):
    job_id: str
    status: IngestJobState
    tenant_id: str
    collection: str | None = None
    paths: list[str]
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
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
