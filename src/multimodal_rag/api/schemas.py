from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestPathsRequest(BaseModel):
    paths: list[str]
    collection: str | None = None


class ResetCollectionRequest(BaseModel):
    collection: str | None = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    text: int
    table: int
    image: int


class ResetCollectionResponse(BaseModel):
    collection: str
    vector_removed: int
    lexical_removed: int
    manifest_removed: int


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


class ProvenanceSourceItem(BaseModel):
    source_path: str
    display_name: str
    modality: str
    chunk_count: int = 0
    best_score: float | None = None
    page_numbers: list[int] = Field(default_factory=list)


class QueryProvenance(BaseModel):
    grounded: bool = True
    source_count: int = 0
    citation_count: int = 0
    modalities: list[str] = Field(default_factory=list)
    retrieval_mode: str | None = None
    corrected: bool = False
    query_variants: list[str] = Field(default_factory=list)
    top_sources: list[ProvenanceSourceItem] = Field(default_factory=list)


class SourcePreviewExcerpt(BaseModel):
    chunk_id: str
    modality: str
    page_number: int | None = None
    excerpt: str


class SourcePreviewResponse(BaseModel):
    source_path: str
    display_name: str
    exists: bool
    byte_size: int | None = None
    updated_at: datetime | None = None
    chunk_count: int = 0
    modality_counts: dict[str, int] = Field(default_factory=dict)
    excerpts: list[SourcePreviewExcerpt] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    citations: list[CitationItem]
    retrieval_mode: str | None = None
    corrected: bool = False
    grounded: bool = True
    provenance: QueryProvenance = Field(default_factory=QueryProvenance)
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
