from __future__ import annotations

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


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    collection: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class SourceItem(BaseModel):
    chunk_id: str
    source_path: str
    modality: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
