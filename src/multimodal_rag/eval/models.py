from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str | None = None
    question: str = Field(min_length=1)
    query_image_path: str | None = None
    collection: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_source_paths: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseEvaluation(BaseModel):
    case_id: str
    retrieval_mode: str | None = None
    latency_ms: float
    hit_count: int
    citation_count: int
    expected_items: int
    matched_items: int
    mrr: float | None = None
    recall_at: dict[str, float] = Field(default_factory=dict)
    citation_hit: bool | None = None
    citation_precision: float | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_source_paths: list[str] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    total_cases: int
    retrieval_evaluable_cases: int
    citation_evaluable_cases: int
    avg_latency_ms: float
    p95_latency_ms: float
    mean_mrr: float | None = None
    mean_recall_at: dict[str, float] = Field(default_factory=dict)
    citation_hit_rate: float | None = None
    mean_citation_precision: float | None = None


class EvaluationReport(BaseModel):
    dataset_path: str
    k_values: list[int]
    generated_at_utc: str
    retrieval_mode: str | None = None
    summary: EvaluationSummary
    cases: list[CaseEvaluation]


class AblationDelta(BaseModel):
    mode: str
    avg_latency_ms_delta: float
    p95_latency_ms_delta: float
    mean_mrr_delta: float | None = None
    mean_recall_at_delta: dict[str, float] = Field(default_factory=dict)
    citation_hit_rate_delta: float | None = None
    mean_citation_precision_delta: float | None = None


class AblationReport(BaseModel):
    dataset_path: str
    k_values: list[int]
    generated_at_utc: str
    baseline_mode: str
    mode_reports: dict[str, EvaluationReport]
    deltas_vs_baseline: list[AblationDelta]
