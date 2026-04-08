from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import orjson

from multimodal_rag.eval.models import (
    AblationDelta,
    AblationReport,
    CaseEvaluation,
    EvalCase,
    EvaluationReport,
    EvaluationSummary,
)
from multimodal_rag.models import Citation, RetrievalHit

RETRIEVAL_MODES = ("dense_only", "hybrid", "hybrid_rerank")


def parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("k values must be positive integers")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("Provide at least one k value")
    return sorted(values)


def parse_retrieval_modes(raw: str) -> list[str]:
    modes: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"Unsupported retrieval mode '{mode}'. Allowed: {', '.join(RETRIEVAL_MODES)}"
            )
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    if not modes:
        raise ValueError("Provide at least one retrieval mode")
    return modes


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _source_matches(expected_source: str, actual_source: str) -> bool:
    expected_norm = _normalize_path(expected_source)
    actual_norm = _normalize_path(actual_source)
    if expected_norm == actual_norm:
        return True
    expected_name = Path(expected_source).name.lower()
    actual_name = Path(actual_source).name.lower()
    return expected_name == actual_name


def _matched_expected_keys_for_hit(hit: RetrievalHit, case: EvalCase) -> set[str]:
    keys: set[str] = set()
    for chunk_id in case.expected_chunk_ids:
        if hit.chunk.chunk_id == chunk_id:
            keys.add(f"id:{chunk_id}")
    for index, source in enumerate(case.expected_source_paths):
        if _source_matches(source, hit.chunk.source_path):
            keys.add(f"src:{index}")
    return keys


def _matched_expected_keys_for_citation(citation: Citation, case: EvalCase) -> set[str]:
    keys: set[str] = set()
    for chunk_id in case.expected_chunk_ids:
        if citation.chunk_id == chunk_id:
            keys.add(f"id:{chunk_id}")
    for index, source in enumerate(case.expected_source_paths):
        if _source_matches(source, citation.source_path):
            keys.add(f"src:{index}")
    return keys


def _expected_item_count(case: EvalCase) -> int:
    return len(case.expected_chunk_ids) + len(case.expected_source_paths)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return float(ordered[rank])


def _delta(new_value: float | None, base_value: float | None) -> float | None:
    if new_value is None or base_value is None:
        return None
    return new_value - base_value


def load_eval_cases(dataset_path: Path) -> list[EvalCase]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {dataset_path}")

    cases: list[EvalCase] = []
    suffix = dataset_path.suffix.lower()

    if suffix == ".jsonl":
        for line_no, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            payload = orjson.loads(raw)
            case = EvalCase.model_validate(payload)
            if not case.case_id:
                case.case_id = f"case_{line_no:03d}"
            cases.append(case)
    elif suffix == ".json":
        payload = orjson.loads(dataset_path.read_bytes())
        if not isinstance(payload, list):
            raise ValueError("JSON evaluation datasets must contain a list of case objects")
        for idx, item in enumerate(payload, start=1):
            case = EvalCase.model_validate(item)
            if not case.case_id:
                case.case_id = f"case_{idx:03d}"
            cases.append(case)
    else:
        raise ValueError("Unsupported dataset format. Use .jsonl or .json")

    if not cases:
        raise ValueError("No evaluation cases found in dataset")
    return cases


def run_evaluation(
    engine,
    cases: list[EvalCase],
    dataset_path: Path,
    default_collection: str | None,
    k_values: list[int],
    default_tenant: str | None = None,
    retrieval_mode: str | None = None,
) -> EvaluationReport:
    latencies: list[float] = []
    case_metrics: list[CaseEvaluation] = []

    mrr_values: list[float] = []
    recall_sums = {str(k): 0.0 for k in k_values}
    retrieval_evaluable = 0

    citation_hits: list[float] = []
    citation_precisions: list[float] = []
    citation_evaluable = 0

    dataset_dir = dataset_path.parent

    for case in cases:
        query_image_path: Path | None = None
        if case.query_image_path:
            candidate = Path(case.query_image_path)
            if not candidate.is_absolute():
                candidate = (dataset_dir / candidate).resolve()
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Query image for case '{case.case_id}' not found: {candidate}"
                )
            query_image_path = candidate

        start = perf_counter()
        result = engine.query(
            question=case.question,
            collection=case.collection or default_collection,
            tenant_id=case.tenant_id or default_tenant,
            top_k=case.top_k,
            query_image_path=query_image_path,
            retrieval_mode=retrieval_mode,
        )
        latency_ms = (perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        expected_count = _expected_item_count(case)
        matched_expected = 0
        mrr: float | None = None
        recall_at: dict[str, float] = {}
        citation_hit: bool | None = None
        citation_precision: float | None = None

        if expected_count > 0:
            retrieval_evaluable += 1
            matched_top_all: set[str] = set()

            first_relevant_rank: int | None = None
            for rank, hit in enumerate(result.hits, start=1):
                matched = _matched_expected_keys_for_hit(hit, case)
                if matched and first_relevant_rank is None:
                    first_relevant_rank = rank
                matched_top_all.update(matched)

            matched_expected = len(matched_top_all)
            mrr_value = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
            mrr = mrr_value
            mrr_values.append(mrr_value)

            for k in k_values:
                matched_top_k: set[str] = set()
                for hit in result.hits[:k]:
                    matched_top_k.update(_matched_expected_keys_for_hit(hit, case))
                recall = len(matched_top_k) / expected_count
                recall_at[str(k)] = recall
                recall_sums[str(k)] += recall

            citation_evaluable += 1
            if result.citations:
                matched_citations = 0
                for citation in result.citations:
                    if _matched_expected_keys_for_citation(citation, case):
                        matched_citations += 1
                citation_hit_value = matched_citations > 0
                citation_precision = matched_citations / len(result.citations)
            else:
                citation_hit_value = False
                citation_precision = 0.0

            citation_hit = citation_hit_value
            citation_hits.append(1.0 if citation_hit_value else 0.0)
            citation_precisions.append(citation_precision)

        case_metrics.append(
            CaseEvaluation(
                case_id=case.case_id or "unknown_case",
                retrieval_mode=result.retrieval_mode,
                latency_ms=latency_ms,
                hit_count=len(result.hits),
                citation_count=len(result.citations),
                expected_items=expected_count,
                matched_items=matched_expected,
                mrr=mrr,
                recall_at=recall_at,
                citation_hit=citation_hit,
                citation_precision=citation_precision,
                retrieved_chunk_ids=[hit.chunk.chunk_id for hit in result.hits],
                retrieved_source_paths=[hit.chunk.source_path for hit in result.hits],
            )
        )

    mean_recall_at = {
        key: (value / retrieval_evaluable if retrieval_evaluable else 0.0)
        for key, value in recall_sums.items()
    }

    summary = EvaluationSummary(
        total_cases=len(cases),
        retrieval_evaluable_cases=retrieval_evaluable,
        citation_evaluable_cases=citation_evaluable,
        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 95),
        mean_mrr=(sum(mrr_values) / len(mrr_values)) if mrr_values else None,
        mean_recall_at=mean_recall_at,
        citation_hit_rate=(sum(citation_hits) / len(citation_hits)) if citation_hits else None,
        mean_citation_precision=(
            sum(citation_precisions) / len(citation_precisions)
            if citation_precisions
            else None
        ),
    )

    return EvaluationReport(
        dataset_path=str(dataset_path),
        k_values=k_values,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        retrieval_mode=retrieval_mode,
        summary=summary,
        cases=case_metrics,
    )


def run_ablation_evaluation(
    engine,
    cases: list[EvalCase],
    dataset_path: Path,
    default_collection: str | None,
    k_values: list[int],
    modes: list[str],
    baseline_mode: str,
    default_tenant: str | None = None,
) -> AblationReport:
    if baseline_mode not in modes:
        raise ValueError("baseline_mode must be present in ablation modes")

    mode_reports: dict[str, EvaluationReport] = {}
    for mode in modes:
        mode_reports[mode] = run_evaluation(
            engine=engine,
            cases=cases,
            dataset_path=dataset_path,
            default_collection=default_collection,
            k_values=k_values,
            default_tenant=default_tenant,
            retrieval_mode=mode,
        )

    baseline_summary = mode_reports[baseline_mode].summary
    deltas: list[AblationDelta] = []
    for mode in modes:
        if mode == baseline_mode:
            continue
        summary = mode_reports[mode].summary

        recall_delta: dict[str, float] = {}
        keys = set(baseline_summary.mean_recall_at) | set(summary.mean_recall_at)
        for key in sorted(keys, key=lambda v: int(v)):
            recall_delta[key] = summary.mean_recall_at.get(key, 0.0) - baseline_summary.mean_recall_at.get(
                key, 0.0
            )

        deltas.append(
            AblationDelta(
                mode=mode,
                avg_latency_ms_delta=summary.avg_latency_ms - baseline_summary.avg_latency_ms,
                p95_latency_ms_delta=summary.p95_latency_ms - baseline_summary.p95_latency_ms,
                mean_mrr_delta=_delta(summary.mean_mrr, baseline_summary.mean_mrr),
                mean_recall_at_delta=recall_delta,
                citation_hit_rate_delta=_delta(summary.citation_hit_rate, baseline_summary.citation_hit_rate),
                mean_citation_precision_delta=_delta(
                    summary.mean_citation_precision,
                    baseline_summary.mean_citation_precision,
                ),
            )
        )

    return AblationReport(
        dataset_path=str(dataset_path),
        k_values=k_values,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline_mode=baseline_mode,
        mode_reports=mode_reports,
        deltas_vs_baseline=deltas,
    )


def save_evaluation_report(report: EvaluationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(report.model_dump(), option=orjson.OPT_INDENT_2))


def save_ablation_report(report: AblationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(report.model_dump(), option=orjson.OPT_INDENT_2))
