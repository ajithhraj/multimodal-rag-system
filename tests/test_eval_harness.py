from pathlib import Path

import pytest

from multimodal_rag.eval import (
    load_eval_cases,
    parse_k_values,
    parse_retrieval_modes,
    run_ablation_evaluation,
    run_evaluation,
)
from multimodal_rag.eval.models import EvalCase
from multimodal_rag.models import Citation, Chunk, Modality, QueryAnswer, RetrievalHit


class EvalEngineStub:
    def __init__(self):
        self.calls: list[dict[str, str | None]] = []

    def query(
        self,
        question,
        collection=None,
        tenant_id=None,
        top_k=None,
        query_image_path=None,
        retrieval_mode=None,
    ):
        self.calls.append(
            {
                "question": question,
                "collection": collection,
                "tenant_id": tenant_id,
                "retrieval_mode": retrieval_mode,
            }
        )
        mode = retrieval_mode or "hybrid"
        if question == "q1":
            if mode == "dense_only":
                ordered_hits = [
                    RetrievalHit(
                        chunk=Chunk(
                            chunk_id="miss-dense",
                            source_path="docs/noise.pdf",
                            modality=Modality.TEXT,
                            content="Noisy context",
                        ),
                        score=0.95,
                        backend="stub",
                    ),
                    RetrievalHit(
                        chunk=Chunk(
                            chunk_id="hit-1",
                            source_path="docs/report.pdf",
                            modality=Modality.TEXT,
                            content="Revenue was 25M",
                        ),
                        score=0.85,
                        backend="stub",
                    ),
                ]
            else:
                ordered_hits = [
                    RetrievalHit(
                        chunk=Chunk(
                            chunk_id="hit-1",
                            source_path="docs/report.pdf",
                            modality=Modality.TEXT,
                            content="Revenue was 25M",
                        ),
                        score=0.9,
                        backend="stub",
                    )
                ]

            hit = RetrievalHit(
                chunk=ordered_hits[0].chunk,
                score=ordered_hits[0].score,
                backend=ordered_hits[0].backend,
            )
            citation = Citation(
                chunk_id=hit.chunk.chunk_id,
                source_path=hit.chunk.source_path,
                modality=hit.chunk.modality,
                page_number=2,
                excerpt="Revenue was 25M",
            )
            return QueryAnswer(answer="stub-1", hits=ordered_hits, citations=[citation], retrieval_mode=mode)

        hit_1 = RetrievalHit(
            chunk=Chunk(
                chunk_id="miss-1",
                source_path="docs/other.pdf",
                modality=Modality.TEXT,
                content="Not relevant",
            ),
            score=0.8,
            backend="stub",
        )
        hit_2 = RetrievalHit(
            chunk=Chunk(
                chunk_id="hit-2",
                source_path="data/table.csv",
                modality=Modality.TABLE,
                content="Table row",
            ),
            score=0.7,
            backend="stub",
        )
        ordered = [hit_2, hit_1] if mode != "dense_only" else [hit_1, hit_2]
        return QueryAnswer(answer="stub-2", hits=ordered, citations=[], retrieval_mode=mode)


def test_parse_k_values():
    assert parse_k_values("1, 3, 5,3") == [1, 3, 5]
    with pytest.raises(ValueError):
        parse_k_values("")
    with pytest.raises(ValueError):
        parse_k_values("0,2")


def test_parse_retrieval_modes():
    assert parse_retrieval_modes("dense_only, hybrid ,hybrid_rerank") == [
        "dense_only",
        "hybrid",
        "hybrid_rerank",
    ]
    with pytest.raises(ValueError):
        parse_retrieval_modes("")
    with pytest.raises(ValueError):
        parse_retrieval_modes("hybrid,unknown")


def test_load_eval_cases_jsonl_auto_case_id(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '\n'.join(
            [
                '{"question":"q1","expected_chunk_ids":["x"]}',
                '{"case_id":"custom_case","question":"q2","expected_source_paths":["report.pdf"]}',
            ]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(dataset)
    assert len(cases) == 2
    assert cases[0].case_id == "case_001"
    assert cases[1].case_id == "custom_case"


def test_run_evaluation_metrics():
    engine = EvalEngineStub()
    cases = [
        EvalCase(case_id="c1", question="q1", expected_chunk_ids=["hit-1"]),
        EvalCase(case_id="c2", question="q2", expected_source_paths=["table.csv"]),
    ]
    report = run_evaluation(
        engine=engine,
        cases=cases,
        dataset_path=Path("eval/datasets/starter_eval.jsonl"),
        default_collection=None,
        k_values=[1, 2],
        default_tenant="tenant-default",
    )

    summary = report.summary
    assert summary.total_cases == 2
    assert summary.retrieval_evaluable_cases == 2
    assert summary.citation_evaluable_cases == 2
    assert summary.mean_mrr is not None
    assert summary.mean_recall_at["1"] == pytest.approx(1.0)
    assert summary.mean_recall_at["2"] == pytest.approx(1.0)
    assert summary.mean_mrr == pytest.approx(1.0)
    assert summary.citation_hit_rate == pytest.approx(0.5)
    assert summary.mean_citation_precision == pytest.approx(0.5)
    assert summary.avg_latency_ms >= 0.0
    assert engine.calls[0]["tenant_id"] == "tenant-default"
    assert engine.calls[1]["tenant_id"] == "tenant-default"


def test_run_ablation_evaluation_produces_lifts():
    engine = EvalEngineStub()
    cases = [
        EvalCase(case_id="c1", question="q1", expected_chunk_ids=["hit-1"]),
        EvalCase(case_id="c2", question="q2", expected_source_paths=["table.csv"]),
    ]
    ablation = run_ablation_evaluation(
        engine=engine,
        cases=cases,
        dataset_path=Path("eval/datasets/starter_eval.jsonl"),
        default_collection=None,
        k_values=[1, 2],
        modes=["dense_only", "hybrid"],
        baseline_mode="dense_only",
        default_tenant="tenant-default",
    )
    assert ablation.baseline_mode == "dense_only"
    assert set(ablation.mode_reports.keys()) == {"dense_only", "hybrid"}
    assert len(ablation.deltas_vs_baseline) == 1
    delta = ablation.deltas_vs_baseline[0]
    assert delta.mode == "hybrid"
    assert delta.mean_mrr_delta is not None
    assert delta.mean_mrr_delta > 0


def test_run_evaluation_case_tenant_overrides_default():
    engine = EvalEngineStub()
    cases = [
        EvalCase(case_id="c1", question="q1", tenant_id="tenant-a", expected_chunk_ids=["hit-1"]),
        EvalCase(case_id="c2", question="q2", expected_source_paths=["table.csv"]),
    ]
    _ = run_evaluation(
        engine=engine,
        cases=cases,
        dataset_path=Path("eval/datasets/starter_eval.jsonl"),
        default_collection="shared",
        k_values=[1],
        default_tenant="tenant-default",
    )
    assert engine.calls[0]["tenant_id"] == "tenant-a"
    assert engine.calls[1]["tenant_id"] == "tenant-default"
