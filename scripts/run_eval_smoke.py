# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from multimodal_rag.config import Settings
from multimodal_rag.engine import MultimodalRAG
from multimodal_rag.eval import load_eval_cases, run_evaluation


def _prepare_workspace(root: Path) -> tuple[Path, Path]:
    if root.exists():
        shutil.rmtree(root)
    corpus_dir = root / "corpus"
    eval_dir = root / "eval"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = corpus_dir / "ops_metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "metric,target,notes",
                "sla_target,99.9%,production services",
                "retention_days,365,audit logs",
                "encryption_standard,AES-256,data at rest",
            ]
        ),
        encoding="utf-8",
    )

    dataset_path = eval_dir / "smoke_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"case_id":"sla","question":"What SLA target is listed?","expected_source_paths":["ops_metrics.csv"]}',
                '{"case_id":"encryption","question":"Which encryption standard is required?","expected_source_paths":["ops_metrics.csv"]}',
            ]
        ),
        encoding="utf-8",
    )
    return corpus_dir, dataset_path


def main() -> int:
    root = Path(".ci_tmp")
    corpus_dir, dataset_path = _prepare_workspace(root)

    settings = Settings(
        storage_dir=root / ".rag_store",
        retrieval_enable_reranker=False,
        retrieval_enable_result_diversity=False,
        retrieval_auto_correct_enabled=False,
    )
    engine = MultimodalRAG(settings=settings)
    ingest_stats = engine.ingest_paths([corpus_dir])
    if ingest_stats["chunks"] <= 0:
        raise RuntimeError("Eval smoke failed: ingestion produced no chunks.")

    cases = load_eval_cases(dataset_path)
    report = run_evaluation(
        engine=engine,
        cases=cases,
        dataset_path=dataset_path,
        default_collection=None,
        default_tenant=None,
        k_values=[1, 3],
        retrieval_mode="hybrid",
    )

    summary = report.summary
    if summary.total_cases < 2:
        raise RuntimeError("Eval smoke failed: expected at least 2 evaluation cases.")
    if summary.mean_recall_at.get("1", 0.0) <= 0.0:
        raise RuntimeError("Eval smoke failed: expected non-zero Recall@1.")
    if summary.citation_hit_rate is None or summary.citation_hit_rate <= 0.0:
        raise RuntimeError("Eval smoke failed: expected non-zero citation hit-rate.")

    print("Eval smoke passed.")
    print(f"Total cases: {summary.total_cases}")
    print(f"Mean Recall@1: {summary.mean_recall_at.get('1', 0.0):.4f}")
    print(f"Citation hit-rate: {summary.citation_hit_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
