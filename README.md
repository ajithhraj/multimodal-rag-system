# Multimodal RAG System

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/api-fastapi-009688)
![Vector DB](https://img.shields.io/badge/vector-qdrant%20%7C%20faiss-5b4bdb)
![License](https://img.shields.io/badge/license-MIT-green)

Production-style multimodal Retrieval-Augmented Generation (RAG) system for real-world documents.

## Overview

This project ingests and queries:
- PDFs (layout-aware text + table extraction)
- images (vision captions + optional OCR)
- CSV/TSV tabular files

It ships with:
- a reusable Python package (`src` layout)
- a CLI (`mmrag`) for local workflows
- a FastAPI REST service for product integration
- Docker support for deployable runtime

## What Makes It Strong

- Hybrid retrieval: dense vector search + lexical BM25
- Reciprocal Rank Fusion (RRF) for robust ranking
- Weighted RRF controls to tune text/table/image/lexical influence
- Adaptive corrective retrieval fallback for low-coverage first-pass results
- Optional cross-encoder reranker for precision
- Result diversification (near-duplicate suppression + per-source balancing)
- Citation-rich answers (`source`, `modality`, `page`, `excerpt`)
- Pluggable vector backends (`faiss` and `qdrant`)
- Layout-aware PDF ingestion with table-region text dedup
- Adaptive chunk sizing by section style (narrative/procedural/table-like)
- Idempotent source refresh on re-ingest (no duplicate chunk buildup)
- Incremental ingest skips unchanged files by fingerprint for faster refresh cycles

## System Architecture

```mermaid
flowchart LR
    A["Input Files (PDF, Image, CSV/TSV)"] --> B["Ingestion"]
    B --> B1["PDF Text + Tables"]
    B --> B2["Image Caption + OCR"]
    B --> B3["Tabular Normalization"]
    B1 --> C["Structure-Aware Chunking + Metadata"]
    B2 --> C
    B3 --> C
    C --> D["Embedding Layer"]
    D --> E["Vector Store (FAISS / Qdrant)"]
    C --> F["Lexical Index (BM25)"]
    E --> G["Dense Retrieval"]
    F --> H["Lexical Retrieval"]
    G --> I["RRF Fusion"]
    H --> I
    I --> J["Optional Cross-Encoder Rerank"]
    J --> K["LLM Synthesis"]
    K --> L["Answer + Citations"]
```

## Quickstart

```bash
cd multimodal-rag-system
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,vision]"
copy .env.example .env
```

```bash
mmrag ingest ./data --tenant acme
mmrag ask "What are the major metrics shown in the latest PDF tables?" --tenant acme
mmrag ask "Summarize contract risks" --tenant acme --retrieval-mode hybrid_rerank
mmrag ask "Find charts similar to this trend" --image ./data/query_chart.png --tenant acme
mmrag serve --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Docker

```bash
docker compose up --build
```

## API Surface

- `GET /health`
- `POST /ingest-paths`
- `POST /ingest-files`
- `POST /query`
- `POST /query-multimodal` (multipart: question + optional image)

`POST /query` returns:
- `answer`
- `sources` (retrieved chunks + score)
- `citations` (source file, modality, page number, excerpt)
- `retrieval_mode` (effective mode used by the engine)
- `corrected` (whether fallback corrective retrieval was applied)
- `retrieval_diagnostics` (initial/final retrieval quality stats)
- `latency_ms` (request retrieval+generation time)
- accepts optional `retrieval_mode` (`dense_only`, `hybrid`, `hybrid_rerank`)

Example multimodal query:

```bash
curl -X POST http://localhost:8000/query-multimodal \
  -H "X-Tenant-ID: acme" \
  -F "question=Find similar chart patterns" \
  -F "image=@./data/query_chart.png"
```

## Multi-Tenant & API Auth

Tenant scoping is enabled by default across ingestion and retrieval:
- each tenant writes to an isolated namespaced collection (`tenant-<id>__<collection>`)
- CLI supports `--tenant` on `ingest`, `ask`, and `eval`
- REST API uses `X-Tenant-ID` (configurable via `MMRAG_AUTH_TENANT_HEADER`)

Optional API-key auth (recommended for deployments):
- set `MMRAG_AUTH_ENABLED=true`
- configure `MMRAG_AUTH_TENANT_API_KEYS` as `tenant_a:key_a,tenant_b:key_b`
- send the API key header (default `X-API-Key`)
- when a tenant header is supplied, it must match the tenant bound to the API key

## Benchmark Section (Portfolio-Friendly)

Use this section to publish your own numbers after running on your dataset.

| Scenario | Data Size | Hardware | Avg Query Latency | p95 Latency | Notes |
|---|---:|---|---:|---:|---|
| Dense only | TBD | TBD | TBD | TBD | baseline |
| Dense + BM25 + RRF | TBD | TBD | TBD | TBD | hybrid |
| Hybrid + reranker | TBD | TBD | TBD | TBD | highest precision |

Suggested evaluation metrics to report:
- `Recall@k`
- `MRR`
- answer groundedness / citation precision
- end-to-end latency

## Evaluation Harness

Run structured retrieval benchmarks with the built-in evaluator:

```bash
mmrag eval ./eval/datasets/starter_eval.jsonl --ingest-path ./data --tenant acme --k-values 1,3,5,10
```

Run retrieval strategy ablations:

```bash
mmrag eval ./eval/datasets/starter_eval.jsonl --ingest-path ./data --tenant acme --ablation
```

What it reports:
- `Recall@k` (for configured k values)
- `MRR`
- citation hit-rate
- mean citation precision
- average and p95 latency

The command saves a JSON report under `.rag_store/eval_reports/` by default.

## Screenshots To Add

Add these images under `docs/assets/` for a polished portfolio presentation:
- API docs screenshot (`/docs`)
- ingestion CLI run
- query response with citations
- architecture diagram snapshot

## Resume-Ready Outcomes

This project demonstrates:
- end-to-end LLM product engineering (ingestion to API)
- retrieval engineering beyond dense-only pipelines
- production-aware Python packaging and deployment
- configurable AI systems with clean interfaces and fallbacks
- testing and code quality workflows (`pytest`, `ruff`)

## Configuration

Important env variables:
- `MMRAG_VECTOR_BACKEND`
- `MMRAG_STORAGE_DIR`
- `MMRAG_COLLECTION`
- `MMRAG_DEFAULT_TENANT`
- `MMRAG_CHUNK_SIZE`
- `MMRAG_CHUNK_OVERLAP`
- `MMRAG_INGESTION_SKIP_UNCHANGED_FILES`
- `MMRAG_ADAPTIVE_CHUNKING_ENABLED`
- `MMRAG_ADAPTIVE_CHUNKING_MIN_SIZE`
- `MMRAG_ADAPTIVE_CHUNKING_TABLE_FACTOR`
- `MMRAG_ADAPTIVE_CHUNKING_PROCEDURAL_FACTOR`
- `MMRAG_ADAPTIVE_CHUNKING_NARRATIVE_FACTOR`
- `MMRAG_ADAPTIVE_CHUNKING_OVERLAP_FACTOR`
- `MMRAG_OPENAI_API_KEY`
- `MMRAG_RETRIEVAL_TOP_K_PER_MODALITY`
- `MMRAG_RETRIEVAL_TOP_K_LEXICAL`
- `MMRAG_RETRIEVAL_RRF_K`
- `MMRAG_RETRIEVAL_RRF_WEIGHT_TEXT`
- `MMRAG_RETRIEVAL_RRF_WEIGHT_TABLE`
- `MMRAG_RETRIEVAL_RRF_WEIGHT_IMAGE`
- `MMRAG_RETRIEVAL_RRF_WEIGHT_LEXICAL`
- `MMRAG_RETRIEVAL_ENABLE_RERANKER`
- `MMRAG_RETRIEVAL_RERANKER_MODEL`
- `MMRAG_RETRIEVAL_RERANK_CANDIDATES`
- `MMRAG_RETRIEVAL_ENABLE_RESULT_DIVERSITY`
- `MMRAG_RETRIEVAL_MAX_CHUNKS_PER_SOURCE`
- `MMRAG_RETRIEVAL_DUPLICATE_SIMILARITY_THRESHOLD`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_ENABLED`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_MIN_HITS`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_MIN_UNIQUE_SOURCES`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_MIN_UNIQUE_MODALITIES`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_TARGET_MODE`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_TOP_K_MULTIPLIER`
- `MMRAG_RETRIEVAL_AUTO_CORRECT_LEXICAL_MULTIPLIER`
- `MMRAG_AUTH_ENABLED`
- `MMRAG_AUTH_API_KEY_HEADER`
- `MMRAG_AUTH_TENANT_HEADER`
- `MMRAG_AUTH_TENANT_API_KEYS`
- `MMRAG_QDRANT_URL`
- `MMRAG_QDRANT_API_KEY`
- `MMRAG_QDRANT_PATH`

## CLI

- `mmrag ingest <path> [--tenant <tenant-id>]`
- `mmrag ask <question> [--tenant <tenant-id>] [--retrieval-mode <mode>]`
- `mmrag ask <question> --image <path-to-image> [--tenant <tenant-id>] [--retrieval-mode <mode>]`
- `mmrag serve`
- `mmrag eval <dataset-path> [--tenant <tenant-id>] [--ablation]`

Run `mmrag --help` for full options.

## Development

```bash
ruff check src tests
pytest -q
```

## Project Structure

```text
src/multimodal_rag/
  ingestion/      # loaders, chunking, pdf/image/table extraction
  embedding/      # text + vision embedders
  storage/        # faiss and qdrant backends
  retrieval/      # lexical index, RRF fusion, reranker
  generation/     # answer synthesis
  api/            # FastAPI app and schemas
```
