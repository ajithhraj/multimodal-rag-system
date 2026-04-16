# Multimodal RAG System

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/api-fastapi-009688)
![Vector DB](https://img.shields.io/badge/vector-qdrant%20%7C%20faiss-5b4bdb)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade **Multimodal Retrieval-Augmented Generation (RAG)** system that ingests and queries PDFs, images, and tabular data — returning grounded, citation-rich answers via CLI or REST API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Docker](#docker)
- [CLI Reference](#cli-reference)
- [REST API](#rest-api)
- [Multi-Tenancy & Auth](#multi-tenancy--auth)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Development](#development)

---

## Overview

This system handles real-world document types end-to-end:

| Input Type | How It's Processed |
|---|---|
| **PDF** | Layout-aware text extraction + table region detection with dedup |
| **Image** | Vision-based captioning + optional OCR |
| **CSV / TSV** | Tabular normalization into structured chunks |

Retrieval goes beyond naive dense search — it combines **dense vector search**, **BM25 lexical search**, and **Reciprocal Rank Fusion (RRF)** with an optional cross-encoder reranker for maximum precision.

---

## Architecture

```
Input Files (PDF, Image, CSV/TSV)
        │
        ▼
   [Ingestion Layer]
   ├── PDF: Text + Tables
   ├── Image: Caption + OCR
   └── CSV/TSV: Tabular Normalization
        │
        ▼
[Structure-Aware Chunking + Metadata]
   (adaptive sizing: narrative / procedural / table-like)
        │
        ├──────────────────────┐
        ▼                      ▼
 [Embedding Layer]     [Lexical Index (BM25)]
        │
        ▼
[Vector Store: FAISS / Qdrant]
        │
        ├─── Dense Retrieval ──┐
        │                      ▼
        └──────────── [RRF Fusion]
                               │
                               ▼
                  [Optional Cross-Encoder Rerank]
                               │
                               ▼
                       [LLM Synthesis]
                               │
                               ▼
                  Answer + Citations (source, page, excerpt)
```

---

## Features

- **Hybrid retrieval** — dense vector search + BM25 lexical search fused via RRF
- **Optional cross-encoder reranker** for higher answer precision
- **Citation-rich responses** — every answer includes `source`, `modality`, `page`, and `excerpt`
- **Pluggable vector backends** — switch between `faiss` (local) and `qdrant` (scalable)
- **Layout-aware PDF parsing** — table-region text deduplication, structure detection
- **Adaptive chunking** — chunk sizes tune automatically based on content style (narrative, procedural, table-like)
- **Multi-tenant support** — isolated namespaced collections per tenant
- **Optional API key auth** — configurable per-tenant API key enforcement
- **FastAPI REST service** — full OpenAPI docs at `/docs`
- **CLI** — `mmrag` command for local ingest, query, serve, and eval workflows
- **Docker support** — single `docker compose up --build` deployment
- **Built-in eval harness** — Recall@k, MRR, citation precision, latency benchmarks

---

## Project Structure

```
multimodal-rag-system/
├── src/
│   └── multimodal_rag/
│       ├── ingestion/       # Document loaders, chunking, PDF/image/table extraction
│       ├── embedding/       # Text and vision embedders
│       ├── storage/         # FAISS and Qdrant vector store backends
│       ├── retrieval/       # BM25 lexical index, RRF fusion, cross-encoder reranker
│       ├── generation/      # LLM answer synthesis with citations
│       └── api/             # FastAPI app, routes, and schemas
├── eval/
│   └── datasets/            # Evaluation JSONL datasets
├── tests/                   # Pytest test suite
├── docs/
│   └── assets/              # Screenshots and architecture diagrams
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/ajithhraj/multimodal-rag-system.git
cd multimodal-rag-system

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -e ".[dev,vision]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set MMRAG_OPENAI_API_KEY and other settings
```

### 3. Ingest and query

```bash
# Ingest a folder of documents
mmrag ingest ./data --tenant acme

# Ask a text question
mmrag ask "What are the major metrics shown in the latest PDF tables?" --tenant acme

# Ask with an image
mmrag ask "Find charts similar to this trend" --image ./data/query_chart.png --tenant acme

# Start the REST server
mmrag serve --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

---

## CLI Reference

| Command | Description |
|---|---|
| `mmrag ingest <path> [--tenant <id>]` | Ingest a file or directory |
| `mmrag ask "<question>" [--tenant <id>]` | Query with a text question |
| `mmrag ask "<question>" --image <path> [--tenant <id>]` | Multimodal query with an image |
| `mmrag serve [--host] [--port]` | Start the FastAPI server |
| `mmrag eval <dataset.jsonl> [--tenant <id>] [--ablation]` | Run evaluation benchmarks |

Run `mmrag --help` for full options.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ingest-paths` | Ingest from server-side file paths |
| `POST` | `/ingest-files` | Ingest from uploaded files (multipart) |
| `POST` | `/query` | Text query |
| `POST` | `/query-multimodal` | Multimodal query (text + optional image) |

### Example: Text query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d '{"question": "What metrics are in the Q3 report?"}'
```

### Example: Multimodal query

```bash
curl -X POST http://localhost:8000/query-multimodal \
  -H "X-Tenant-ID: acme" \
  -F "question=Find similar chart patterns" \
  -F "image=@./data/query_chart.png"
```

### Response shape (`/query`)

```json
{
  "answer": "...",
  "sources": [
    { "chunk": "...", "score": 0.92 }
  ],
  "citations": [
    { "source": "report.pdf", "modality": "pdf", "page": 4, "excerpt": "..." }
  ]
}
```

---

## Multi-Tenancy & Auth

Every ingestion and retrieval call is scoped to a tenant, stored in an isolated namespace (`tenant-<id>__<collection>`).

**CLI:** use `--tenant <id>` on `ingest`, `ask`, and `eval`.

**REST API:** pass `X-Tenant-ID: <id>` header (header name configurable via `MMRAG_AUTH_TENANT_HEADER`).

### Optional API key auth

Set the following in `.env` to enable:

```env
MMRAG_AUTH_ENABLED=true
MMRAG_AUTH_TENANT_API_KEYS=acme:key_acme,contoso:key_contoso
```

When enabled, send the API key in the request header (default: `X-API-Key`). The tenant in the header must match the tenant bound to the key.

---

## Configuration

All settings are controlled via environment variables (see `.env.example`).

| Variable | Description |
|---|---|
| `MMRAG_OPENAI_API_KEY` | OpenAI API key for embeddings and generation |
| `MMRAG_VECTOR_BACKEND` | `faiss` (default) or `qdrant` |
| `MMRAG_STORAGE_DIR` | Local storage directory (default: `.rag_store`) |
| `MMRAG_COLLECTION` | Vector store collection name |
| `MMRAG_DEFAULT_TENANT` | Default tenant ID when none is specified |
| `MMRAG_CHUNK_SIZE` | Base chunk size in tokens |
| `MMRAG_CHUNK_OVERLAP` | Token overlap between chunks |
| `MMRAG_ADAPTIVE_CHUNKING_ENABLED` | Enable content-aware adaptive chunking |
| `MMRAG_ADAPTIVE_CHUNKING_MIN_SIZE` | Minimum adaptive chunk size |
| `MMRAG_ADAPTIVE_CHUNKING_TABLE_FACTOR` | Chunk size multiplier for table content |
| `MMRAG_ADAPTIVE_CHUNKING_PROCEDURAL_FACTOR` | Chunk size multiplier for procedural content |
| `MMRAG_ADAPTIVE_CHUNKING_NARRATIVE_FACTOR` | Chunk size multiplier for narrative content |
| `MMRAG_ADAPTIVE_CHUNKING_OVERLAP_FACTOR` | Overlap scaling factor |
| `MMRAG_RETRIEVAL_TOP_K_PER_MODALITY` | Top-K chunks retrieved per modality |
| `MMRAG_RETRIEVAL_TOP_K_LEXICAL` | Top-K results from BM25 lexical retrieval |
| `MMRAG_RETRIEVAL_RRF_K` | RRF constant (typically 60) |
| `MMRAG_RETRIEVAL_ENABLE_RERANKER` | Enable cross-encoder reranking |
| `MMRAG_RETRIEVAL_RERANKER_MODEL` | Reranker model name |
| `MMRAG_RETRIEVAL_RERANK_CANDIDATES` | Number of candidates passed to reranker |
| `MMRAG_AUTH_ENABLED` | Enable API key auth |
| `MMRAG_AUTH_API_KEY_HEADER` | Header name for API key (default: `X-API-Key`) |
| `MMRAG_AUTH_TENANT_HEADER` | Header name for tenant ID (default: `X-Tenant-ID`) |
| `MMRAG_AUTH_TENANT_API_KEYS` | Comma-separated `tenant:key` pairs |
| `MMRAG_QDRANT_URL` | Qdrant server URL (if using Qdrant backend) |
| `MMRAG_QDRANT_API_KEY` | Qdrant API key |
| `MMRAG_QDRANT_PATH` | Local Qdrant path (for embedded mode) |

---

## Evaluation

The built-in eval harness benchmarks retrieval quality end-to-end.

```bash
# Standard eval
mmrag eval ./eval/datasets/starter_eval.jsonl \
  --ingest-path ./data \
  --tenant acme \
  --k-values 1,3,5,10

# Retrieval strategy ablation (dense vs hybrid vs hybrid+reranker)
mmrag eval ./eval/datasets/starter_eval.jsonl \
  --ingest-path ./data \
  --tenant acme \
  --ablation
```

Reports include:

- `Recall@k` (for each configured k)
- `MRR` (Mean Reciprocal Rank)
- Citation hit-rate and mean citation precision
- Average and p95 query latency

Reports are saved to `.rag_store/eval_reports/` as JSON.

### Benchmark Template

| Scenario | Data Size | Hardware | Avg Latency | p95 Latency | Notes |
|---|---|---|---|---|---|
| Dense only | TBD | TBD | TBD | TBD | Baseline |
| Dense + BM25 + RRF | TBD | TBD | TBD | TBD | Hybrid |
| Hybrid + reranker | TBD | TBD | TBD | TBD | Highest precision |

---

## Development

```bash
# Lint
ruff check src tests

# Tests
pytest -q
```

---

## License

MIT
