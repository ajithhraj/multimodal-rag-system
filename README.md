# Multimodal RAG System

A production-pattern Retrieval-Augmented Generation (RAG) package that ingests and queries:

- PDFs (text + extracted tables)
- Images (vision captions + optional OCR)
- Structured table files (CSV/TSV)

It includes:

- Clean Python package (`src/` layout)
- CLI (`mmrag`)
- REST API (FastAPI)
- Pluggable vector backends (FAISS fallback mode + Qdrant)
- Optional Vision LLM captioning
- Docker support

## Architecture

1. **Ingestion**
   - Parse PDF text
   - Extract PDF tables
   - Caption images with a vision model (optional)
   - Build normalized multimodal chunks
2. **Embedding**
   - Text/table embedding
   - Image-semantic embedding (caption + optional CLIP)
3. **Storage**
   - `faiss` backend (native FAISS if installed, NumPy fallback)
   - `qdrant` backend (embedded local path or remote server)
4. **Retrieval + Generation**
   - Per-modality retrieval
   - Score fusion
   - Context synthesis with LangChain chat models

## Quickstart

```bash
cd multimodal-rag-system
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,vision]"
copy .env.example .env
```

Ingest local files:

```bash
mmrag ingest ./data
```

Ask questions:

```bash
mmrag ask "What are the major metrics shown in the latest PDF tables?"
```

Run API:

```bash
mmrag serve --host 0.0.0.0 --port 8000
```

Swagger docs: `http://localhost:8000/docs`

## Docker

```bash
docker compose up --build
```

## Environment

Main env vars:

- `MMRAG_VECTOR_BACKEND` = `faiss` or `qdrant`
- `MMRAG_STORAGE_DIR` = local state directory (default: `.rag_store`)
- `MMRAG_OPENAI_API_KEY` = enables OpenAI text + vision generation
- `MMRAG_QDRANT_URL` + `MMRAG_QDRANT_API_KEY` for remote Qdrant
- `MMRAG_QDRANT_PATH` for embedded local Qdrant mode

Without external APIs, the system still works using deterministic local embedding fallbacks.

## CLI commands

- `mmrag ingest <path>`
- `mmrag ask <question>`
- `mmrag serve`

Run `mmrag --help` for full options.
