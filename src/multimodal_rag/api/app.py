from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from multimodal_rag.api.deps import get_engine, resolve_tenant_id
from multimodal_rag.api.schemas import (
    CitationItem,
    IngestPathsRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from multimodal_rag.engine import MultimodalRAG


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal RAG API", version="0.1.0")

    def to_query_response(result, latency_ms: float) -> QueryResponse:
        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceItem(
                    chunk_id=hit.chunk.chunk_id,
                    source_path=hit.chunk.source_path,
                    modality=hit.chunk.modality.value,
                    score=hit.score,
                )
                for hit in result.hits
            ],
            citations=[
                CitationItem(
                    chunk_id=citation.chunk_id,
                    source_path=citation.source_path,
                    modality=citation.modality.value,
                    page_number=citation.page_number,
                    excerpt=citation.excerpt,
                )
                for citation in result.citations
            ],
            retrieval_mode=result.retrieval_mode,
            corrected=result.corrected,
            retrieval_diagnostics=result.retrieval_diagnostics,
            latency_ms=latency_ms,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest-paths", response_model=IngestResponse)
    def ingest_paths(
        payload: IngestPathsRequest,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> IngestResponse:
        stats = engine.ingest_paths(
            [Path(path) for path in payload.paths],
            collection=payload.collection,
            tenant_id=tenant_id,
        )
        return IngestResponse(**stats)

    @app.post("/ingest-files", response_model=IngestResponse)
    async def ingest_files(
        files: list[UploadFile] = File(...),
        collection: str | None = None,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> IngestResponse:
        upload_dir = engine.settings.storage_dir / "tmp_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for upload in files:
            target = upload_dir / upload.filename
            content = await upload.read()
            target.write_bytes(content)
            saved_paths.append(target)

        stats = engine.ingest_paths(
            saved_paths,
            collection=collection,
            tenant_id=tenant_id,
        )
        return IngestResponse(**stats)

    @app.post("/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> QueryResponse:
        start = perf_counter()
        result = engine.query(
            question=payload.question,
            collection=payload.collection,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            tenant_id=tenant_id,
        )
        latency_ms = (perf_counter() - start) * 1000.0
        return to_query_response(result, latency_ms=latency_ms)

    @app.post("/query-multimodal", response_model=QueryResponse)
    async def query_multimodal(
        question: str = Form(default=""),
        image: UploadFile | None = File(default=None),
        collection: str | None = Form(default=None),
        top_k: int | None = Form(default=None),
        retrieval_mode: Literal["dense_only", "hybrid", "hybrid_rerank"] | None = Form(default=None),
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> QueryResponse:
        prompt = question.strip()
        if not prompt and image is None:
            raise HTTPException(status_code=400, detail="Provide either question text or an image.")
        if top_k is not None and not (1 <= top_k <= 50):
            raise HTTPException(status_code=422, detail="top_k must be between 1 and 50.")

        query_image_path: Path | None = None
        if image is not None:
            query_dir = engine.settings.storage_dir / "tmp_queries"
            query_dir.mkdir(parents=True, exist_ok=True)
            filename = image.filename or "query_image.bin"
            query_image_path = query_dir / filename
            content = await image.read()
            query_image_path.write_bytes(content)

        query_text = prompt or "Find visually similar or related context for this image."
        start = perf_counter()
        result = engine.query(
            question=query_text,
            collection=collection,
            top_k=top_k,
            query_image_path=query_image_path,
            retrieval_mode=retrieval_mode,
            tenant_id=tenant_id,
        )
        latency_ms = (perf_counter() - start) * 1000.0
        return to_query_response(result, latency_ms=latency_ms)

    return app


app = create_app()
