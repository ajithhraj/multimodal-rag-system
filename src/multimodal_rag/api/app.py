from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, UploadFile

from multimodal_rag.api.deps import get_engine
from multimodal_rag.api.schemas import (
    IngestPathsRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from multimodal_rag.engine import MultimodalRAG


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal RAG API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest-paths", response_model=IngestResponse)
    def ingest_paths(
        payload: IngestPathsRequest,
        engine: MultimodalRAG = Depends(get_engine),
    ) -> IngestResponse:
        stats = engine.ingest_paths([Path(path) for path in payload.paths], collection=payload.collection)
        return IngestResponse(**stats)

    @app.post("/ingest-files", response_model=IngestResponse)
    async def ingest_files(
        files: list[UploadFile] = File(...),
        collection: str | None = None,
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

        stats = engine.ingest_paths(saved_paths, collection=collection)
        return IngestResponse(**stats)

    @app.post("/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        engine: MultimodalRAG = Depends(get_engine),
    ) -> QueryResponse:
        result = engine.query(
            question=payload.question,
            collection=payload.collection,
            top_k=payload.top_k,
        )
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
        )

    return app


app = create_app()
