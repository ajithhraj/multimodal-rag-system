from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import orjson

from multimodal_rag.api.deps import get_engine, resolve_tenant_id
from multimodal_rag.api.jobs import IngestJobManager, IngestJobRecord
from multimodal_rag.api.rate_limit import RateLimiter
from multimodal_rag.api.schemas import (
    CitationItem,
    IngestJobResponse,
    IngestPathsRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from multimodal_rag.config import Settings, get_settings
from multimodal_rag.engine import MultimodalRAG
from multimodal_rag.observability import TelemetryManager


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    telemetry = TelemetryManager(runtime_settings)
    telemetry.setup()
    ingest_jobs = IngestJobManager(
        max_workers=runtime_settings.ingestion_jobs_max_workers,
        ttl_seconds=runtime_settings.ingestion_jobs_ttl_seconds,
        max_retained=runtime_settings.ingestion_jobs_max_retained,
    )
    rate_limiter = RateLimiter(
        requests_per_minute=runtime_settings.rate_limit_requests_per_minute,
        burst=runtime_settings.rate_limit_burst,
    )
    rate_limited_paths = {
        "/query",
        "/query-stream",
        "/query-multimodal",
    }
    app = FastAPI(title="Multimodal RAG API", version="0.1.0")
    app.state.settings = runtime_settings
    app.state.telemetry = telemetry
    app.state.ingest_jobs = ingest_jobs
    app.state.rate_limiter = rate_limiter

    def _sse_event(event: str, payload: dict) -> str:
        json_payload = orjson.dumps(payload).decode("utf-8")
        return f"event: {event}\ndata: {json_payload}\n\n"

    def _answer_chunks(answer: str, chunk_size: int = 64):
        clean = answer or ""
        if not clean:
            yield ""
            return
        for index in range(0, len(clean), chunk_size):
            yield clean[index : index + chunk_size]

    def _request_route(request: Request) -> str:
        route = request.scope.get("route")
        route_path = getattr(route, "path", None)
        if isinstance(route_path, str) and route_path:
            return route_path
        return request.url.path

    @app.middleware("http")
    async def instrument_request(request: Request, call_next):
        request_id_header = runtime_settings.request_id_header
        request_id = telemetry.generate_request_id(request.headers.get(request_id_header))
        request.state.request_id = request_id

        route = _request_route(request)
        start = perf_counter()
        if runtime_settings.rate_limit_enabled and route in rate_limited_paths:
            tenant_raw = request.headers.get(runtime_settings.auth_tenant_header)
            tenant_id = runtime_settings.normalize_tenant_id(tenant_raw or runtime_settings.default_tenant)
            key = f"{tenant_id}:{request.method}:{route}"
            allowed, retry_after = rate_limiter.allow(key)
            if not allowed:
                duration_ms = (perf_counter() - start) * 1000.0
                telemetry.record_http(request.method, route, status_code=429, duration_ms=duration_ms)
                retry_after_seconds = max(1, int(math.ceil(retry_after)))
                headers = {
                    request_id_header: request_id,
                    "Retry-After": str(retry_after_seconds),
                }
                return JSONResponse(
                    status_code=429,
                    headers=headers,
                    content={"detail": "Rate limit exceeded. Retry later."},
                )

        with telemetry.timed_span(
            "http.request",
            attributes={
                "http.method": request.method,
                "http.route": route,
                "mmrag.request_id": request_id,
            },
        ):
            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception:
                duration_ms = (perf_counter() - start) * 1000.0
                telemetry.record_http(request.method, route, status_code=500, duration_ms=duration_ms)
                raise

        duration_ms = (perf_counter() - start) * 1000.0
        response.headers[request_id_header] = request_id
        telemetry.record_http(request.method, route, status_code=status_code, duration_ms=duration_ms)
        return response

    @app.on_event("shutdown")
    def _shutdown_job_executor() -> None:
        ingest_jobs.shutdown()

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
            grounded=result.grounded,
            retrieval_diagnostics=result.retrieval_diagnostics,
            latency_ms=latency_ms,
        )

    def _to_ingest_job_response(job: IngestJobRecord) -> IngestJobResponse:
        return IngestJobResponse(
            job_id=job.job_id,
            status=job.status,
            tenant_id=job.tenant_id,
            collection=job.collection,
            paths=job.paths,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            result=job.result,
            error=job.error,
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

    @app.post("/ingest-jobs", response_model=IngestJobResponse, status_code=202)
    def create_ingest_job(
        payload: IngestPathsRequest,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> IngestJobResponse:
        if not runtime_settings.ingestion_jobs_enabled:
            raise HTTPException(status_code=503, detail="Async ingestion jobs are disabled.")
        paths = [Path(path) for path in payload.paths]
        job = ingest_jobs.submit(
            engine=engine,
            paths=paths,
            collection=payload.collection,
            tenant_id=tenant_id,
        )
        return _to_ingest_job_response(job)

    @app.get("/ingest-jobs/{job_id}", response_model=IngestJobResponse)
    def get_ingest_job(job_id: str) -> IngestJobResponse:
        if not runtime_settings.ingestion_jobs_enabled:
            raise HTTPException(status_code=503, detail="Async ingestion jobs are disabled.")
        job = ingest_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        return _to_ingest_job_response(job)

    @app.post("/ingest-jobs/{job_id}/cancel", response_model=IngestJobResponse)
    def cancel_ingest_job(job_id: str) -> IngestJobResponse:
        if not runtime_settings.ingestion_jobs_enabled:
            raise HTTPException(status_code=503, detail="Async ingestion jobs are disabled.")
        job = ingest_jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        if job.status != "cancelled":
            raise HTTPException(status_code=409, detail=f"Cannot cancel job in '{job.status}' state.")
        return _to_ingest_job_response(job)

    @app.get("/ingest-jobs", response_model=list[IngestJobResponse])
    def list_ingest_jobs(limit: int = Query(default=50, ge=1, le=200)) -> list[IngestJobResponse]:
        if not runtime_settings.ingestion_jobs_enabled:
            raise HTTPException(status_code=503, detail="Async ingestion jobs are disabled.")
        return [_to_ingest_job_response(job) for job in ingest_jobs.list(limit=limit)]

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
        request: Request,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> QueryResponse:
        start = perf_counter()
        with telemetry.timed_span(
            "rag.query",
            attributes={
                "http.route": "/query",
                "mmrag.request_id": getattr(request.state, "request_id", None),
            },
        ):
            result = engine.query(
                question=payload.question,
                collection=payload.collection,
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                tenant_id=tenant_id,
            )
        latency_ms = (perf_counter() - start) * 1000.0
        telemetry.record_query(
            route="/query",
            retrieval_mode=result.retrieval_mode,
            corrected=result.corrected,
            grounded=result.grounded,
            duration_ms=latency_ms,
        )
        return to_query_response(result, latency_ms=latency_ms)

    @app.post("/query-stream")
    def query_stream(
        payload: QueryRequest,
        request: Request,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> StreamingResponse:
        start = perf_counter()
        with telemetry.timed_span(
            "rag.query.stream",
            attributes={
                "http.route": "/query-stream",
                "mmrag.request_id": getattr(request.state, "request_id", None),
            },
        ):
            result = engine.query(
                question=payload.question,
                collection=payload.collection,
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                tenant_id=tenant_id,
            )
        latency_ms = (perf_counter() - start) * 1000.0
        telemetry.record_query(
            route="/query-stream",
            retrieval_mode=result.retrieval_mode,
            corrected=result.corrected,
            grounded=result.grounded,
            duration_ms=latency_ms,
        )
        response = to_query_response(result, latency_ms=latency_ms)

        def stream():
            yield _sse_event(
                "meta",
                {
                    "retrieval_mode": response.retrieval_mode,
                    "corrected": response.corrected,
                    "grounded": response.grounded,
                    "latency_ms": response.latency_ms,
                },
            )
            for index, delta in enumerate(_answer_chunks(response.answer), start=1):
                yield _sse_event("token", {"index": index, "delta": delta})
            yield _sse_event(
                "citations",
                {"citations": [citation.model_dump() for citation in response.citations]},
            )
            yield _sse_event(
                "done",
                {
                    "answer": response.answer,
                    "source_count": len(response.sources),
                    "citation_count": len(response.citations),
                },
            )

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/query-multimodal", response_model=QueryResponse)
    async def query_multimodal(
        request: Request,
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
        with telemetry.timed_span(
            "rag.query.multimodal",
            attributes={
                "http.route": "/query-multimodal",
                "mmrag.request_id": getattr(request.state, "request_id", None),
            },
        ):
            result = engine.query(
                question=query_text,
                collection=collection,
                top_k=top_k,
                query_image_path=query_image_path,
                retrieval_mode=retrieval_mode,
                tenant_id=tenant_id,
            )
        latency_ms = (perf_counter() - start) * 1000.0
        telemetry.record_query(
            route="/query-multimodal",
            retrieval_mode=result.retrieval_mode,
            corrected=result.corrected,
            grounded=result.grounded,
            duration_ms=latency_ms,
        )
        return to_query_response(result, latency_ms=latency_ms)

    return app


app = create_app()
