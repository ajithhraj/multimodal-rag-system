from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

import orjson
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from multimodal_rag.api.deps import get_engine, resolve_tenant_id
from multimodal_rag.api.schemas import (
    CitationItem,
    IngestPathsRequest,
    IngestResponse,
    ProvenanceSourceItem,
    QueryRequest,
    QueryProvenance,
    QueryResponse,
    ResetCollectionRequest,
    ResetCollectionResponse,
    SourcePreviewExcerpt,
    SourcePreviewResponse,
    SourceItem,
)
from multimodal_rag.engine import MultimodalRAG

_JOB_STORE: dict[str, dict] = {}
_JOB_ORDER: deque[str] = deque()
_MAX_INGEST_JOBS = 500
_RATE_WINDOWS: dict[str, deque[float]] = defaultdict(deque)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_job(job_id: str, status: str, **extra) -> None:
    existing = _JOB_STORE.get(job_id)
    now = _now_utc_iso()
    if existing is None:
        _JOB_ORDER.append(job_id)
        record = {
            "job_id": job_id,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        _JOB_STORE[job_id] = record
    else:
        record = existing
        record["status"] = status
        record["updated_at"] = now
    record.update(extra)

    while len(_JOB_ORDER) > _MAX_INGEST_JOBS:
        oldest = _JOB_ORDER.popleft()
        _JOB_STORE.pop(oldest, None)


def _list_jobs(status: str | None = None, limit: int = 50) -> list[dict]:
    items = [_JOB_STORE[job_id] for job_id in reversed(_JOB_ORDER) if job_id in _JOB_STORE]
    if status:
        items = [item for item in items if item.get("status") == status]
    return items[:limit]


def _delete_job(job_id: str) -> bool:
    if job_id not in _JOB_STORE:
        return False
    _JOB_STORE.pop(job_id, None)
    try:
        _JOB_ORDER.remove(job_id)
    except ValueError:
        pass
    return True


def _check_rate_limit(tenant_id: str, rpm: int) -> None:
    if rpm <= 0:
        return
    now = time.monotonic()
    window = _RATE_WINDOWS[tenant_id]
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= rpm:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({rpm} req/min for tenant '{tenant_id}'). Retry after 60 s.",
            headers={"Retry-After": "60"},
        )
    window.append(now)


def create_app() -> FastAPI:
    app = FastAPI(title="Ragora API", version="0.3.0")
    _JOB_STORE.clear()
    _JOB_ORDER.clear()
    _RATE_WINDOWS.clear()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def rate_limit(
        request: Request,
        tenant_id: str = Depends(resolve_tenant_id),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> str:
        if engine.settings.rate_limit_enabled:
            _check_rate_limit(tenant_id, engine.settings.rate_limit_rpm)
        return tenant_id

    def _sse_event(event: str, payload: dict) -> str:
        json_payload = orjson.dumps(payload).decode("utf-8")
        return f"event: {event}\ndata: {json_payload}\n\n"

    def _resolve_source_file(path_value: str) -> Path:
        raw = Path(path_value).expanduser()
        resolved = (Path.cwd() / raw).resolve() if not raw.is_absolute() else raw.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="Source file not found.")
        return resolved

    def _ui_path() -> Path:
        repo_dir = Path(__file__).resolve().parents[3]
        package_dir = Path(__file__).resolve().parents[1]
        candidates = [
            repo_dir / "face_of_ragora.html",
            repo_dir / "multimodal_rag_ui.html",
            package_dir / "web" / "face_of_ragora.html",
            package_dir / "web" / "multimodal_rag_ui.html",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise HTTPException(status_code=404, detail="Ragora UI file not found.")

    def _fallback_stream_tokens(answer: str, chunk_size: int = 64):
        text = answer or ""
        if not text:
            return
        for index in range(0, len(text), chunk_size):
            yield text[index : index + chunk_size]

    def _service_unavailable(exc: Exception) -> HTTPException:
        return HTTPException(status_code=503, detail=str(exc))

    def _display_source_name(path_value: str) -> str:
        normalized = str(path_value).replace("\\", "/")
        last = normalized.rsplit("/", 1)[-1]
        return last[33:] if len(last) > 33 and last[:32].isalnum() and last[32] == "_" else last

    def _build_provenance(result) -> QueryProvenance:
        source_map: dict[str, dict] = {}
        for hit in result.hits:
            entry = source_map.setdefault(
                hit.chunk.source_path,
                {
                    "source_path": hit.chunk.source_path,
                    "display_name": _display_source_name(hit.chunk.source_path),
                    "modality": hit.chunk.modality.value,
                    "chunk_count": 0,
                    "best_score": None,
                    "page_numbers": set(),
                },
            )
            entry["chunk_count"] += 1
            if entry["best_score"] is None or float(hit.score) > float(entry["best_score"]):
                entry["best_score"] = float(hit.score)
            page_number = hit.chunk.metadata.get("page_number")
            if isinstance(page_number, int):
                entry["page_numbers"].add(page_number)
            elif isinstance(page_number, str) and page_number.isdigit():
                entry["page_numbers"].add(int(page_number))

        top_sources = [
            ProvenanceSourceItem(
                source_path=entry["source_path"],
                display_name=entry["display_name"],
                modality=entry["modality"],
                chunk_count=entry["chunk_count"],
                best_score=entry["best_score"],
                page_numbers=sorted(entry["page_numbers"]),
            )
            for entry in sorted(
                source_map.values(),
                key=lambda item: (
                    -(float(item["best_score"]) if item["best_score"] is not None else 0.0),
                    item["display_name"],
                ),
            )[:5]
        ]

        diagnostics = result.retrieval_diagnostics or {}
        return QueryProvenance(
            grounded=result.grounded,
            source_count=len(source_map),
            citation_count=len(result.citations),
            modalities=sorted({hit.chunk.modality.value for hit in result.hits}),
            retrieval_mode=result.retrieval_mode,
            corrected=result.corrected,
            query_variants=list(diagnostics.get("query_variants", []) or []),
            top_sources=top_sources,
        )

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
            provenance=_build_provenance(result),
            retrieval_diagnostics=result.retrieval_diagnostics,
            latency_ms=latency_ms,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(path=_ui_path(), media_type="text/html")

    @app.get("/ui")
    def ui() -> FileResponse:
        return root()

    @app.get("/source-file")
    def source_file(
        path: str,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> FileResponse:
        file_path = _resolve_source_file(path)
        return FileResponse(path=file_path, filename=file_path.name)

    @app.get("/source-preview", response_model=SourcePreviewResponse)
    def source_preview(
        path: str,
        collection: str | None = None,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> SourcePreviewResponse:
        preview = engine.source_preview(path, collection=collection, tenant_id=tenant_id, limit=6)
        raw = Path(path).expanduser()
        resolved = (Path.cwd() / raw).resolve() if not raw.is_absolute() else raw.resolve()
        exists = resolved.exists() and resolved.is_file()
        stat = resolved.stat() if exists else None
        return SourcePreviewResponse(
            source_path=path,
            display_name=_display_source_name(path),
            exists=exists,
            byte_size=stat.st_size if stat else None,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc) if stat else None,
            chunk_count=int(preview.get("chunk_count", 0)),
            modality_counts=dict(preview.get("modality_counts", {})),
            excerpts=[
                SourcePreviewExcerpt(**excerpt)
                for excerpt in list(preview.get("excerpts", []))
            ],
        )

    @app.post("/ingest-paths", response_model=IngestResponse)
    def ingest_paths(
        payload: IngestPathsRequest,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> IngestResponse:
        try:
            stats = engine.ingest_paths(
                [Path(path) for path in payload.paths],
                collection=payload.collection,
                tenant_id=tenant_id,
            )
        except RuntimeError as exc:
            raise _service_unavailable(exc) from exc
        return IngestResponse(**stats)

    @app.post("/reset-collection", response_model=ResetCollectionResponse)
    def reset_collection(
        payload: ResetCollectionRequest,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> ResetCollectionResponse:
        result = engine.reset_collection(collection=payload.collection, tenant_id=tenant_id)
        return ResetCollectionResponse(**result)

    @app.post("/ingest-files", status_code=202)
    async def ingest_files(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        collection: str | None = None,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> dict:
        upload_dir = engine.settings.storage_dir / "tmp_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for upload in files:
            safe_name = Path(upload.filename or "").name or "upload.bin"
            target = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
            content = await upload.read()
            target.write_bytes(content)
            saved_paths.append(target)

        job_id = str(uuid.uuid4())
        file_paths = [str(path) for path in saved_paths]
        _set_job(job_id, "pending", file_count=len(saved_paths), files=file_paths)

        def _run() -> None:
            try:
                _set_job(job_id, "running", files=file_paths)
                stats = engine.ingest_paths(saved_paths, collection=collection, tenant_id=tenant_id)
                _set_job(job_id, "done", result=stats, files=file_paths)
            except Exception as exc:
                _set_job(job_id, "error", error=str(exc), files=file_paths)

        background_tasks.add_task(_run)
        return {"job_id": job_id, "status": "pending", "file_count": len(saved_paths)}

    @app.get("/ingest-jobs")
    def list_ingest_jobs(
        status: Literal["pending", "running", "done", "error"] | None = None,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict:
        return {"jobs": _list_jobs(status=status, limit=limit)}

    @app.get("/ingest-jobs/{job_id}")
    def get_ingest_job(job_id: str) -> dict:
        job = _JOB_STORE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return job

    @app.delete("/ingest-jobs/{job_id}", status_code=204)
    def delete_ingest_job(job_id: str) -> Response:
        if not _delete_job(job_id):
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return Response(status_code=204)

    @app.post("/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> QueryResponse:
        start = perf_counter()
        try:
            result = engine.query(
                question=payload.question,
                collection=payload.collection,
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                tenant_id=tenant_id,
            )
        except RuntimeError as exc:
            raise _service_unavailable(exc) from exc
        latency_ms = (perf_counter() - start) * 1000.0
        return to_query_response(result, latency_ms=latency_ms)

    @app.post("/query-stream")
    def query_stream(
        payload: QueryRequest,
        tenant_id: str = Depends(rate_limit),
        engine: MultimodalRAG = Depends(get_engine),
    ) -> StreamingResponse:
        start = perf_counter()
        try:
            retrieval_result = engine.query(
                question=payload.question,
                collection=payload.collection,
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                tenant_id=tenant_id,
            )
        except RuntimeError as exc:
            raise _service_unavailable(exc) from exc
        retrieval_latency_ms = (perf_counter() - start) * 1000.0
        response = to_query_response(retrieval_result, latency_ms=retrieval_latency_ms)

        def stream():
            yield _sse_event(
                "meta",
                {
                    "retrieval_mode": response.retrieval_mode,
                    "corrected": response.corrected,
                    "grounded": response.grounded,
                    "latency_retrieval_ms": response.latency_ms,
                    "provenance": response.provenance.model_dump(),
                },
            )
            full_answer_parts: list[str] = []
            synthesizer = getattr(engine, "synthesizer", None)
            if synthesizer is not None and hasattr(synthesizer, "stream"):
                token_iter = synthesizer.stream(payload.question, retrieval_result.hits)
            else:
                token_iter = _fallback_stream_tokens(response.answer)
            for idx, token in enumerate(token_iter, start=1):
                full_answer_parts.append(token)
                yield _sse_event("token", {"index": idx, "delta": token})
            full_answer = "".join(full_answer_parts)
            yield _sse_event(
                "citations",
                {"citations": [citation.model_dump() for citation in response.citations]},
            )
            yield _sse_event(
                "done",
                {
                    "answer": full_answer or response.answer,
                    "source_count": len(response.sources),
                    "citation_count": len(response.citations),
                    "latency_ms": (perf_counter() - start) * 1000.0,
                    "provenance": response.provenance.model_dump(),
                },
            )

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/query-multimodal", response_model=QueryResponse)
    async def query_multimodal(
        question: str = Form(default=""),
        image: UploadFile | None = File(default=None),
        collection: str | None = Form(default=None),
        top_k: int | None = Form(default=None),
        retrieval_mode: Literal["dense_only", "hybrid", "hybrid_rerank"] | None = Form(default=None),
        tenant_id: str = Depends(rate_limit),
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
            filename = Path(image.filename or "query_image.bin").name
            query_image_path = query_dir / filename
            content = await image.read()
            query_image_path.write_bytes(content)

        query_text = prompt or "Find visually similar or related context for this image."
        start = perf_counter()
        try:
            result = engine.query(
                question=query_text,
                collection=collection,
                top_k=top_k,
                query_image_path=query_image_path,
                retrieval_mode=retrieval_mode,
                tenant_id=tenant_id,
            )
        except RuntimeError as exc:
            raise _service_unavailable(exc) from exc
        latency_ms = (perf_counter() - start) * 1000.0
        return to_query_response(result, latency_ms=latency_ms)

    return app


app = create_app()
