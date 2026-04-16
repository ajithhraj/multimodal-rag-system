from pathlib import Path
import time

from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app
from multimodal_rag.api.deps import get_engine
from multimodal_rag.config import Settings
from multimodal_rag.models import Citation, Chunk, Modality, QueryAnswer, RetrievalHit


class StubEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_query_image_path = None
        self.last_tenant_id = None
        self.last_retrieval_mode = None

    def ingest_paths(self, paths, collection=None, tenant_id=None):
        self.last_tenant_id = tenant_id
        return {"files": len(paths), "chunks": 1, "text": 1, "table": 0, "image": 0}

    def query(
        self,
        question,
        collection=None,
        top_k=None,
        query_image_path=None,
        retrieval_mode=None,
        tenant_id=None,
    ):
        self.last_query_image_path = query_image_path
        self.last_tenant_id = tenant_id
        self.last_retrieval_mode = retrieval_mode
        hit = RetrievalHit(
            chunk=Chunk(
                chunk_id="x1",
                source_path="doc.pdf",
                modality=Modality.TEXT,
                content="ctx",
            ),
            score=0.92,
            backend="stub",
        )
        citation = Citation(
            chunk_id="x1",
            source_path="doc.pdf",
            modality=Modality.TEXT,
            page_number=3,
            excerpt="ctx",
        )
        return QueryAnswer(
            answer=f"stub:{question}",
            hits=[hit],
            citations=[citation],
            retrieval_mode="hybrid",
        )


def _build_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(storage_dir=tmp_path, **overrides)


def _build_client(engine: StubEngine) -> TestClient:
    app = create_app(engine.settings)
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


def test_health():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_health_uses_custom_request_id_header(tmp_path):
    settings = _build_settings(tmp_path, request_id_header="X-Trace-ID")
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Trace-ID" in response.headers
    assert response.headers["X-Trace-ID"]


def test_query_endpoint(tmp_path):
    settings = _build_settings(tmp_path)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "stub:hello"
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["modality"] == "text"
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["page_number"] == 3
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["corrected"] is False
    assert payload["grounded"] is True
    assert isinstance(payload["retrieval_diagnostics"], dict)
    assert payload["latency_ms"] is not None
    assert payload["latency_ms"] >= 0.0
    assert engine.last_tenant_id == "public"
    assert engine.last_retrieval_mode is None


def test_query_endpoint_passes_retrieval_mode(tmp_path):
    settings = _build_settings(tmp_path)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post(
        "/query",
        json={"question": "hello", "retrieval_mode": "dense_only"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["corrected"] is False
    assert payload["grounded"] is True
    assert engine.last_retrieval_mode == "dense_only"


def test_query_stream_endpoint_emits_sse_events(tmp_path):
    settings = _build_settings(tmp_path)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post("/query-stream", json={"question": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: meta" in body
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body
    assert "stub:hello" in body


def test_query_multimodal_endpoint_with_image(tmp_path):
    settings = _build_settings(tmp_path)
    engine = StubEngine(settings)
    client = _build_client(engine)

    files = {
        "image": ("query.png", b"fake-image-bytes", "image/png"),
    }
    data = {
        "question": "find similar chart",
        "top_k": "3",
    }
    response = client.post("/query-multimodal", data=data, files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "stub:find similar chart"
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["corrected"] is False
    assert payload["grounded"] is True
    assert isinstance(payload["retrieval_diagnostics"], dict)
    assert payload["latency_ms"] is not None
    assert payload["latency_ms"] >= 0.0
    assert engine.last_query_image_path is not None
    assert engine.last_tenant_id == "public"


def test_query_multimodal_passes_retrieval_mode(tmp_path):
    settings = _build_settings(tmp_path)
    engine = StubEngine(settings)
    client = _build_client(engine)

    data = {
        "question": "find similar chart",
        "retrieval_mode": "hybrid_rerank",
    }
    response = client.post("/query-multimodal", data=data)
    assert response.status_code == 200
    assert engine.last_retrieval_mode == "hybrid_rerank"


def test_query_uses_tenant_header_when_auth_disabled(tmp_path):
    settings = _build_settings(tmp_path, default_tenant="public", auth_enabled=False)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post(
        "/query",
        headers={"X-Tenant-ID": "Team_A"},
        json={"question": "hello"},
    )
    assert response.status_code == 200
    assert engine.last_tenant_id == "team_a"


def test_query_requires_api_key_when_auth_enabled(tmp_path):
    settings = _build_settings(
        tmp_path,
        auth_enabled=True,
        auth_tenant_api_keys="tenant-a:key-a",
    )
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 401


def test_query_rejects_tenant_mismatch_when_auth_enabled(tmp_path):
    settings = _build_settings(
        tmp_path,
        auth_enabled=True,
        auth_tenant_api_keys="tenant-a:key-a",
    )
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post(
        "/query",
        headers={"X-API-Key": "key-a", "X-Tenant-ID": "tenant-b"},
        json={"question": "hello"},
    )
    assert response.status_code == 403


def test_query_accepts_valid_api_key_when_auth_enabled(tmp_path):
    settings = _build_settings(
        tmp_path,
        auth_enabled=True,
        auth_tenant_api_keys="tenant-a:key-a",
    )
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post(
        "/query",
        headers={"X-API-Key": "key-a", "X-Tenant-ID": "tenant-a"},
        json={"question": "hello"},
    )
    assert response.status_code == 200
    assert engine.last_tenant_id == "tenant-a"


def test_ingest_jobs_lifecycle(tmp_path):
    settings = _build_settings(tmp_path, ingestion_jobs_enabled=True)
    engine = StubEngine(settings)
    client = _build_client(engine)

    create_response = client.post(
        "/ingest-jobs",
        json={"paths": ["./data"], "collection": "demo"},
    )
    assert create_response.status_code == 202
    created = create_response.json()
    job_id = created["job_id"]
    assert created["status"] in {"queued", "running", "completed"}
    assert created["tenant_id"] == "public"

    final_payload = created
    for _ in range(40):
        status_response = client.get(f"/ingest-jobs/{job_id}")
        assert status_response.status_code == 200
        final_payload = status_response.json()
        if final_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert final_payload["status"] == "completed"
    assert final_payload["result"] is not None
    assert final_payload["result"]["files"] == 1

    list_response = client.get("/ingest-jobs")
    assert list_response.status_code == 200
    jobs = list_response.json()
    assert any(job["job_id"] == job_id for job in jobs)


def test_ingest_jobs_returns_404_for_unknown_id(tmp_path):
    settings = _build_settings(tmp_path, ingestion_jobs_enabled=True)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.get("/ingest-jobs/missing-job")
    assert response.status_code == 404


def test_ingest_jobs_can_be_disabled(tmp_path):
    settings = _build_settings(tmp_path, ingestion_jobs_enabled=False)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post("/ingest-jobs", json={"paths": ["./data"]})
    assert response.status_code == 503


def test_ingest_jobs_requires_non_empty_paths(tmp_path):
    settings = _build_settings(tmp_path, ingestion_jobs_enabled=True)
    engine = StubEngine(settings)
    client = _build_client(engine)

    response = client.post("/ingest-jobs", json={"paths": []})
    assert response.status_code == 422
