from pathlib import Path

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
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


def test_health():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
    assert engine.last_retrieval_mode == "dense_only"


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
