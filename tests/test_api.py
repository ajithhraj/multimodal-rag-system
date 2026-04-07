from pathlib import Path

from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app
from multimodal_rag.api.deps import get_engine
from multimodal_rag.models import Citation, Chunk, Modality, QueryAnswer, RetrievalHit


class StubEngine:
    def __init__(self):
        self.settings = type("StubSettings", (), {"storage_dir": Path(".")})()
        self.last_query_image_path = None

    def ingest_paths(self, paths, collection=None):
        return {"files": len(paths), "chunks": 1, "text": 1, "table": 0, "image": 0}

    def query(self, question, collection=None, top_k=None, query_image_path=None):
        self.last_query_image_path = query_image_path
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
        return QueryAnswer(answer=f"stub:{question}", hits=[hit], citations=[citation])


def test_health():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint():
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: StubEngine()
    client = TestClient(app)

    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "stub:hello"
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["modality"] == "text"
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["page_number"] == 3


def test_query_multimodal_endpoint_with_image():
    app = create_app()
    engine = StubEngine()
    app.dependency_overrides[get_engine] = lambda: engine
    client = TestClient(app)

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
    assert engine.last_query_image_path is not None
