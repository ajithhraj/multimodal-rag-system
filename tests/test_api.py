from pathlib import Path

from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app
from multimodal_rag.api.deps import get_engine
from multimodal_rag.models import Chunk, Modality, QueryAnswer, RetrievalHit


class StubEngine:
    def __init__(self):
        self.settings = type("StubSettings", (), {"storage_dir": Path(".")})()

    def ingest_paths(self, paths, collection=None):
        return {"files": len(paths), "chunks": 1, "text": 1, "table": 0, "image": 0}

    def query(self, question, collection=None, top_k=None):
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
        return QueryAnswer(answer=f"stub:{question}", hits=[hit])


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
