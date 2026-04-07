from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.engine import MultimodalRAG
from multimodal_rag.models import Chunk, Modality, RetrievalHit
from multimodal_rag.storage.base import VectorStore


class FakeStore(VectorStore):
    def __init__(self):
        self._items: dict[str, list[RetrievalHit]] = {}

    def _key(self, collection: str, modality: str) -> str:
        return f"{collection}:{modality}"

    def upsert(self, collection: str, modality: str, vectors, chunks):  # type: ignore[override]
        key = self._key(collection, modality)
        existing = self._items.get(key, [])
        for chunk in chunks:
            existing.append(RetrievalHit(chunk=chunk, score=1.0, backend="fake"))
        self._items[key] = existing
        return len(chunks)

    def query(self, collection: str, modality: str, query_vector, top_k):  # type: ignore[override]
        key = self._key(collection, modality)
        return self._items.get(key, [])[:top_k]


class DummyEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class DummyVisionEmbedder(DummyEmbedder):
    def embed_images(self, image_paths, fallback_texts):
        return [[0.5, 0.5] for _ in image_paths]


class DummySynthesizer:
    def generate(self, question, hits):
        return f"answer:{question}:{len(hits)}"


def test_engine_query_returns_fused_hits():
    settings = Settings(
        storage_dir=Path("."),
        retrieval_top_k_per_modality=2,
        max_context_chunks=5,
    )
    store = FakeStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    c1 = Chunk("1", "a.pdf", Modality.TEXT, "text content")
    c2 = Chunk("2", "a.pdf", Modality.TABLE, "table content")
    c3 = Chunk("3", "img.png", Modality.IMAGE, "image content")

    store.upsert("default", "text", [[1, 0]], [c1])
    store.upsert("default", "table", [[1, 0]], [c2])
    store.upsert("default", "image", [[1, 0]], [c3])

    result = engine.query("what is inside")
    assert "answer:what is inside" in result.answer
    assert len(result.hits) == 3
    assert {hit.chunk.modality for hit in result.hits} == {Modality.TEXT, Modality.TABLE, Modality.IMAGE}
