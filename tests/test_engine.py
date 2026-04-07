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


class SpyVisionEmbedder(DummyVisionEmbedder):
    def __init__(self):
        self.image_calls = []

    def embed_images(self, image_paths, fallback_texts):
        self.image_calls.append((list(image_paths), list(fallback_texts)))
        return super().embed_images(image_paths, fallback_texts)


class DummySynthesizer:
    def generate(self, question, hits):
        return f"answer:{question}:{len(hits)}"


def test_engine_query_returns_fused_hits(tmp_path):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=2,
        max_context_chunks=5,
        retrieval_enable_reranker=False,
    )
    store = FakeStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    c1 = Chunk("1", "a.pdf", Modality.TEXT, "text content", metadata={"page_number": 2})
    c2 = Chunk("2", "a.pdf", Modality.TABLE, "table content")
    c3 = Chunk("3", "img.png", Modality.IMAGE, "image content")

    store.upsert("default", "text", [[1, 0]], [c1])
    store.upsert("default", "table", [[1, 0]], [c2])
    store.upsert("default", "image", [[1, 0]], [c3])
    engine.lexical_index.upsert("default", [c1, c2, c3])

    result = engine.query("what is inside")
    assert "answer:what is inside" in result.answer
    assert len(result.hits) == 3
    assert {hit.chunk.modality for hit in result.hits} == {Modality.TEXT, Modality.TABLE, Modality.IMAGE}
    assert len(result.citations) == 3
    citation_by_id = {citation.chunk_id: citation for citation in result.citations}
    assert citation_by_id["1"].page_number == 2
    assert citation_by_id["2"].page_number is None


class EmptyQueryStore(FakeStore):
    def query(self, collection: str, modality: str, query_vector, top_k):  # type: ignore[override]
        return []


def test_engine_query_uses_lexical_index_when_dense_empty(tmp_path):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=2,
        retrieval_top_k_lexical=5,
        max_context_chunks=5,
        retrieval_enable_reranker=False,
    )
    store = EmptyQueryStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    chunk = Chunk("lex-1", "contract.pdf", Modality.TEXT, "Contract number ZX-42")
    engine.lexical_index.upsert("default", [chunk])

    result = engine.query("what is the contract number")
    assert len(result.hits) == 1
    assert result.hits[0].chunk.chunk_id == "lex-1"


def test_engine_query_uses_query_image_vector_when_image_provided(tmp_path):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=2,
        max_context_chunks=5,
        retrieval_enable_reranker=False,
    )
    store = EmptyQueryStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    vision = SpyVisionEmbedder()
    engine.vision_embedder = vision  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    query_image = Path(tmp_path) / "query.png"
    query_image.write_bytes(b"fake")

    result = engine.query("find similar", query_image_path=query_image)
    assert "answer:find similar" in result.answer
    assert len(vision.image_calls) == 1
    assert vision.image_calls[0][0][0] == query_image
