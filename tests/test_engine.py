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

    def delete_by_source(self, collection: str, modality: str, source_paths):  # type: ignore[override]
        key = self._key(collection, modality)
        existing = self._items.get(key, [])
        if not existing:
            return 0

        source_set = set(source_paths)
        kept = [hit for hit in existing if hit.chunk.source_path not in source_set]
        removed = len(existing) - len(kept)
        self._items[key] = kept
        return removed


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

    scoped_collection = engine._scoped_collection(None, None)
    store.upsert(scoped_collection, "text", [[1, 0]], [c1])
    store.upsert(scoped_collection, "table", [[1, 0]], [c2])
    store.upsert(scoped_collection, "image", [[1, 0]], [c3])
    engine.lexical_index.upsert(scoped_collection, [c1, c2, c3])

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
    scoped_collection = engine._scoped_collection(None, None)
    engine.lexical_index.upsert(scoped_collection, [chunk])

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


def test_engine_scopes_queries_by_tenant(tmp_path):
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

    tenant_a_chunk = Chunk("tenant-a-1", "a.pdf", Modality.TEXT, "Alpha tenant context")
    tenant_b_chunk = Chunk("tenant-b-1", "b.pdf", Modality.TEXT, "Beta tenant context")

    collection_a = engine._scoped_collection("shared", "tenant-a")
    collection_b = engine._scoped_collection("shared", "tenant-b")
    store.upsert(collection_a, "text", [[1, 0]], [tenant_a_chunk])
    store.upsert(collection_b, "text", [[1, 0]], [tenant_b_chunk])
    engine.lexical_index.upsert(collection_a, [tenant_a_chunk])
    engine.lexical_index.upsert(collection_b, [tenant_b_chunk])

    result_a = engine.query("context", collection="shared", tenant_id="tenant-a")
    result_b = engine.query("context", collection="shared", tenant_id="tenant-b")

    assert len(result_a.hits) == 1
    assert len(result_b.hits) == 1
    assert result_a.hits[0].chunk.chunk_id == "tenant-a-1"
    assert result_b.hits[0].chunk.chunk_id == "tenant-b-1"


def test_engine_ingest_refreshes_source_without_duplicates(tmp_path, monkeypatch):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=2,
        max_context_chunks=5,
        retrieval_enable_reranker=False,
        ingestion_skip_unchanged_files=False,
    )
    store = FakeStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    target_file = Path(tmp_path) / "doc.pdf"
    target_file.write_bytes(b"fake")

    first_pass = [
        Chunk("doc-text-1", str(target_file), Modality.TEXT, "Old content"),
        Chunk("doc-table-1", str(target_file), Modality.TABLE, "Old table"),
    ]
    second_pass = [
        Chunk("doc-text-1", str(target_file), Modality.TEXT, "Updated content"),
    ]
    ingestion_passes = [first_pass, second_pass]

    monkeypatch.setattr("multimodal_rag.engine.discover_files", lambda _: [target_file])

    def fake_ingest_files(paths, _settings):
        _ = paths
        return ingestion_passes.pop(0)

    monkeypatch.setattr("multimodal_rag.engine.ingest_files", fake_ingest_files)

    engine.ingest_paths([target_file])
    engine.ingest_paths([target_file])

    result = engine.query("content")
    assert len(result.hits) == 1
    assert result.hits[0].chunk.content == "Updated content"


def test_engine_ingest_skips_unchanged_files_by_default(tmp_path, monkeypatch):
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

    target_file = Path(tmp_path) / "doc.pdf"
    target_file.write_bytes(b"v1")

    calls = {"count": 0}

    monkeypatch.setattr("multimodal_rag.engine.discover_files", lambda _: [target_file])

    def fake_ingest_files(paths, _settings):
        calls["count"] += 1
        return [Chunk("doc-text-1", str(paths[0]), Modality.TEXT, f"content-{calls['count']}")]

    monkeypatch.setattr("multimodal_rag.engine.ingest_files", fake_ingest_files)

    engine.ingest_paths([target_file])
    engine.ingest_paths([target_file])

    target_file.write_bytes(b"v2-new-content")
    engine.ingest_paths([target_file])

    assert calls["count"] == 2


def test_engine_query_applies_per_source_diversity_cap(tmp_path):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=5,
        max_context_chunks=5,
        retrieval_enable_reranker=False,
        retrieval_max_chunks_per_source=1,
        retrieval_duplicate_similarity_threshold=0.95,
    )
    store = FakeStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    source_a_hit_1 = Chunk("a-1", "docs/a.pdf", Modality.TEXT, "A first section")
    source_a_hit_2 = Chunk("a-2", "docs/a.pdf", Modality.TEXT, "A second section")
    source_b_hit = Chunk("b-1", "docs/b.pdf", Modality.TEXT, "B section")
    scoped_collection = engine._scoped_collection(None, None)
    store.upsert(scoped_collection, "text", [[1, 0], [0.95, 0.05], [0.9, 0.1]], [source_a_hit_1, source_a_hit_2, source_b_hit])

    result = engine.query("section", retrieval_mode="dense_only")
    assert len(result.hits) == 2
    assert {hit.chunk.chunk_id for hit in result.hits} == {"a-1", "b-1"}


def test_engine_query_deduplicates_near_identical_context(tmp_path):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=5,
        max_context_chunks=5,
        retrieval_enable_reranker=False,
        retrieval_max_chunks_per_source=3,
        retrieval_duplicate_similarity_threshold=0.8,
    )
    store = FakeStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    duplicate_1 = Chunk("dup-1", "docs/a.pdf", Modality.TEXT, "Revenue grew 25 percent year over year")
    duplicate_2 = Chunk("dup-2", "docs/b.pdf", Modality.TEXT, "Revenue grew 25 percent year over year")
    distinct = Chunk("distinct-1", "docs/c.pdf", Modality.TEXT, "Operating margin was 12 percent")
    scoped_collection = engine._scoped_collection(None, None)
    store.upsert(scoped_collection, "text", [[1, 0], [0.99, 0.01], [0.8, 0.2]], [duplicate_1, duplicate_2, distinct])

    result = engine.query("revenue and margin", retrieval_mode="dense_only")
    assert len(result.hits) == 2
    assert {hit.chunk.chunk_id for hit in result.hits} == {"dup-1", "distinct-1"}


def test_engine_query_weighted_hybrid_prioritizes_lexical_signal(tmp_path):
    settings = Settings(
        storage_dir=Path(tmp_path),
        retrieval_top_k_per_modality=2,
        retrieval_top_k_lexical=5,
        retrieval_enable_reranker=False,
        retrieval_enable_result_diversity=False,
        retrieval_rrf_weight_text=0.1,
        retrieval_rrf_weight_table=0.0,
        retrieval_rrf_weight_image=0.0,
        retrieval_rrf_weight_lexical=3.0,
    )
    store = FakeStore()
    engine = MultimodalRAG(settings=settings, store=store)
    engine.text_embedder = DummyEmbedder()  # type: ignore[assignment]
    engine.vision_embedder = DummyVisionEmbedder()  # type: ignore[assignment]
    engine.synthesizer = DummySynthesizer()  # type: ignore[assignment]

    scoped_collection = engine._scoped_collection(None, None)
    dense_chunk = Chunk("dense-1", "docs/dense.pdf", Modality.TEXT, "boilerplate")
    lexical_chunk = Chunk("lex-1", "docs/lexical.pdf", Modality.TEXT, "critical_signal token")
    store.upsert(scoped_collection, "text", [[1, 0]], [dense_chunk])
    engine.lexical_index.upsert(scoped_collection, [lexical_chunk])

    result = engine.query("critical_signal", retrieval_mode="hybrid")
    assert len(result.hits) >= 2
    assert result.hits[0].chunk.chunk_id == "lex-1"
