from pathlib import Path

from multimodal_rag.models import Chunk, Modality
from multimodal_rag.storage.faiss_store import FaissStore


def test_faiss_store_upsert_replaces_existing_chunk_id(tmp_path):
    store = FaissStore(Path(tmp_path) / "faiss")
    collection = "tenant-public__default"

    old_chunk = Chunk(
        chunk_id="dup-1",
        source_path="docs/report.pdf",
        modality=Modality.TEXT,
        content="old",
    )
    new_chunk = Chunk(
        chunk_id="dup-1",
        source_path="docs/report.pdf",
        modality=Modality.TEXT,
        content="new",
    )

    store.upsert(collection, Modality.TEXT.value, [[1.0, 0.0]], [old_chunk])
    store.upsert(collection, Modality.TEXT.value, [[0.0, 1.0]], [new_chunk])

    hits = store.query(collection, Modality.TEXT.value, [0.0, 1.0], top_k=10)
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "dup-1"
    assert hits[0].chunk.content == "new"


def test_faiss_store_delete_by_source(tmp_path):
    store = FaissStore(Path(tmp_path) / "faiss")
    collection = "tenant-public__default"

    chunk_a = Chunk(
        chunk_id="a-1",
        source_path="docs/a.pdf",
        modality=Modality.TEXT,
        content="A content",
    )
    chunk_b = Chunk(
        chunk_id="b-1",
        source_path="docs/b.pdf",
        modality=Modality.TEXT,
        content="B content",
    )

    store.upsert(collection, Modality.TEXT.value, [[1.0, 0.0], [0.0, 1.0]], [chunk_a, chunk_b])
    removed = store.delete_by_source(collection, Modality.TEXT.value, ["docs/a.pdf"])

    assert removed == 1
    hits = store.query(collection, Modality.TEXT.value, [0.0, 1.0], top_k=10)
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "b-1"
