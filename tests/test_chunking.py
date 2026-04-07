from multimodal_rag.ingestion.chunking import split_text


def test_split_text_produces_chunks():
    text = " ".join(["token"] * 1200)
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
