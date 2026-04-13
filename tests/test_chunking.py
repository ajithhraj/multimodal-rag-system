from multimodal_rag.ingestion.chunking import (
    classify_section_style,
    looks_like_heading,
    split_structured_segments,
    split_text,
)


def test_split_text_produces_chunks():
    text = " ".join(["token"] * 1200)
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_looks_like_heading_variants():
    assert looks_like_heading("1. Executive Summary")
    assert looks_like_heading("RISK FACTORS")
    assert looks_like_heading("Key Metrics")
    assert not looks_like_heading("This is a long sentence that should end with punctuation.")
    assert not looks_like_heading("ok")


def test_split_structured_segments_keeps_metadata():
    segments = [
        {
            "text": "Section text " * 100,
            "metadata": {"page_number": 2, "section_title": "Executive Summary"},
        }
    ]
    split = split_structured_segments(segments, chunk_size=120, chunk_overlap=20)
    assert len(split) > 1
    assert all(item["metadata"]["page_number"] == 2 for item in split)
    assert all(item["metadata"]["section_title"] == "Executive Summary" for item in split)
    assert split[0]["metadata"]["segment_part"] == 1


def test_classify_section_style():
    assert classify_section_style("| A | B |\n|---|---|", section_title="Table 1 Revenue") == "table_like"
    assert classify_section_style("1. Step one\n2. Step two", section_title="Procedure") == "procedural"
    assert classify_section_style("This section explains the system design and rationale.") == "narrative"
