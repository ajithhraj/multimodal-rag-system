import pytest

from multimodal_rag.models import Chunk, Modality, RetrievalHit
from multimodal_rag.retrieval.hybrid import reciprocal_rank_fusion


def _hit(chunk_id: str, source: str) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_path=source,
            modality=Modality.TEXT,
            content=f"content-{chunk_id}",
        ),
        score=1.0,
        backend="stub",
    )


def test_reciprocal_rank_fusion_respects_weights():
    a = _hit("a", "a.pdf")
    b = _hit("b", "b.pdf")
    result = reciprocal_rank_fusion(
        [[a], [b]],
        k=10,
        weights=[2.0, 1.0],
    )
    assert [hit.chunk.chunk_id for hit in result] == ["a", "b"]


def test_reciprocal_rank_fusion_ignores_zero_weight_lists():
    a = _hit("a", "a.pdf")
    b = _hit("b", "b.pdf")
    result = reciprocal_rank_fusion(
        [[a], [b]],
        k=10,
        weights=[0.0, 1.0],
    )
    assert [hit.chunk.chunk_id for hit in result] == ["b"]


def test_reciprocal_rank_fusion_validates_weights_length():
    a = _hit("a", "a.pdf")
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[a]], k=10, weights=[1.0, 2.0])
