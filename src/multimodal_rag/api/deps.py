from __future__ import annotations

from multimodal_rag.engine import MultimodalRAG

_engine: MultimodalRAG | None = None


def get_engine() -> MultimodalRAG:
    global _engine
    if _engine is None:
        _engine = MultimodalRAG.from_settings()
    return _engine
