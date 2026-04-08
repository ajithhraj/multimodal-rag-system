from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_path: str
    modality: Modality
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "modality": self.modality.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            source_path=str(payload["source_path"]),
            modality=Modality(str(payload["modality"])),
            content=str(payload["content"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    backend: str


@dataclass(slots=True)
class Citation:
    chunk_id: str
    source_path: str
    modality: Modality
    page_number: int | None = None
    excerpt: str | None = None


@dataclass(slots=True)
class QueryAnswer:
    answer: str
    hits: list[RetrievalHit]
    citations: list[Citation] = field(default_factory=list)
    retrieval_mode: str | None = None
    corrected: bool = False
    retrieval_diagnostics: dict[str, Any] = field(default_factory=dict)
