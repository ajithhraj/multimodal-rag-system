from __future__ import annotations

from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.ingestion.extractors import (
    extract_image_chunks,
    extract_pdf_chunks,
    extract_table_file_chunks,
)
from multimodal_rag.ingestion.vision import VisionCaptioner
from multimodal_rag.models import Chunk

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".csv",
    ".tsv",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TABLE_EXTENSIONS = {".csv", ".tsv"}


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    files: list[Path] = []
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)
    return sorted(files)


def ingest_files(paths: list[Path], settings: Settings) -> list[Chunk]:
    captioner = VisionCaptioner(settings)
    chunks: list[Chunk] = []

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            chunks.extend(extract_pdf_chunks(path, settings))
        elif suffix in IMAGE_EXTENSIONS:
            chunks.extend(extract_image_chunks(path, captioner))
        elif suffix in TABLE_EXTENSIONS:
            chunks.extend(extract_table_file_chunks(path))
    return chunks
