from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from multimodal_rag.config import Settings
from multimodal_rag.ingestion.chunking import split_text
from multimodal_rag.ingestion.vision import VisionCaptioner, run_ocr
from multimodal_rag.models import Chunk, Modality


def _make_chunk_id(source_path: Path, modality: Modality, ordinal: int) -> str:
    key = f"{source_path.resolve()}::{modality.value}::{ordinal}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned_rows: list[list[str]] = []
    for row in rows:
        cleaned_rows.append([(cell or "").replace("\n", " ").strip() for cell in row])
    if not cleaned_rows:
        return ""
    width = max(len(row) for row in cleaned_rows)
    normalized = [row + [""] * (width - len(row)) for row in cleaned_rows]
    header = normalized[0]
    divider = ["---"] * width
    body = normalized[1:] if len(normalized) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_pdf_chunks(path: Path, settings: Settings, captioner: VisionCaptioner) -> list[Chunk]:
    chunks: list[Chunk] = []
    page_text_blocks: list[str] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            page_text_blocks.append(f"[Page {page_number}]\n{text}")

    text_content = "\n\n".join(page_text_blocks)
    for idx, piece in enumerate(
        split_text(text_content, settings.chunk_size, settings.chunk_overlap), start=1
    ):
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(path, Modality.TEXT, idx),
                source_path=str(path),
                modality=Modality.TEXT,
                content=piece,
                metadata={"kind": "pdf_text"},
            )
        )

    table_index = 0
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                markdown = _rows_to_markdown(table)
                if not markdown.strip():
                    continue
                table_index += 1
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(path, Modality.TABLE, table_index),
                        source_path=str(path),
                        modality=Modality.TABLE,
                        content=markdown,
                        metadata={
                            "kind": "pdf_table",
                            "page_number": page_number,
                            "table_index": table_index,
                        },
                    )
                )

    return chunks


def extract_image_chunks(path: Path, captioner: VisionCaptioner) -> list[Chunk]:
    caption = captioner.caption(path)
    ocr_text = run_ocr(path)
    content = caption
    if ocr_text:
        content = f"{caption}\n\nOCR Text:\n{ocr_text}"
    return [
        Chunk(
            chunk_id=_make_chunk_id(path, Modality.IMAGE, 1),
            source_path=str(path),
            modality=Modality.IMAGE,
            content=content,
            metadata={
                "kind": "image",
                "image_path": str(path),
                "caption": caption,
                "ocr_text": ocr_text,
            },
        )
    ]


def extract_table_file_chunks(path: Path) -> list[Chunk]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = [row for row in reader]

    markdown = _rows_to_markdown(rows)
    if not markdown.strip():
        return []

    return [
        Chunk(
            chunk_id=_make_chunk_id(path, Modality.TABLE, 1),
            source_path=str(path),
            modality=Modality.TABLE,
            content=markdown,
            metadata={"kind": "table_file", "delimiter": delimiter},
        )
    ]
