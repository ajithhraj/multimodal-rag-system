from __future__ import annotations

import csv
import hashlib
import statistics
from pathlib import Path
from typing import Any

import pdfplumber

from multimodal_rag.config import Settings
from multimodal_rag.ingestion.chunking import looks_like_heading, split_structured_segments
from multimodal_rag.ingestion.vision import VisionCaptioner, run_ocr
from multimodal_rag.models import Chunk, Modality

BBox = tuple[float, float, float, float]


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


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max((x1 - x0) * (y1 - y0), 0.0)


def _intersection_area(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _iou(a: BBox, b: BBox) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _overlap_fraction(subject: BBox, container: BBox) -> float:
    subject_area = _bbox_area(subject)
    if subject_area <= 0:
        return 0.0
    return _intersection_area(subject, container) / subject_area


def _word_overlaps_table(word_bbox: BBox, table_bboxes: list[BBox], threshold: float = 0.35) -> bool:
    for table_bbox in table_bboxes:
        if _overlap_fraction(word_bbox, table_bbox) >= threshold:
            return True
    return False


def _dedupe_table_candidates(candidates: list[tuple[BBox, list[list[str | None]]]]) -> list[tuple[BBox, list[list[str | None]]]]:
    deduped: list[tuple[BBox, list[list[str | None]]]] = []
    for bbox, rows in candidates:
        if not rows:
            continue
        duplicate = False
        for existing_bbox, _ in deduped:
            if _iou(bbox, existing_bbox) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            deduped.append((bbox, rows))
    return deduped


def _extract_page_tables(page: Any) -> list[tuple[BBox, list[list[str | None]]]]:
    candidates: list[tuple[BBox, list[list[str | None]]]] = []

    table_settings_variants = [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
        },
    ]

    for table_settings in table_settings_variants:
        try:
            found = page.find_tables(table_settings=table_settings)
        except Exception:
            continue
        for table in found:
            bbox = tuple(float(value) for value in table.bbox)
            rows = table.extract()
            candidates.append((bbox, rows))

    # Fallback for difficult PDFs where table objects were not detected.
    if not candidates:
        try:
            raw_tables = page.extract_tables() or []
        except Exception:
            raw_tables = []
        for rows in raw_tables:
            if not rows:
                continue
            # Unknown bbox in fallback mode: use a non-overlapping sentinel area.
            candidates.append(((0.0, 0.0, 0.0, 0.0), rows))

    return _dedupe_table_candidates(candidates)


def _group_words_into_lines(words: list[dict[str, Any]], y_tolerance: float = 3.0) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        bottom = float(word["bottom"])

        best_idx: int | None = None
        best_delta = float("inf")
        for idx, line in enumerate(lines):
            delta = abs(top - float(line["top"]))
            if delta <= y_tolerance and delta < best_delta:
                best_delta = delta
                best_idx = idx

        if best_idx is None:
            lines.append({"top": top, "bottom": bottom, "words": [word]})
        else:
            lines[best_idx]["words"].append(word)
            lines[best_idx]["bottom"] = max(float(lines[best_idx]["bottom"]), bottom)

    normalized: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: float(item["top"])):
        words_sorted = sorted(line["words"], key=lambda item: float(item["x0"]))
        text = " ".join(str(word["text"]).strip() for word in words_sorted if str(word["text"]).strip())
        if not text:
            continue
        normalized.append(
            {
                "top": float(line["top"]),
                "bottom": float(line["bottom"]),
                "text": text.strip(),
            }
        )
    return normalized


def _page_structured_segments(page: Any, page_number: int, table_bboxes: list[BBox]) -> list[dict[str, Any]]:
    try:
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            use_text_flow=True,
            keep_blank_chars=False,
        )
    except Exception:
        words = []

    filtered_words: list[dict[str, Any]] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        bbox: BBox = (
            float(word.get("x0", 0.0)),
            float(word.get("top", 0.0)),
            float(word.get("x1", 0.0)),
            float(word.get("bottom", 0.0)),
        )
        if table_bboxes and _word_overlaps_table(bbox, table_bboxes):
            continue
        filtered_words.append(word)

    lines = _group_words_into_lines(filtered_words)
    if not lines:
        return []

    line_heights = [float(line["bottom"]) - float(line["top"]) for line in lines]
    median_height = statistics.median(line_heights) if line_heights else 10.0
    paragraph_gap_threshold = max(8.0, median_height * 1.6)

    segments: list[dict[str, Any]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    current_start_top: float | None = None
    previous_bottom: float | None = None

    def flush_current() -> None:
        nonlocal current_lines, current_start_top
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            current_lines = []
            current_start_top = None
            return
        if current_heading:
            text = f"[Page {page_number}] Section: {current_heading}\n{body}"
        else:
            text = f"[Page {page_number}]\n{body}"
        segments.append(
            {
                "text": text,
                "metadata": {
                    "page_number": page_number,
                    "section_title": current_heading,
                    "bbox_top": current_start_top,
                },
            }
        )
        current_lines = []
        current_start_top = None

    for line in lines:
        line_text = str(line["text"]).strip()
        line_top = float(line["top"])
        line_bottom = float(line["bottom"])
        if not line_text:
            continue

        if looks_like_heading(line_text):
            flush_current()
            current_heading = line_text
            previous_bottom = line_bottom
            continue

        if (
            previous_bottom is not None
            and current_lines
            and (line_top - previous_bottom) > paragraph_gap_threshold
        ):
            flush_current()

        if current_start_top is None:
            current_start_top = line_top
        current_lines.append(line_text)
        previous_bottom = line_bottom

    flush_current()
    return segments


def extract_pdf_chunks(path: Path, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    text_index = 0
    table_index = 0

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            table_candidates = _extract_page_tables(page)
            page_table_bboxes: list[BBox] = []
            table_chunks: list[Chunk] = []

            for bbox, rows in table_candidates:
                markdown = _rows_to_markdown(rows)
                if not markdown.strip():
                    continue
                table_index += 1
                if _bbox_area(bbox) > 0:
                    page_table_bboxes.append(bbox)
                table_chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(path, Modality.TABLE, table_index),
                        source_path=str(path),
                        modality=Modality.TABLE,
                        content=markdown,
                        metadata={
                            "kind": "pdf_table",
                            "page_number": page_number,
                            "table_index": table_index,
                            "bbox": list(bbox) if _bbox_area(bbox) > 0 else None,
                        },
                    )
                )

            chunks.extend(table_chunks)

            segments = _page_structured_segments(page, page_number=page_number, table_bboxes=page_table_bboxes)
            split_segments = split_structured_segments(
                segments,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                adaptive=settings.adaptive_chunking_enabled,
                min_chunk_size=settings.adaptive_chunking_min_size,
                table_factor=settings.adaptive_chunking_table_factor,
                procedural_factor=settings.adaptive_chunking_procedural_factor,
                narrative_factor=settings.adaptive_chunking_narrative_factor,
                overlap_factor=settings.adaptive_chunking_overlap_factor,
            )
            for segment in split_segments:
                text_index += 1
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(path, Modality.TEXT, text_index),
                        source_path=str(path),
                        modality=Modality.TEXT,
                        content=str(segment["text"]),
                        metadata={
                            "kind": "pdf_text",
                            **dict(segment.get("metadata") or {}),
                        },
                    )
                )

    return chunks


def extract_image_chunks(path: Path, captioner: VisionCaptioner) -> list[Chunk]:
    caption = captioner.caption(path)
    ocr_text = run_ocr(path) if captioner.allow_local_ocr() else ""
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
