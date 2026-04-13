from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

SECTION_NUMBER_RE = re.compile(r"^(\d+(\.\d+){0,3}|[IVXLCM]+)[\)\.\-:\s]+[A-Za-z]")
BULLET_RE = re.compile(r"^(\s*[-*•]\s+|\s*\d+[\.\)]\s+)")
TABLE_HINT_RE = re.compile(
    r"(table|appendix|annex|schedule|kpi|metric|revenue|cost|balance|statement|ledger)",
    flags=re.IGNORECASE,
)
PROCEDURAL_HINT_RE = re.compile(
    r"(step|procedure|runbook|playbook|workflow|checklist|how to|instructions?)",
    flags=re.IGNORECASE,
)


def split_text(content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = content.strip()
    if not clean:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [piece.strip() for piece in splitter.split_text(clean) if piece.strip()]


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if len(text) < 3 or len(text) > 120:
        return False
    if text.endswith((".", "?", "!")):
        return False

    words = [w for w in text.split() if w]
    if not words or len(words) > 14:
        return False

    if SECTION_NUMBER_RE.match(text):
        return True

    alpha_chars = [ch for ch in text if ch.isalpha()]
    if alpha_chars:
        uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if uppercase_ratio >= 0.75:
            return True

    titled_words = 0
    for word in words:
        stripped = word.strip("()[]{}:;,.-_")
        if stripped and stripped[0].isupper():
            titled_words += 1
    if len(words) <= 8 and titled_words / len(words) >= 0.8:
        return True

    return False


def split_structured_segments(
    segments: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    adaptive: bool = True,
    min_chunk_size: int = 280,
    table_factor: float = 0.58,
    procedural_factor: float = 0.78,
    narrative_factor: float = 1.0,
    overlap_factor: float = 0.8,
) -> list[dict[str, Any]]:
    """Split already segmented sections while preserving section metadata."""
    output: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        base_metadata = dict(segment.get("metadata", {}) or {})
        section_title = str(base_metadata.get("section_title") or "")
        section_style = classify_section_style(text=text, section_title=section_title)
        segment_chunk_size = chunk_size
        segment_overlap = chunk_overlap

        if adaptive:
            if section_style == "table_like":
                segment_chunk_size = max(min_chunk_size, int(chunk_size * table_factor))
                segment_overlap = max(20, int(chunk_overlap * overlap_factor * 0.6))
            elif section_style == "procedural":
                segment_chunk_size = max(min_chunk_size, int(chunk_size * procedural_factor))
                segment_overlap = max(24, int(chunk_overlap * overlap_factor))
            else:
                segment_chunk_size = max(min_chunk_size, int(chunk_size * narrative_factor))
                segment_overlap = max(24, int(chunk_overlap * overlap_factor))

            if segment_overlap >= segment_chunk_size:
                segment_overlap = max(8, segment_chunk_size // 5)

        parts = split_text(text, chunk_size=segment_chunk_size, chunk_overlap=segment_overlap)
        for part_index, part in enumerate(parts, start=1):
            output.append(
                {
                    "text": part,
                    "metadata": {
                        **base_metadata,
                        "segment_part": part_index,
                        "section_style": section_style,
                        "chunk_size_used": segment_chunk_size,
                        "chunk_overlap_used": segment_overlap,
                    },
                }
            )
    return output


def classify_section_style(text: str, section_title: str = "") -> str:
    title = section_title.strip()
    content = text.strip()
    if not content:
        return "narrative"

    if title and TABLE_HINT_RE.search(title):
        return "table_like"
    if title and PROCEDURAL_HINT_RE.search(title):
        return "procedural"

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    words = [word for word in re.split(r"\s+", content) if word]
    token_count = len(words)

    bullet_lines = sum(1 for line in lines if BULLET_RE.match(line))
    if lines and bullet_lines / len(lines) >= 0.35 and bullet_lines >= 2:
        return "procedural"

    pipe_count = content.count("|")
    if pipe_count >= 8:
        return "table_like"

    numeric_tokens = 0
    for word in words:
        if any(ch.isdigit() for ch in word):
            numeric_tokens += 1
    numeric_ratio = numeric_tokens / max(token_count, 1)
    if numeric_ratio >= 0.28:
        return "table_like"

    return "narrative"
