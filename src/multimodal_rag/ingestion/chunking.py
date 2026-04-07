from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
