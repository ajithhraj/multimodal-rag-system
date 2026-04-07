from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
import uvicorn

from multimodal_rag.config import get_settings
from multimodal_rag.engine import MultimodalRAG

app = typer.Typer(help="Multimodal RAG CLI")


def _build_engine(backend: Literal["faiss", "qdrant"] | None = None) -> MultimodalRAG:
    settings = get_settings()
    if backend:
        settings = settings.model_copy(update={"vector_backend": backend})
    return MultimodalRAG(settings=settings)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory path to ingest."),
    collection: str | None = typer.Option(None, help="Collection name."),
    backend: Literal["faiss", "qdrant"] | None = typer.Option(None, help="Vector backend override."),
) -> None:
    engine = _build_engine(backend)
    stats = engine.ingest_paths([path], collection=collection)
    typer.echo(
        f"Ingested files={stats['files']} chunks={stats['chunks']} "
        f"text={stats['text']} table={stats['table']} image={stats['image']}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question for retrieval + generation."),
    image: Path | None = typer.Option(None, help="Optional query image path for multimodal retrieval."),
    collection: str | None = typer.Option(None, help="Collection name."),
    top_k: int | None = typer.Option(None, min=1, max=50, help="Top-k per modality."),
    backend: Literal["faiss", "qdrant"] | None = typer.Option(None, help="Vector backend override."),
) -> None:
    engine = _build_engine(backend)
    if image is not None and not image.exists():
        raise typer.BadParameter(f"Image path not found: {image}")
    result = engine.query(
        question=question,
        collection=collection,
        top_k=top_k,
        query_image_path=image,
    )
    typer.echo(result.answer)
    if result.citations:
        typer.echo("\nCitations:")
        for citation in result.citations:
            page = f", page={citation.page_number}" if citation.page_number is not None else ""
            typer.echo(
                f"- [{citation.modality.value}] {citation.source_path}{page}"
            )
    if result.hits:
        typer.echo("\nSources:")
        for hit in result.hits:
            typer.echo(
                f"- [{hit.chunk.modality.value}] {hit.chunk.source_path} "
                f"(score={hit.score:.4f})"
            )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    uvicorn.run("multimodal_rag.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
