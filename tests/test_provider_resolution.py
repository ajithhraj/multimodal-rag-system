from __future__ import annotations

from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.embedding import providers


class FakeOpenAIEmbeddings:
    last_documents: list[str] | None = None
    last_query: str | None = None

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        type(self).last_documents = texts
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        type(self).last_query = text
        return [42.0, float(len(text))]


class FakeCaptioner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def caption(self, image_path: Path) -> str:
        return f"caption for {image_path.name}"


def test_settings_auto_promotes_to_openai_when_key_present():
    settings = Settings(storage_dir=Path("."), openai_api_key="sk-test", llm_provider="auto")
    assert settings.resolved_llm_provider() == "openai"


def test_settings_local_is_promoted_to_openai_when_key_present():
    settings = Settings(storage_dir=Path("."), openai_api_key="sk-test", llm_provider="local")
    assert settings.resolved_llm_provider() == "openai"


def test_settings_stays_local_without_key():
    settings = Settings(storage_dir=Path("."), openai_api_key=None, llm_provider="auto")
    assert settings.resolved_llm_provider() == "local"


def test_settings_strict_api_only_mode_for_openai_provider():
    settings = Settings(storage_dir=Path("."), openai_api_key="sk-test", llm_provider="openai")
    assert settings.strict_api_only_mode() is True


def test_text_embedder_uses_openai_embeddings_when_key_present(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    settings = Settings(storage_dir=tmp_path, openai_api_key="sk-test")

    embedder = providers.TextEmbedder(settings)
    documents = embedder.embed_documents(["alpha", "beta"])
    query = embedder.embed_query("hello")

    assert embedder.uses_openai is True
    assert FakeOpenAIEmbeddings.last_documents == ["alpha", "beta"]
    assert FakeOpenAIEmbeddings.last_query == "hello"
    assert documents == [[1.0, 5.0], [2.0, 4.0]]
    assert query == [42.0, 5.0]


def test_text_embedder_raises_without_openai_client_in_strict_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "OpenAIEmbeddings", None)
    settings = Settings(storage_dir=tmp_path, openai_api_key="sk-test", llm_provider="openai")

    try:
        providers.TextEmbedder(settings)
    except RuntimeError as exc:
        assert "langchain_openai" in str(exc)
    else:
        raise AssertionError("Expected TextEmbedder to fail in strict API-only mode")


def test_vision_embedder_uses_openai_text_path_for_query_image(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(providers, "VisionCaptioner", FakeCaptioner)
    monkeypatch.setattr(providers, "run_ocr", lambda image_path: "ocr data")

    settings = Settings(storage_dir=tmp_path, openai_api_key="sk-test")
    text_embedder = providers.TextEmbedder(settings)
    vision_embedder = providers.VisionEmbedder(settings, text_embedder)

    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"fake")

    vector = vision_embedder.embed_query("find a chart", image_path=image_path)

    assert vector[0] == 42.0
    assert FakeOpenAIEmbeddings.last_query is not None
    assert "find a chart" in FakeOpenAIEmbeddings.last_query
    assert "caption for query.png" in FakeOpenAIEmbeddings.last_query
    assert "ocr data" not in FakeOpenAIEmbeddings.last_query
