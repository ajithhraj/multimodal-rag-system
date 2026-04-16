from __future__ import annotations

import json
import logging
from typing import Iterator

from multimodal_rag.config import Settings
from multimodal_rag.models import RetrievalHit

logger = logging.getLogger(__name__)


def _format_context(hits: list[RetrievalHit]) -> str:
    sections: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Context {idx}]",
                    f"modality={hit.chunk.modality.value}",
                    f"source={hit.chunk.source_path}",
                    f"score={hit.score:.4f}",
                    f"content={hit.chunk.content}",
                ]
            )
        )
    return "\n\n".join(sections)


def _system_prompt() -> str:
    return (
        "You are an enterprise multimodal RAG assistant. "
        "Answer strictly from supplied context and admit uncertainty when context is insufficient."
    )


def _user_prompt(question: str, context: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Instructions:\n"
        "- Use concise bullets when helpful.\n"
        "- Reference source file names in your answer.\n"
        "- If data is missing, explicitly state what is missing."
    )


class AnswerSynthesizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm_provider = (settings.llm_provider or "openai").lower()

    def _get_langchain_llm(self):
        if not self.settings.openai_api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.settings.chat_model,
                api_key=self.settings.openai_api_key,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Could not initialize LangChain/OpenAI client: %s", exc)
            return None

    @staticmethod
    def _chunk_to_text(value) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    def _generate_langchain(self, question: str, hits: list[RetrievalHit]) -> str:
        llm = self._get_langchain_llm()
        if not llm:
            return self._fallback_answer(question, hits)
        context = _format_context(hits)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [
                    SystemMessage(content=_system_prompt()),
                    HumanMessage(content=_user_prompt(question, context)),
                ]
            )
            return self._chunk_to_text(response.content).strip()
        except Exception as exc:
            logger.warning("OpenAI generation failed: %s", exc)
            return self._fallback_answer(question, hits)

    def _stream_langchain(self, question: str, hits: list[RetrievalHit]) -> Iterator[str]:
        llm = self._get_langchain_llm()
        if not llm:
            yield self._fallback_answer(question, hits)
            return

        context = _format_context(hits)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            for chunk in llm.stream(
                [
                    SystemMessage(content=_system_prompt()),
                    HumanMessage(content=_user_prompt(question, context)),
                ]
            ):
                text = self._chunk_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text
        except Exception as exc:
            logger.warning("OpenAI streaming failed: %s", exc)
            yield self._fallback_answer(question, hits)

    def _generate_anthropic(self, question: str, hits: list[RetrievalHit]) -> str:
        if not self.settings.anthropic_api_key:
            return self._fallback_answer(question, hits)
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            context = _format_context(hits)
            message = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=1024,
                system=_system_prompt(),
                messages=[{"role": "user", "content": _user_prompt(question, context)}],
            )
            if message.content:
                return str(message.content[0].text).strip()
            return self._fallback_answer(question, hits)
        except Exception as exc:
            logger.warning("Anthropic generation failed: %s", exc)
            return self._fallback_answer(question, hits)

    def _stream_anthropic(self, question: str, hits: list[RetrievalHit]) -> Iterator[str]:
        if not self.settings.anthropic_api_key:
            yield self._fallback_answer(question, hits)
            return
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            context = _format_context(hits)
            with client.messages.stream(
                model=self.settings.anthropic_model,
                max_tokens=1024,
                system=_system_prompt(),
                messages=[{"role": "user", "content": _user_prompt(question, context)}],
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:
            logger.warning("Anthropic streaming failed: %s", exc)
            yield self._fallback_answer(question, hits)

    def _generate_ollama(self, question: str, hits: list[RetrievalHit]) -> str:
        try:
            import requests

            context = _format_context(hits)
            payload = {
                "model": self.settings.ollama_model,
                "prompt": f"{_system_prompt()}\n\n{_user_prompt(question, context)}",
                "stream": False,
            }
            response = requests.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("Ollama generation failed: %s", exc)
            return self._fallback_answer(question, hits)

    def _stream_ollama(self, question: str, hits: list[RetrievalHit]) -> Iterator[str]:
        try:
            import requests

            context = _format_context(hits)
            payload = {
                "model": self.settings.ollama_model,
                "prompt": f"{_system_prompt()}\n\n{_user_prompt(question, context)}",
                "stream": True,
            }
            with requests.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = str(data.get("response", ""))
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except Exception as exc:
            logger.warning("Ollama streaming failed: %s", exc)
            yield self._fallback_answer(question, hits)

    def _generate_llamaindex(self, question: str, hits: list[RetrievalHit]) -> str:
        try:
            from llama_index.core import Document, SummaryIndex

            docs = [Document(text=hit.chunk.content, doc_id=hit.chunk.chunk_id) for hit in hits]
            if not docs:
                return "I could not find relevant context in the index."
            index = SummaryIndex.from_documents(docs)
            query_engine = index.as_query_engine()
            result = query_engine.query(
                "Answer this question only from the retrieved multimodal context:\n"
                f"{question}\n"
            )
            content = str(result).strip()
            return content if content else self._fallback_answer(question, hits)
        except Exception:
            return self._generate_langchain(question, hits)

    @staticmethod
    def _fallback_answer(question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "I could not find relevant context in the indexed sources."
        lines = [f"Question: {question}", "", "Top supporting context:"]
        for hit in hits[:4]:
            preview = hit.chunk.content[:240].replace("\n", " ")
            lines.append(
                f"- [{hit.chunk.modality.value}] {hit.chunk.source_path} (score={hit.score:.3f}): {preview}"
            )
        return "\n".join(lines)

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        provider = self._llm_provider
        if provider == "anthropic":
            return self._generate_anthropic(question, hits)
        if provider == "ollama":
            return self._generate_ollama(question, hits)
        if provider == "llamaindex":
            return self._generate_llamaindex(question, hits)
        return self._generate_langchain(question, hits)

    def stream(self, question: str, hits: list[RetrievalHit]) -> Iterator[str]:
        provider = self._llm_provider
        if provider == "anthropic":
            yield from self._stream_anthropic(question, hits)
            return
        if provider == "ollama":
            yield from self._stream_ollama(question, hits)
            return
        if provider == "llamaindex":
            yield self._generate_llamaindex(question, hits)
            return
        yield from self._stream_langchain(question, hits)
