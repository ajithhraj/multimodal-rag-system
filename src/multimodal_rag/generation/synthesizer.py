from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from multimodal_rag.config import Settings
from multimodal_rag.models import RetrievalHit

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None
        if settings.openai_api_key:
            self._llm = ChatOpenAI(
                model=settings.chat_model,
                api_key=settings.openai_api_key,
                temperature=0.0,
            )

    @staticmethod
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

    def _generate_langchain(self, question: str, hits: list[RetrievalHit]) -> str:
        if not self._llm:
            return self._fallback_answer(question, hits)
        context = self._format_context(hits)
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are an enterprise multimodal RAG assistant. "
                            "Answer strictly from supplied context and admit uncertainty when context is insufficient."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Question:\n"
                            f"{question}\n\n"
                            "Context:\n"
                            f"{context}\n\n"
                            "Instructions:\n"
                            "- Use concise bullets when helpful.\n"
                            "- Reference source file names in your answer.\n"
                            "- If data is missing, explicitly state what is missing."
                        )
                    ),
                ]
            )
            return str(response.content).strip()
        except Exception as exc:  # pragma: no cover - model/network branch
            logger.warning("LLM generation failed, using fallback answer: %s", exc)
            return self._fallback_answer(question, hits)

    def _generate_llamaindex(self, question: str, hits: list[RetrievalHit]) -> str:
        """Experimental path: falls back to LangChain if llamaindex synthesis is unavailable."""
        try:
            from llama_index.core import Document, SummaryIndex

            if not self._llm:
                return self._fallback_answer(question, hits)

            docs = [Document(text=hit.chunk.content, doc_id=hit.chunk.chunk_id) for hit in hits]
            if not docs:
                return "I could not find relevant context in the index."

            # Reuse OpenAI credentials through environment config if available.
            # If LlamaIndex LLM is not configured, this path may fail and fallback.
            index = SummaryIndex.from_documents(docs)
            query_engine = index.as_query_engine()
            query_text = (
                "Answer this question only from the retrieved multimodal context:\n"
                f"{question}\n"
            )
            result = query_engine.query(query_text)
            content = str(result).strip()
            if content:
                return content
            return self._fallback_answer(question, hits)
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
        if self.settings.orchestrator == "llamaindex":
            return self._generate_llamaindex(question, hits)
        return self._generate_langchain(question, hits)
