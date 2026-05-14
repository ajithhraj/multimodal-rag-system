from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from multimodal_rag.config import Settings

logger = logging.getLogger(__name__)


class VisionCaptioner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._enabled = settings.has_openai_api_key()
        self._model_name = settings.vision_model
        self._api_key = settings.openai_api_key
        self._llm = None
        self._human_message_type = None
        self._system_message_type = None
        if self._enabled:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                self._human_message_type = HumanMessage
                self._system_message_type = SystemMessage
                self._llm = ChatOpenAI(
                    model=self._model_name,
                    api_key=self._api_key,
                    temperature=0.0,
                )
            except Exception as exc:  # pragma: no cover - optional dependency branch
                if self.settings.strict_api_only_mode():
                    raise RuntimeError(
                        "OpenAI vision captioner is required in API-only mode but could not be initialized."
                    ) from exc
                logger.warning("OpenAI vision captioner unavailable, using filename-only captions: %s", exc)
                self._llm = None
        elif self.settings.strict_api_only_mode():
            raise RuntimeError("OpenAI API key is required for image understanding in API-only mode.")

    def allow_local_ocr(self) -> bool:
        return not self.settings.strict_api_only_mode()

    @staticmethod
    def _to_data_url(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def caption(self, image_path: Path) -> str:
        if not self._llm or self._human_message_type is None or self._system_message_type is None:
            if self.settings.strict_api_only_mode():
                raise RuntimeError("OpenAI vision captioning is required in API-only mode.")
            return f"Image file named {image_path.name}."
        try:
            message = self._human_message_type(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe this image for retrieval in a multimodal RAG system. "
                            "Mention visible objects, labels, numbers, charts, and scene context."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._to_data_url(image_path),
                        },
                    },
                ]
            )
            response = self._llm.invoke(
                [
                    self._system_message_type(
                        content="You write concise factual image descriptions for semantic retrieval."
                    ),
                    message,
                ]
            )
            return str(response.content).strip()
        except Exception as exc:  # pragma: no cover - external model/network branch
            if self.settings.strict_api_only_mode():
                raise RuntimeError(f"Vision captioning failed for {image_path}: {exc}") from exc
            logger.warning("Vision captioning failed for %s: %s", image_path, exc)
            return f"Image file named {image_path.name}."


def run_ocr(image_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception:  # pragma: no cover - OCR runtime branch
        return ""
