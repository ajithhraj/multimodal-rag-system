from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from multimodal_rag.config import Settings

logger = logging.getLogger(__name__)


class VisionCaptioner:
    def __init__(self, settings: Settings):
        self._enabled = bool(settings.openai_api_key)
        self._model_name = settings.vision_model
        self._api_key = settings.openai_api_key
        self._llm = None
        if self._enabled:
            self._llm = ChatOpenAI(
                model=self._model_name,
                api_key=self._api_key,
                temperature=0.0,
            )

    @staticmethod
    def _to_data_url(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def caption(self, image_path: Path) -> str:
        if not self._llm:
            return f"Image file named {image_path.name}."
        try:
            message = HumanMessage(
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
                    SystemMessage(
                        content="You write concise factual image descriptions for semantic retrieval."
                    ),
                    message,
                ]
            )
            return str(response.content).strip()
        except Exception as exc:  # pragma: no cover - external model/network branch
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
