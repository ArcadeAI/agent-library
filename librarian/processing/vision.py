"""
VLM-backed vision pipeline -- the v0.14 single image path.

A *vision describer* turns raw image bytes into searchable **text**: a single
hosted vision-language-model (VLM) call returns a short description plus any
text transcribed from the image. The orchestrator stores that text as the
chunk's content and embeds it in the ordinary TEXT space. This replaces the
retired CLIP image-embedding path (strategy "Option C / Hybrid"; see issue #53),
so image search needs only the text index -- no separate vector space.

The interface is provider-agnostic: :class:`LLMVisionDescriber` drives either an
OpenAI (default ``gpt-4o``) or Anthropic (``claude-3-5-sonnet``) vision model
through their synchronous SDKs. The SDKs are imported lazily so the dependency
is optional; callers that inject their own describer (e.g. tests) need neither.
"""

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from librarian.config import VLM_MAX_TOKENS, VLM_MODEL, VLM_PROVIDER

logger = logging.getLogger(__name__)

__all__ = [
    "VLM_SUPPORTED_MIME",
    "BaseVisionDescriber",
    "LLMVisionDescriber",
    "VisionError",
    "VisionResult",
    "get_vision_describer",
    "guess_image_mime",
    "is_vlm_supported_mime",
]

# Media types every supported hosted VLM accepts. Anthropic's messages API
# restricts image ``media_type`` to these four, and OpenAI vision input is the
# same set (non-animated GIF); .bmp/.tiff are recognised as images by the parser
# registry but no provider ingests them, so they are marked terminally
# ``unsupported`` rather than retried forever.
VLM_SUPPORTED_MIME: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def is_vlm_supported_mime(mime: str) -> bool:
    """Whether a hosted VLM provider accepts this image media type."""
    return mime in VLM_SUPPORTED_MIME


# A fenced reply -- ```json ... ``` or ``` ... ``` -- on one or many lines.
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

# The single prompt sent with every image. We ask for a strict JSON object so
# the description and the verbatim transcription stay separable downstream.
_VISION_SYSTEM = (
    "You are an image understanding assistant for a search index. Describe images "
    "factually and transcribe any text exactly as it appears."
)
_VISION_PROMPT = (
    "Describe this image for a search index, then transcribe any text visible in "
    "it verbatim. Respond with a single JSON object with exactly two string keys: "
    '"description" (a concise factual description) and "text" (all text '
    "transcribed verbatim, or an empty string if there is none). Output only the "
    "JSON object, with no markdown fences or commentary."
)

# Extension -> MIME media type for the image formats the registry recognises.
_IMAGE_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


def guess_image_mime(suffix_or_path: str) -> str:
    """Return the image MIME type for a file suffix or path (default PNG)."""
    text = suffix_or_path.lower()
    for ext, mime in _IMAGE_MIME.items():
        if text.endswith(ext):
            return mime
    return "image/png"


class VisionError(RuntimeError):
    """Raised when a VLM call fails or returns unusable output."""


@dataclass
class VisionResult:
    """The text a VLM extracted from an image."""

    description: str
    transcribed_text: str = ""

    @property
    def content(self) -> str:
        """The combined searchable text used as the chunk content."""
        parts: list[str] = []
        if self.description.strip():
            parts.append(self.description.strip())
        if self.transcribed_text.strip():
            parts.append(f"Transcribed text:\n{self.transcribed_text.strip()}")
        return "\n\n".join(parts).strip()


def parse_vlm_response(raw: str) -> VisionResult:
    """Parse a VLM's raw text into a :class:`VisionResult`.

    Expects the strict JSON object the prompt requests, but tolerates models that
    wrap it in markdown fences or ignore the format entirely (the whole reply is
    then treated as the description).
    """
    text = (raw or "").strip()
    if not text:
        raise VisionError("VLM returned an empty response")

    # Strip a surrounding markdown fence if present (handles single-line
    # ```{...}``` as well as multi-line ```json\n...\n```).
    match = _FENCE_RE.match(text)
    candidate = match.group(1).strip() if match else text

    try:
        data = json.loads(candidate)
    except (TypeError, ValueError):
        data = None

    if isinstance(data, dict):
        # ``or ""`` (not a ``.get`` default) so an explicit JSON ``null`` -- which
        # models emit for absent text -- doesn't become the literal string "None".
        return VisionResult(
            description=str(data.get("description") or ""),
            transcribed_text=str(data.get("text") or ""),
        )
    # Not JSON: keep the whole reply as the description.
    return VisionResult(description=text)


class BaseVisionDescriber(ABC):
    """Provider-agnostic interface for describing an image as text."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model (stored as ``model_version``)."""
        ...

    @abstractmethod
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> VisionResult:
        """Return a :class:`VisionResult` for ``image_bytes``.

        Raises:
            VisionError: if the model call fails or returns unusable output.
        """
        ...


class LLMVisionDescriber(BaseVisionDescriber):
    """A :class:`BaseVisionDescriber` backed by a hosted VLM (OpenAI/Anthropic)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        client: Any = None,
    ) -> None:
        self._provider = (provider or VLM_PROVIDER).lower()
        self._model = model or VLM_MODEL
        self._api_key = api_key
        self._max_tokens = max_tokens or VLM_MAX_TOKENS
        self._client = client
        if self._provider not in ("openai", "anthropic"):
            raise VisionError(
                f"Unknown VLM provider {self._provider!r}; expected 'openai' or 'anthropic'."
            )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def client(self) -> Any:
        """Lazily build the synchronous provider SDK client."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        if self._provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise VisionError(
                    "openai package required for the VLM vision path. "
                    "Install with: pip install 'agent-library[vlm]'"
                ) from e
            return OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise VisionError(
                "anthropic package required for the VLM vision path. "
                "Install with: pip install 'agent-library[vlm]'"
            ) from e
        return Anthropic(api_key=self._api_key) if self._api_key else Anthropic()

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> VisionResult:
        if not image_bytes:
            raise VisionError("empty image bytes")
        b64 = base64.b64encode(image_bytes).decode("ascii")
        try:
            raw = (
                self._call_openai(b64, mime_type)
                if self._provider == "openai"
                else self._call_anthropic(b64, mime_type)
            )
        except VisionError:
            raise
        except Exception as e:  # provider/network/SDK errors
            raise VisionError(f"VLM call failed ({self._provider}/{self._model}): {e}") from e
        return parse_vlm_response(raw)

    def _call_openai(self, b64: str, mime_type: str) -> str:
        data_url = f"data:{mime_type};base64,{b64}"
        response = self.client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            # A truncated reply is almost certainly invalid JSON mid-string;
            # raise so the image lands in the retryable 'failed' state instead of
            # storing the garbage prefix with status 'ok'.
            raise VisionError(
                "VLM response truncated at max_tokens; raise VLM_MAX_TOKENS and retry"
            )
        return choice.message.content or ""

    def _call_anthropic(self, b64: str, mime_type: str) -> str:
        response = self.client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_VISION_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _VISION_PROMPT},
                    ],
                }
            ],
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise VisionError(
                "VLM response truncated at max_tokens; raise VLM_MAX_TOKENS and retry"
            )
        parts = [getattr(block, "text", "") for block in response.content]
        return "".join(parts)


def get_vision_describer() -> BaseVisionDescriber | None:
    """Build the configured VLM describer, or ``None`` if it can't be created.

    Returns ``None`` only on a configuration error (unknown provider); a missing
    SDK or API key surfaces later as a :class:`VisionError` at ``describe()`` time
    so the orchestrator records a retryable ``processing_status='failed'`` chunk.
    """
    try:
        return LLMVisionDescriber()
    except VisionError as e:
        logger.warning("Vision describer unavailable: %s", e)
        return None
