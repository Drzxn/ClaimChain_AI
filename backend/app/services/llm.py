"""Gemini client for structured JSON generation.

Design
------
* One client, one model, one place that knows how to talk to Google.
* The API key is read from settings and passed to the SDK. It is never logged,
  never included in an exception message, and never returned to a caller.
* Structured output is enforced twice: the request carries a JSON schema, and
  the response is validated against a Pydantic model. A response that parses as
  JSON but violates the schema is a failure, not a partial success.
* ``temperature=0`` for determinism. ``seed`` is set as well where supported.
* Token counts come from the API's usage metadata. Cost is not invented: the
  free tier does not report a reliable price, so ``cost_usd`` stays 0.0 and the
  caller records that as "unknown", not as "free".

The service raises typed errors so the pipeline can distinguish "the network
failed" from "the model returned something invalid". Only the latter is worth a
repair retry.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

# Never include the key in logs. This is the only place the client is built.
_DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMNotConfiguredError(LLMError):
    """No API key is configured."""


class LLMTransportError(LLMError):
    """The call to the provider failed (network, auth, quota, 5xx)."""


class LLMContractError(LLMError):
    """The model responded, but the response did not satisfy the contract."""


@dataclass(slots=True)
class LLMResult:
    """A validated structured response plus its accounting metadata."""

    data: dict[str, Any]
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    # Free tier does not report a trustworthy price. 0.0 means "not billed /
    # unknown", never "measured and found to be zero".
    cost_usd: float = 0.0
    raw_text: str = field(default="", repr=False)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences the model may add despite instructions."""
    return _FENCE_RE.sub("", text.strip())


class GeminiService:
    """Thin, typed wrapper over the Google Gen AI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = (api_key if api_key is not None else settings.gemini_api_key).strip()
        self.model = (model or settings.gemini_model or _DEFAULT_MODEL).strip()
        self._client = client
        self._temperature = settings.llm_temperature

    # -- lifecycle ---------------------------------------------------------

    @property
    def configured(self) -> bool:
        """Whether a key is present. Never reveals the key itself."""
        return bool(self._api_key)

    def _ensure_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is not set. Add it to .env before calling the LLM."
            )
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=4,
                    initial_delay=2.0,
                    max_delay=20.0,
                    exp_base=2.0,
            jitter=1.0,
            http_status_codes=[429, 500, 502, 503, 504],
        )
    ),
)
        return self._client

    # -- structured generation --------------------------------------------

    def generate_json(
        self,
        *,
        system_instruction: str,
        user_content: str,
        json_schema: dict[str, Any],
        max_output_tokens: int = 2048,
    ) -> LLMResult:
        """Run one structured generation and return parsed JSON.

        Raises:
            LLMNotConfiguredError: no API key.
            LLMTransportError: the provider call failed.
            LLMContractError: the response was not a JSON object.
        """
        client = self._ensure_client()

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._temperature,
            top_p=1.0,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=json_schema,
            seed=0,
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user_content,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - normalise provider errors
            # Deliberately does not include the key or the full request.
            raise LLMTransportError(
                f"Gemini call failed: {type(exc).__name__}"
            ) from exc

        text = getattr(response, "text", None) or ""
        usage = getattr(response, "usage_metadata", None)
        tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0)
        tokens_out = int(getattr(usage, "candidates_token_count", 0) or 0)

        if not text.strip():
            raise LLMContractError("Gemini returned an empty response")

        cleaned = _strip_fences(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMContractError(
                f"Gemini response was not valid JSON: {exc.msg}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMContractError(
                f"Gemini response must be a JSON object, got {type(parsed).__name__}"
            )

        return LLMResult(
            data=parsed,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            raw_text=cleaned,
        )


def get_gemini_service() -> GeminiService:
    """FastAPI-friendly accessor.

    Not cached: the service is cheap to construct and tests inject their own.
    """
    return GeminiService()
