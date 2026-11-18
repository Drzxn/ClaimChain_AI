"""Test doubles for the Gemini service.

No test in the default suite calls the real Gemini API. ``FakeGemini`` returns
scripted responses and records what it was asked, so tests can assert on the
number of attempts and on the repair prompt without touching the network.
"""

from __future__ import annotations

from typing import Any

from app.services.llm import (
    GeminiService,
    LLMContractError,
    LLMNotConfiguredError,
    LLMResult,
    LLMTransportError,
)

__all__ = [
    "FakeGemini",
    "LLMContractError",
    "LLMNotConfiguredError",
    "LLMTransportError",
    "valid_decomposition",
]


def valid_decomposition(
    *,
    domain: str = "technology",
    subclaims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A minimal valid Stage 1 payload."""
    if subclaims is None:
        subclaims = [
            {
                "text": "HTTPS encrypts data in transit between browser and server.",
                "claim_type": "factual",
                "weight": 1.0,
            },
            {
                "text": "HTTPS makes a website completely secure against all attacks.",
                "claim_type": "factual",
                "weight": 1.0,
            },
        ]
    return {"domain": domain, "subclaims": subclaims}


class FakeGemini(GeminiService):
    """A GeminiService that replays scripted responses.

    Each entry in ``script`` is consumed by one call. An entry may be:

    * a ``dict`` — returned as parsed JSON.
    * an ``Exception`` instance — raised instead of returning.
    * a ``str`` — returned as raw text that fails to parse as JSON.
    """

    def __init__(
        self,
        script: list[Any] | None = None,
        *,
        model: str = "gemini-3.5-flash-lite",
        tokens_in: int = 100,
        tokens_out: int = 50,
    ) -> None:
        # Deliberately skip super().__init__: no settings, no real client.
        self._api_key = "fake-key-for-tests"
        self.model = model
        self._client = None
        self._temperature = 0.0
        self._script = list(script or [])
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self.calls: list[dict[str, Any]] = []

    @property
    def configured(self) -> bool:
        return True

    def generate_json(
        self,
        *,
        system_instruction: str,
        user_content: str,
        json_schema: dict[str, Any],
        max_output_tokens: int = 2048,
    ) -> LLMResult:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "user_content": user_content,
                "json_schema": json_schema,
            }
        )

        if not self._script:
            raise AssertionError(
                f"FakeGemini ran out of scripted responses after {len(self.calls)} call(s)"
            )

        item = self._script.pop(0)

        if isinstance(item, Exception):
            raise item

        if isinstance(item, str):
            # Simulate a response that is not valid JSON.
            raise LLMContractError(f"Gemini response was not valid JSON: {item}")

        return LLMResult(
            data=item,
            model=self.model,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            cost_usd=0.0,
            raw_text="{}",
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)
