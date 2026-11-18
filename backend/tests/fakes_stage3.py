"""Test doubles for Stage 3 evidence analysis.

No test in the default suite calls Gemini. ``FakeAnalysisLLM`` replays scripted
responses and records what it was asked, so tests can assert on batching, on the
repair path, and on prompt-injection handling without spending quota.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.services.llm import LLMContractError, LLMResult

__all__ = [
    "FakeAnalysisLLM",
    "analysis_payload",
    "analysis_row",
]


def analysis_row(
    passage_id: str | uuid.UUID,
    *,
    relationship: str = "SUPPORTS",
    relevance: float = 0.9,
    evidence_quality: float = 0.8,
    stance_confidence: float = 0.85,
    quote: str = "",
    rationale: str = "The passage bears directly on the subclaim.",
) -> dict[str, Any]:
    """One analysis object in the shape the model returns."""
    return {
        "passage_id": str(passage_id),
        "relationship": relationship,
        "relevance": relevance,
        "evidence_quality": evidence_quality,
        "stance_confidence": stance_confidence,
        "quote": quote,
        "rationale": rationale,
    }


def analysis_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap analysis rows in the top-level response object."""
    return {"analyses": rows}


class FakeAnalysisLLM:
    """A scripted analysis provider.

    Each entry in ``script`` is consumed by one call. An entry may be:

    * a ``dict`` — returned as parsed JSON.
    * an ``Exception`` — raised instead of returning.
    """

    def __init__(
        self,
        script: list[Any] | None = None,
        *,
        model: str = "gemini-3.5-flash-lite",
        tokens_in: int = 200,
        tokens_out: int = 120,
    ) -> None:
        self.model = model
        self._script = list(script or [])
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self.calls: list[dict[str, Any]] = []

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
                f"FakeAnalysisLLM ran out of scripted responses after {len(self.calls)} call(s)"
            )

        item = self._script.pop(0)

        if isinstance(item, Exception):
            raise item

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
