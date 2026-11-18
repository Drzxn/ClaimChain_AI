"""Pydantic schemas for Stage 3 evidence analysis.

The LLM output is untrusted. Everything here exists to reject it rather than
coerce it: an invalid relationship is an error, not a default; an out-of-range
score is an error, not a clamp to the nearest bound. Silent coercion would let a
malformed response become a persisted evidence link that looks fine.

Quote validation is deliberately separate from parsing. Parsing says "this is
well-formed"; validation says "this quote actually appears in the passage it
claims to come from". Only the second one matters for traceability.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Enumerations ----------------------------------------------------------

Relationship = Literal["SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT"]

RELATIONSHIPS: tuple[str, ...] = (
    "SUPPORTS",
    "CONTRADICTS",
    "MIXED",
    "INSUFFICIENT",
)

# The overall-verdict tokens. The model must never emit these; their presence
# in a relationship field is a contract violation worth naming explicitly.
FORBIDDEN_VERDICT_TOKENS: frozenset[str] = frozenset(
    {"SUPPORTED", "CONTRADICTED", "MISLEADING", "INSUFFICIENT_EVIDENCE", "TRUE", "FALSE"}
)

Score = Annotated[float, Field(ge=0.0, le=1.0)]

# --- Limits ----------------------------------------------------------------

MAX_PASSAGES_PER_CALL = 8
MAX_QUOTE_CHARS = 600
MAX_RATIONALE_CHARS = 800

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip.

    Used for quote comparison only. The stored quote keeps its original form so
    it remains a true substring of the passage.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# LLM-facing contract
# ---------------------------------------------------------------------------

class PassageAnalysis(BaseModel):
    """One judgment, as returned by the model."""

    model_config = ConfigDict(extra="ignore")

    passage_id: str
    relationship: Relationship
    relevance: Score
    evidence_quality: Score
    stance_confidence: Score
    quote: str = Field(default="", max_length=MAX_QUOTE_CHARS)
    rationale: str = Field(default="", max_length=MAX_RATIONALE_CHARS)

    @field_validator("relationship", mode="before")
    @classmethod
    def _reject_verdict_tokens(cls, value: object) -> object:
        """Catch the model answering the overall question instead of the link.

        ``SUPPORTED`` is not a relationship. If it appears, the model has
        drifted into verdict territory, and that must fail loudly rather than
        be silently reinterpreted as ``SUPPORTS``.
        """
        if isinstance(value, str):
            token = value.strip().upper()
            if token in FORBIDDEN_VERDICT_TOKENS:
                raise ValueError(
                    f"{token!r} is an overall verdict, not an evidence relationship; "
                    f"expected one of {RELATIONSHIPS}"
                )
            return token
        return value

    @field_validator("passage_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value.strip()))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"passage_id is not a UUID: {value!r}") from exc

    @field_validator("quote", "rationale")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        return value.strip()


class AnalysisResponse(BaseModel):
    """The full structured response expected from ``evidence_analysis.v1``."""

    model_config = ConfigDict(extra="ignore")

    analyses: list[PassageAnalysis] = Field(default_factory=list)

    @field_validator("analyses")
    @classmethod
    def _no_duplicate_ids(cls, value: list[PassageAnalysis]) -> list[PassageAnalysis]:
        seen: set[str] = set()
        for item in value:
            if item.passage_id in seen:
                raise ValueError(f"duplicate passage_id in response: {item.passage_id}")
            seen.add(item.passage_id)
        return value


# ---------------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------------

class QuoteValidation(BaseModel):
    """Result of checking a quote against its passage."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    reason: str = ""
    # The quote as it should be stored: the original span from the passage when
    # an exact match was found, otherwise the model's text (kept for debugging).
    stored_quote: str = ""


def validate_quote(quote: str, passage_text: str) -> QuoteValidation:
    """Check that ``quote`` really appears in ``passage_text``.

    Two acceptance paths, in order:

    1. **Exact substring.** The quote appears verbatim. This is the only path
       that is fully trustworthy, and the stored quote is the matched span.
    2. **Whitespace-normalized substring.** The quote matches once whitespace
       runs are collapsed on both sides. This tolerates a model re-wrapping a
       line without tolerating paraphrase. The stored quote is taken from the
       *passage*, not from the model, so it remains a true substring.

    Anything else fails. A paraphrase, a quote from a different passage, or an
    empty quote all fail. An empty quote is accepted only for INSUFFICIENT
    judgments, which the caller handles.

    Returns:
        QuoteValidation with ``passed`` and the span to store.
    """
    if not quote or not quote.strip():
        return QuoteValidation(passed=False, reason="empty quote")

    if quote in passage_text:
        return QuoteValidation(passed=True, stored_quote=quote)

    # Whitespace-tolerant path: locate the normalized quote in the normalized
    # passage, then recover the corresponding span from the original text.
    norm_quote = normalize_whitespace(quote)
    if not norm_quote:
        return QuoteValidation(passed=False, reason="empty quote after normalization")

    norm_passage = normalize_whitespace(passage_text)
    position = norm_passage.find(norm_quote)
    if position == -1:
        return QuoteValidation(
            passed=False,
            reason="quote is not a verbatim substring of the passage",
        )

    # Map the normalized match back onto the original passage by walking both
    # strings in step. This keeps the stored quote a genuine substring.
    span = _map_normalized_span(passage_text, norm_passage, position, len(norm_quote))
    if span is None:
        return QuoteValidation(
            passed=False,
            reason="could not map normalized match back to passage text",
        )

    start, end = span
    return QuoteValidation(passed=True, stored_quote=passage_text[start:end])


def _map_normalized_span(
    original: str, normalized: str, norm_start: int, norm_length: int
) -> tuple[int, int] | None:
    """Map a span in ``normalized`` back to character offsets in ``original``."""
    norm_end = norm_start + norm_length

    orig_index = 0
    norm_index = 0
    start_orig: int | None = None

    while orig_index < len(original) and norm_index < norm_end:
        char = original[orig_index]

        if char.isspace():
            # A whitespace run in the original is a single space in normalized.
            if norm_index == norm_start and start_orig is None:
                start_orig = orig_index
            orig_index += 1
            if norm_index < len(normalized) and normalized[norm_index] == " ":
                norm_index += 1
            continue

        if norm_index == norm_start and start_orig is None:
            start_orig = orig_index

        orig_index += 1
        norm_index += 1

    if start_orig is None:
        return None

    return start_orig, orig_index


# ---------------------------------------------------------------------------
# Persistence-facing models
# ---------------------------------------------------------------------------

class ValidatedAnalysis(BaseModel):
    """A judgment that passed every check, ready to persist."""

    model_config = ConfigDict(frozen=True)

    passage_id: uuid.UUID
    relationship: Relationship
    relevance: float
    evidence_quality: float
    stance_confidence: float
    quote: str
    rationale: str
    quote_validation: Literal["passed", "failed"]

__all__ = [
    "AnalysisResponse",
    "FORBIDDEN_VERDICT_TOKENS",
    "MAX_PASSAGES_PER_CALL",
    "PassageAnalysis",
    "RELATIONSHIPS",
    "QuoteValidation",
    "ValidatedAnalysis",
    "normalize_whitespace",
    "validate_quote",
]
