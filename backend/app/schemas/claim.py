"""Pydantic schemas for claim creation and Stage 1 decomposition.

Validation lives here, not only in the database. A CHECK constraint rejects a
bad row at write time; these models reject it before an LLM call is spent and
before a transaction is opened, and they produce a useful error message.

Two distinct layers are modelled deliberately:

* ``Stage1Decomposition`` — what the LLM is allowed to return. Its ``weight``
  field accepts a range so a model proposing 0.7 does not hard-fail the whole
  response; the backend overwrites it with 1.0 regardless.
* ``SubclaimResponse`` — what is actually persisted and returned, where weight
  is pinned to exactly 1.0.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Enumerations (mirror the CHECK constraints in app/db/models.py) --------

ClaimDomain = Literal["technology", "science", "general"]
ClaimType = Literal[
    "factual", "causal", "comparative", "definitional", "quantitative"
]
ClaimStatus = Literal["queued", "running", "complete", "failed"]

# --- Limits ----------------------------------------------------------------

CLAIM_MIN_CHARS = 1
CLAIM_MAX_CHARS = 2000
SUB_CLAIM_MIN_CHARS = 1
SUB_CLAIM_MAX_CHARS = 300
MIN_SUBCLAIMS = 1
MAX_SUBCLAIMS = 8
HARD_CAP_SUBCLAIMS = 12

# Every persisted subclaim weight is exactly this. The LLM cannot influence it.
FIXED_SUBCLAIM_WEIGHT = Decimal("1.0")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class CreateClaimRequest(BaseModel):
    """Body of ``POST /api/claims``."""

    model_config = ConfigDict(extra="forbid")

    raw_text: Annotated[str, Field(min_length=CLAIM_MIN_CHARS, max_length=CLAIM_MAX_CHARS)]

    @field_validator("raw_text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject whitespace-only input and normalise surrounding space."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("claim text must not be empty or whitespace only")
        if not any(ch.isalpha() for ch in cleaned):
            raise ValueError("claim text must contain at least one letter")
        return cleaned


# ---------------------------------------------------------------------------
# Stage 1 — LLM-facing contract
# ---------------------------------------------------------------------------

class Stage1Subclaim(BaseModel):
    """One subclaim as returned by the model.

    ``weight`` is accepted but never trusted: see ``to_persisted``.
    """

    model_config = ConfigDict(extra="ignore")

    text: Annotated[str, Field(min_length=SUB_CLAIM_MIN_CHARS, max_length=SUB_CLAIM_MAX_CHARS)]
    claim_type: ClaimType
    weight: float = Field(default=1.0, ge=0.1, le=3.0)
    decomposition_note: str | None = None

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("subclaim text must not be empty")
        return cleaned

    def to_persisted(self, ordinal: int) -> "PersistedSubclaim":
        """Convert to a stored subclaim with the weight forced to 1.0."""
        return PersistedSubclaim(
            ordinal=ordinal,
            text=self.text,
            claim_type=self.claim_type,
            weight=FIXED_SUBCLAIM_WEIGHT,
            decomposition_note=self.decomposition_note,
        )


class Stage1Decomposition(BaseModel):
    """The full structured response expected from ``decompose.v1``."""

    model_config = ConfigDict(extra="ignore")

    domain: ClaimDomain
    subclaims: Annotated[
        list[Stage1Subclaim],
        Field(min_length=MIN_SUBCLAIMS, max_length=HARD_CAP_SUBCLAIMS),
    ]

    @field_validator("subclaims")
    @classmethod
    def _no_duplicates(cls, value: list[Stage1Subclaim]) -> list[Stage1Subclaim]:
        """Reject duplicate subclaims after normalisation.

        Comparison is casefolded with collapsed whitespace, so "HTTPS encrypts
        traffic." and "https  encrypts traffic" count as the same subclaim.
        """
        seen: set[str] = set()
        for item in value:
            key = " ".join(item.text.casefold().split())
            if key in seen:
                raise ValueError(f"duplicate subclaim: {item.text!r}")
            seen.add(key)
        return value

    @model_validator(mode="after")
    def _within_soft_cap(self) -> "Stage1Decomposition":
        """Enforce the soft cap of 8; the schema allows up to 12 to detect it."""
        if len(self.subclaims) > MAX_SUBCLAIMS:
            raise ValueError(
                f"expected at most {MAX_SUBCLAIMS} subclaims, got {len(self.subclaims)}"
            )
        return self

    def to_persisted(self) -> list["PersistedSubclaim"]:
        """Ordered persisted subclaims, ordinals 1..n, weights all 1.0."""
        return [
            sub.to_persisted(ordinal=index)
            for index, sub in enumerate(self.subclaims, start=1)
        ]


class PersistedSubclaim(BaseModel):
    """A subclaim in its stored form: weight is always exactly 1.0."""

    model_config = ConfigDict(extra="forbid")

    ordinal: Annotated[int, Field(ge=1)]
    text: Annotated[str, Field(min_length=SUB_CLAIM_MIN_CHARS, max_length=SUB_CLAIM_MAX_CHARS)]
    claim_type: ClaimType
    weight: Decimal = Field(default=FIXED_SUBCLAIM_WEIGHT)
    decomposition_note: str | None = None

    @field_validator("weight")
    @classmethod
    def _pin_weight(cls, value: Decimal) -> Decimal:
        """Any weight other than 1.0 is a bug, not user input."""
        if value != FIXED_SUBCLAIM_WEIGHT:
            raise ValueError(
                f"persisted subclaim weight must be exactly {FIXED_SUBCLAIM_WEIGHT}, got {value}"
            )
        return FIXED_SUBCLAIM_WEIGHT


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class Stage1Response(BaseModel):
    """Result of a completed Stage 1 run, as used by the pipeline.

    Carries the validated decomposition plus the run metadata the caller needs
    to record, without exposing the raw model response.
    """

    domain: ClaimDomain
    subclaims: list[PersistedSubclaim]
    prompt_version: str = "decompose.v1"
    model: str
    attempts: int
    tokens_in: int = 0
    tokens_out: int = 0


class SubclaimResponse(BaseModel):
    """A subclaim as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ordinal: int
    text: str
    claim_type: str
    weight: Decimal


class CreateClaimResponse(BaseModel):
    """Body of a successful ``POST /api/claims``."""

    id: uuid.UUID
    status: ClaimStatus
    domain: ClaimDomain
    subclaim_count: int


class ClaimResponse(BaseModel):
    """Body of ``GET /api/claims/{id}``.

    Deliberately excludes prompt contents, model config internals and any
    credential. ``error`` is surfaced because a failed claim is a legitimate
    state the client needs to explain to a user.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_text: str
    domain: ClaimDomain
    status: ClaimStatus
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    subclaims: list[SubclaimResponse] = Field(default_factory=list)
