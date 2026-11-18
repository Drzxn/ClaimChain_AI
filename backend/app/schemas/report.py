"""Pydantic contract for the Stage 6 explainable report.

The report is a **read-only projection** of persisted rows. Every field is
sourced from the database; nothing here is computed, inferred or generated.
That is what makes the report traceable: a reader can walk

    claim -> subclaim -> subclaim_verdict -> evidence_link -> passage -> source

and land on the same rows the report was built from.

The verdict fields are copied verbatim from ``verdicts`` and
``subclaim_verdicts``. Stage 6 never recalculates them, and no model may
override them.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OverallVerdict = Literal[
    "SUPPORTED",
    "CONTRADICTED",
    "MIXED",
    "INSUFFICIENT_EVIDENCE",
    "MISLEADING",
]

SubclaimVerdictLiteral = Literal["SUPPORTED", "CONTRADICTED", "MIXED", "INSUFFICIENT"]

Relationship = Literal["SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT"]


# ---------------------------------------------------------------------------
# A. Claim
# ---------------------------------------------------------------------------

class ReportClaim(BaseModel):
    """The claim being reported on."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_text: str
    domain: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# B. Overall verdict
# ---------------------------------------------------------------------------

class ReportVerdict(BaseModel):
    """The persisted Stage 4 verdict, copied without modification."""

    model_config = ConfigDict(from_attributes=True)

    verdict: OverallVerdict
    score: float
    coverage: float
    contradicted_weight_ratio: float
    subclaim_counts: dict[str, int] = Field(default_factory=dict)
    summary: str | None = None
    aggregation_version: str
    computed_at: datetime


# ---------------------------------------------------------------------------
# D. Evidence
# ---------------------------------------------------------------------------

class ReportEvidence(BaseModel):
    """One evidence link, joined to its passage and source.

    Carries the full identity chain so the frontend can render an evidence
    graph without a second query.
    """

    model_config = ConfigDict(from_attributes=True)

    evidence_link_id: uuid.UUID
    subclaim_id: uuid.UUID
    passage_id: uuid.UUID
    source_id: uuid.UUID

    relationship: Relationship
    relevance: float
    evidence_quality: float
    stance_confidence: float

    quote: str
    rationale: str
    quote_validation: Literal["passed", "failed"]

    # Passage context, so a quote can be read in place.
    passage_text: str
    passage_chunk_index: int
    quote_start: int
    quote_end: int

    # Source identity, denormalized for display.
    source_url: str
    source_domain: str
    source_title: str | None = None


# ---------------------------------------------------------------------------
# C. Subclaim breakdown
# ---------------------------------------------------------------------------

class ReportSubclaimVerdict(BaseModel):
    """The persisted Stage 4 subclaim verdict, copied without modification."""

    model_config = ConfigDict(from_attributes=True)

    verdict: SubclaimVerdictLiteral
    score: float
    coverage: float
    support_weight: float
    contradict_weight: float
    mixed_weight: float
    link_count: int
    rationale: str
    aggregation_version: str
    computed_at: datetime


class ReportSubclaim(BaseModel):
    """One subclaim with its verdict and the evidence judged against it."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ordinal: int
    text: str
    claim_type: str
    weight: float
    decomposition_note: str | None = None

    verdict: ReportSubclaimVerdict
    evidence: list[ReportEvidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# E. Sources
# ---------------------------------------------------------------------------

class ReportSource(BaseModel):
    """A source used by this report, with its Stage 5 assessment."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    url: str
    domain: str
    title: str | None = None
    publisher: str | None = None
    source_type: str
    published_at: date | None = None
    retrieved_at: datetime

    authority_score: float | None = None
    recency_score: float | None = None
    assessment_rationale: str | None = None
    assessed_at: datetime | None = None
    cites_primary_sources: bool | None = None

    # Set when this source's content duplicates another source's.
    duplicate_of: uuid.UUID | None = None
    is_duplicate: bool = False

    # How many evidence links in this report point at this source.
    evidence_count: int = 0


# ---------------------------------------------------------------------------
# F. Explanation
# ---------------------------------------------------------------------------

class ReportExplanation(BaseModel):
    """Prose explaining the verdict.

    ``narrative`` is the only field a model may ever write, and it is
    constrained to describe facts already present in the report. Everything
    else is assembled deterministically from stored values.
    """

    # Deterministic: built from the persisted verdict and counts.
    verdict_statement: str
    # Deterministic: one line per subclaim, from its persisted verdict.
    subclaim_statements: list[str] = Field(default_factory=list)
    # Deterministic: what supported and what contradicted, from stored links.
    supporting_points: list[str] = Field(default_factory=list)
    contradicting_points: list[str] = Field(default_factory=list)

    # Optional model-written prose. Empty when the model was not used or failed.
    narrative: str = ""
    narrative_source: Literal["deterministic", "model"] = "deterministic"


# ---------------------------------------------------------------------------
# G. Limitations
# ---------------------------------------------------------------------------

class ReportLimitation(BaseModel):
    """One limitation, with where it came from."""

    code: str
    message: str
    source: Literal["verdict", "evidence", "sources", "report"]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class ClaimReport(BaseModel):
    """The complete Stage 6 report.

    Stable, typed and frontend-friendly. Every value traces to a database row.
    """

    model_config = ConfigDict(from_attributes=True)

    claim: ReportClaim
    verdict: ReportVerdict
    subclaims: list[ReportSubclaim] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    explanation: ReportExplanation
    limitations: list[ReportLimitation] = Field(default_factory=list)

    # Provenance for the report itself.
    report_version: str = "report.v1"
    generated_at: datetime
    evidence_link_count: int = 0


__all__ = [
    "ClaimReport",
    "OverallVerdict",
    "Relationship",
    "ReportClaim",
    "ReportEvidence",
    "ReportExplanation",
    "ReportLimitation",
    "ReportSource",
    "ReportSubclaim",
    "ReportSubclaimVerdict",
    "ReportVerdict",
    "SubclaimVerdictLiteral",
]
