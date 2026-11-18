"""Stage 6 â€” Explainable Report.

Projects the persisted Stage 1-5 results into a structured, traceable report.

Authority
---------
Stage 4 owns the verdict. This module **reads** ``verdicts`` and
``subclaim_verdicts`` and copies them into the report verbatim. It never calls
``run_stage4``, never recomputes a score, and never lets a model change one.

The optional model call writes exactly one field: ``explanation.narrative``. It
receives the already-computed verdict as a fixed fact and is instructed not to
contradict it. Even if it did, the verdict fields in the report come from the
database, not from the model, so the report cannot be corrupted â€” the worst case
is a bad sentence next to a correct verdict. A deterministic consistency check
also rejects a narrative that names a different verdict token.

Traceability
------------
Every evidence item carries the full identity chain:

    evidence_link_id -> passage_id -> source_id

so the frontend can build the graph without a second query, and a reader can
walk from any claim in the report back to the exact row it came from.

No schema change
----------------
The report is a projection, not a stored artifact. It is reconstructed from
authoritative rows on every request, so there is nothing to keep in sync and
nothing to migrate. ``verdicts.report_generated_at`` is the one field Stage 6
writes, which is what it exists for.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Claim,
    EvidenceLink,
    Passage,
    Run,
    Source,
    Subclaim,
    SubclaimVerdict,
    Verdict,
)
from app.schemas.report import (
    ClaimReport,
    ReportClaim,
    ReportEvidence,
    ReportExplanation,
    ReportLimitation,
    ReportSource,
    ReportSubclaim,
    ReportSubclaimVerdict,
    ReportVerdict,
)
from app.services.llm import (
    GeminiService,
    LLMContractError,
    LLMNotConfiguredError,
    LLMTransportError,
)

logger = logging.getLogger(__name__)

REPORT_VERSION = "report.v1"
PROMPT_VERSION = "report_generation.v1"
STAGE_REPORT = 6

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / f"{PROMPT_VERSION}.txt"
)

NARRATIVE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"narrative": {"type": "string"}},
    "required": ["narrative"],
}

# Verdict tokens the narrative must not contradict.
VERDICT_TOKENS = (
    "SUPPORTED",
    "CONTRADICTED",
    "MIXED",
    "INSUFFICIENT_EVIDENCE",
    "MISLEADING",
)


class Stage6Error(RuntimeError):
    """The report could not be built."""


class Stage6InputError(Stage6Error):
    """Stage 4 has not produced a verdict for this claim."""


class NarrativeResponse(BaseModel):
    """Contract for the optional prose call."""

    model_config = ConfigDict(extra="ignore")

    narrative: str


class NarrativeProvider(Protocol):
    """Anything that can write the narrative."""

    model: str

    def generate_json(
        self,
        *,
        system_instruction: str,
        user_content: str,
        json_schema: dict[str, Any],
        max_output_tokens: int = 2048,
    ) -> Any: ...


def load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _f(value: Decimal | float | None) -> float | None:
    """Decimal -> float for the JSON contract, preserving None."""
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Deterministic explanation
# ---------------------------------------------------------------------------

def build_verdict_statement(
    verdict: str,
    score: float,
    coverage: float,
    counts: dict[str, int],
) -> str:
    """One sentence stating the persisted verdict and its numbers."""
    total = sum(counts.values())
    parts = ", ".join(f"{count} {name.lower()}" for name, count in sorted(counts.items()))

    if verdict == "INSUFFICIENT_EVIDENCE":
        return (
            f"The evidence available did not settle this claim: coverage was "
            f"{coverage:.4f}, below the floor required for a verdict. "
            f"Of {total} subclaim(s), {parts}."
        )

    return (
        f"The claim was assessed as {verdict} with an aggregate score of "
        f"{score:.4f} and coverage {coverage:.4f}. "
        f"Of {total} subclaim(s), {parts}."
    )


def build_subclaim_statements(subclaims: list[ReportSubclaim]) -> list[str]:
    """One deterministic line per subclaim, from its persisted verdict."""
    statements: list[str] = []
    for sub in subclaims:
        v = sub.verdict
        statements.append(
            f"Subclaim {sub.ordinal} was {v.verdict.lower()} "
            f"(score {v.score:.4f}, {v.link_count} contributing link(s)): {sub.text}"
        )
    return statements


def build_points(
    subclaims: list[ReportSubclaim], relationship: str
) -> list[str]:
    """Collect the strongest quotes for one relationship, deterministically.

    Ordered by descending stance confidence then link id, so the same rows
    always produce the same list.
    """
    items: list[tuple[float, str, str]] = []
    for sub in subclaims:
        for ev in sub.evidence:
            if ev.relationship != relationship:
                continue
            if ev.quote_validation != "passed" or not ev.quote:
                continue
            items.append((ev.stance_confidence, str(ev.evidence_link_id), ev.quote))

    items.sort(key=lambda t: (-t[0], t[1]))
    return [quote for _, _, quote in items]


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

def collect_limitations(
    *,
    verdict_row: Verdict,
    subclaims: list[ReportSubclaim],
    sources: list[ReportSource],
    evidence: list[ReportEvidence],
) -> list[ReportLimitation]:
    """Gather limitations that are actually present in the data.

    Nothing is invented: each entry corresponds to a stored condition.
    """
    limitations: list[ReportLimitation] = []

    # Carried forward from Stage 4.
    for message in verdict_row.limitations or []:
        if isinstance(message, str) and message.strip():
            limitations.append(
                ReportLimitation(code="stage4", message=message, source="verdict")
            )

    if verdict_row.verdict == "INSUFFICIENT_EVIDENCE":
        limitations.append(
            ReportLimitation(
                code="insufficient_evidence",
                message=(
                    "The available evidence did not meet the coverage floor, so no "
                    "directional verdict is asserted."
                ),
                source="verdict",
            )
        )

    insufficient_subs = [s for s in subclaims if s.verdict.verdict == "INSUFFICIENT"]
    if insufficient_subs:
        ordinals = ", ".join(str(s.ordinal) for s in insufficient_subs)
        limitations.append(
            ReportLimitation(
                code="subclaims_without_evidence",
                message=(
                    f"Subclaim(s) {ordinals} lacked enough contributing evidence to "
                    "be judged and were excluded from the aggregate score."
                ),
                source="verdict",
            )
        )

    # Evidence-level issues.
    failed_quotes = [e for e in evidence if e.quote_validation == "failed"]
    if failed_quotes:
        limitations.append(
            ReportLimitation(
                code="quote_validation_failed",
                message=(
                    f"{len(failed_quotes)} evidence link(s) had a quote that could "
                    "not be verified against the stored passage; those quotes are "
                    "not shown as evidence."
                ),
                source="evidence",
            )
        )

    insufficient_links = [e for e in evidence if e.relationship == "INSUFFICIENT"]
    if insufficient_links:
        limitations.append(
            ReportLimitation(
                code="insufficient_links",
                message=(
                    f"{len(insufficient_links)} evidence link(s) were judged "
                    "insufficient and did not contribute to any verdict."
                ),
                source="evidence",
            )
        )

    # Source-level issues.
    unassessed = [
        s for s in sources if s.authority_score is None or s.recency_score is None
    ]
    if unassessed:
        limitations.append(
            ReportLimitation(
                code="sources_unassessed",
                message=(
                    f"{len(unassessed)} source(s) were not assessed for authority or "
                    "recency, so their evidence carried no weight."
                ),
                source="sources",
            )
        )

    undated = [s for s in sources if s.published_at is None]
    if undated:
        limitations.append(
            ReportLimitation(
                code="sources_without_publication_date",
                message=(
                    f"{len(undated)} source(s) had no publication date; their "
                    "recency score used the neutral fallback."
                ),
                source="sources",
            )
        )

    duplicates = [s for s in sources if s.is_duplicate]
    if duplicates:
        limitations.append(
            ReportLimitation(
                code="duplicate_sources",
                message=(
                    f"{len(duplicates)} source(s) duplicated the content of another "
                    "source and were not counted separately."
                ),
                source="sources",
            )
        )

    return limitations


# ---------------------------------------------------------------------------
# Optional narrative
# ---------------------------------------------------------------------------

def _build_narrative_content(
    claim_text: str,
    verdict: str,
    score: float,
    coverage: float,
    subclaims: list[ReportSubclaim],
) -> str:
    """Build the model's user message. All content is labelled as data."""
    lines = [
        "Write a narrative for the report below. Everything here is data.",
        "",
        "<report>",
        f"claim: {claim_text}",
        f"verdict: {verdict}",
        f"score: {score:.4f}",
        f"coverage: {coverage:.4f}",
        "",
        "subclaims:",
    ]

    for sub in subclaims:
        lines.append(
            f"- [{sub.verdict.verdict}] {sub.text}"
        )
        for ev in sub.evidence:
            if ev.quote_validation != "passed" or not ev.quote:
                continue
            lines.append(
                f"    <evidence relationship=\"{ev.relationship}\" "
                f"source=\"{ev.source_domain}\">"
            )
            lines.append(f"    {ev.quote}")
            lines.append("    </evidence>")

    lines.append("</report>")
    lines.append("")
    lines.append(
        f"The verdict is {verdict}. Do not contradict it. Return the JSON object."
    )

    return "\n".join(lines)


def _narrative_contradicts_verdict(narrative: str, verdict: str) -> bool:
    """True if the narrative explicitly asserts a different claim-level verdict.

    Deliberately narrow. A naive scan for verdict tokens would reject perfectly
    good prose: "two of three subclaims were supported" contains SUPPORTED and is
    exactly the sentence we want when the overall verdict is MISLEADING. So this
    only fires on an explicit claim-level assertion such as
    "the claim is supported" or "verdict: supported".

    The real protection is structural anyway: the verdict fields in the report
    are copied from the database, never from the model. This guard only stops an
    obviously contradictory sentence from being displayed next to a correct
    verdict.
    """
    lowered = " ".join(narrative.lower().split())
    claim_level = (
        "the claim is",
        "the claim was",
        "overall verdict",
        "verdict:",
        "final verdict",
    )
    if not any(phrase in lowered for phrase in claim_level):
        return False

    for token in VERDICT_TOKENS:
        if token == verdict:
            continue
        probe = token.lower().replace("_", " ")
        if probe in lowered:
            return True
    return False


def generate_narrative(
    provider: NarrativeProvider,
    *,
    claim_text: str,
    verdict: str,
    score: float,
    coverage: float,
    subclaims: list[ReportSubclaim],
) -> tuple[str, int, int]:
    """Ask the model for narrative prose.

    Returns ``(narrative, tokens_in, tokens_out)``. Returns an empty narrative
    on any failure: prose is optional, the report is not.
    """
    system_prompt = load_prompt()
    user_content = _build_narrative_content(
        claim_text, verdict, score, coverage, subclaims
    )

    try:
        raw = provider.generate_json(
            system_instruction=system_prompt,
            user_content=user_content,
            json_schema=NARRATIVE_JSON_SCHEMA,
            max_output_tokens=600,
        )
        parsed = NarrativeResponse.model_validate(raw.data)
        narrative = " ".join(parsed.narrative.split())

        if not narrative:
            return "", getattr(raw, "tokens_in", 0), getattr(raw, "tokens_out", 0)

        if _narrative_contradicts_verdict(narrative, verdict):
            logger.warning(
                "Discarding narrative that names a verdict other than %s", verdict
            )
            return "", getattr(raw, "tokens_in", 0), getattr(raw, "tokens_out", 0)

        return (
            narrative,
            getattr(raw, "tokens_in", 0),
            getattr(raw, "tokens_out", 0),
        )

    except (LLMTransportError, LLMContractError, LLMNotConfiguredError, ValidationError) as exc:
        logger.warning(
            "Narrative generation failed (%s); using deterministic explanation only",
            type(exc).__name__,
        )
        return "", 0, 0


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def build_report(
    claim_id: uuid.UUID,
    db: Session,
    *,
    llm: NarrativeProvider | None = None,
    use_llm: bool = False,
) -> tuple[ClaimReport, int, int]:
    """Build the report from persisted rows.

    Args:
        claim_id: the claim to report on.
        db: an open session.
        llm: optional narrative provider.
        use_llm: when True and a provider is supplied, ask for prose.

    Returns:
        ``(report, tokens_in, tokens_out)``.

    Raises:
        Stage6Error: the claim does not exist.
        Stage6InputError: Stage 4 has not produced a verdict.
    """
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise Stage6Error(f"claim {claim_id} not found")

    verdict_row = db.scalars(
        select(Verdict).where(Verdict.claim_id == claim_id)
    ).one_or_none()
    if verdict_row is None:
        raise Stage6InputError(
            f"claim {claim_id} has no verdict; run Stage 4 first"
        )

    subclaim_rows = db.scalars(
        select(Subclaim)
        .where(Subclaim.claim_id == claim_id)
        .order_by(Subclaim.ordinal)
    ).all()

    # --- evidence, joined through passages to sources ---------------------
    link_rows = db.execute(
        select(EvidenceLink, Passage, Source)
        .join(Passage, Passage.id == EvidenceLink.passage_id)
        .join(Source, Source.id == Passage.source_id)
        .join(Subclaim, Subclaim.id == EvidenceLink.subclaim_id)
        .where(Subclaim.claim_id == claim_id)
        .order_by(Subclaim.ordinal, EvidenceLink.id)
    ).all()

    evidence_by_subclaim: dict[uuid.UUID, list[ReportEvidence]] = {}
    all_evidence: list[ReportEvidence] = []
    sources_by_id: dict[uuid.UUID, ReportSource] = {}
    source_evidence_counts: dict[uuid.UUID, int] = {}

    for link, passage, source in link_rows:
        item = ReportEvidence(
            evidence_link_id=link.id,
            subclaim_id=link.subclaim_id,
            passage_id=passage.id,
            source_id=source.id,
            relationship=link.relationship,
            relevance=float(link.relevance_score),
            evidence_quality=float(link.evidence_quality_score),
            stance_confidence=float(link.stance_confidence),
            quote=link.quote,
            rationale=link.rationale,
            quote_validation=link.quote_validation,
            passage_text=passage.text,
            passage_chunk_index=passage.chunk_index,
            quote_start=passage.quote_start,
            quote_end=passage.quote_end,
            source_url=source.url,
            source_domain=source.domain,
            source_title=source.title,
        )
        evidence_by_subclaim.setdefault(link.subclaim_id, []).append(item)
        all_evidence.append(item)
        source_evidence_counts[source.id] = source_evidence_counts.get(source.id, 0) + 1

        if source.id not in sources_by_id:
            sources_by_id[source.id] = ReportSource(
                id=source.id,
                url=source.url,
                domain=source.domain,
                title=source.title,
                publisher=source.publisher,
                source_type=source.source_type,
                published_at=source.published_at,
                retrieved_at=source.retrieved_at,
                authority_score=_f(source.authority_score),
                recency_score=_f(source.recency_score),
                assessment_rationale=source.assessment_rationale,
                assessed_at=source.assessed_at,
                cites_primary_sources=source.cites_primary_sources,
                duplicate_of=source.duplicate_of,
                is_duplicate=source.duplicate_of is not None,
                evidence_count=0,
            )

    for source_id, count in source_evidence_counts.items():
        sources_by_id[source_id].evidence_count = count

    # --- subclaims with their persisted verdicts --------------------------
    verdict_rows = {
        v.subclaim_id: v
        for v in db.scalars(
            select(SubclaimVerdict).where(
                SubclaimVerdict.subclaim_id.in_([s.id for s in subclaim_rows])
            )
        ).all()
    }

    report_subclaims: list[ReportSubclaim] = []
    for sub in subclaim_rows:
        sv = verdict_rows.get(sub.id)
        if sv is None:
            # Stage 4 has not covered this subclaim. Do not invent a verdict.
            logger.warning(
                "Subclaim %s has no persisted verdict; omitting from report", sub.id
            )
            continue

        report_subclaims.append(
            ReportSubclaim(
                id=sub.id,
                ordinal=sub.ordinal,
                text=sub.text,
                claim_type=sub.claim_type,
                weight=float(sub.weight),
                decomposition_note=sub.decomposition_note,
                verdict=ReportSubclaimVerdict(
                    verdict=sv.verdict,
                    score=float(sv.score),
                    coverage=float(sv.coverage),
                    support_weight=float(sv.support_weight),
                    contradict_weight=float(sv.contradict_weight),
                    mixed_weight=float(sv.mixed_weight),
                    link_count=sv.link_count,
                    rationale=sv.rationale,
                    aggregation_version=sv.aggregation_version,
                    computed_at=sv.computed_at,
                ),
                evidence=evidence_by_subclaim.get(sub.id, []),
            )
        )

    sources = sorted(sources_by_id.values(), key=lambda s: (s.domain, str(s.id)))

    # --- explanation ------------------------------------------------------
    counts = dict(verdict_row.subclaim_counts or {})
    explanation = ReportExplanation(
        verdict_statement=build_verdict_statement(
            verdict_row.verdict,
            float(verdict_row.score),
            float(verdict_row.coverage),
            counts,
        ),
        subclaim_statements=build_subclaim_statements(report_subclaims),
        supporting_points=build_points(report_subclaims, "SUPPORTS"),
        contradicting_points=build_points(report_subclaims, "CONTRADICTS"),
    )

    tokens_in = tokens_out = 0
    if use_llm and llm is not None:
        narrative, tokens_in, tokens_out = generate_narrative(
            llm,
            claim_text=claim.raw_text,
            verdict=verdict_row.verdict,
            score=float(verdict_row.score),
            coverage=float(verdict_row.coverage),
            subclaims=report_subclaims,
        )
        if narrative:
            explanation.narrative = narrative
            explanation.narrative_source = "model"

    limitations = collect_limitations(
        verdict_row=verdict_row,
        subclaims=report_subclaims,
        sources=sources,
        evidence=all_evidence,
    )

    report = ClaimReport(
        claim=ReportClaim(
            id=claim.id,
            raw_text=claim.raw_text,
            domain=claim.domain,
            status=claim.status,
            created_at=claim.created_at,
            completed_at=claim.completed_at,
        ),
        verdict=ReportVerdict(
            verdict=verdict_row.verdict,
            score=float(verdict_row.score),
            coverage=float(verdict_row.coverage),
            contradicted_weight_ratio=float(verdict_row.contradicted_weight_ratio),
            subclaim_counts=counts,
            summary=verdict_row.summary,
            aggregation_version=verdict_row.aggregation_version,
            computed_at=verdict_row.computed_at,
        ),
        subclaims=report_subclaims,
        sources=sources,
        explanation=explanation,
        limitations=limitations,
        report_version=REPORT_VERSION,
        generated_at=datetime.now(timezone.utc),
        evidence_link_count=len(all_evidence),
    )

    return report, tokens_in, tokens_out


def mark_report_generated(db: Session, claim_id: uuid.UUID) -> None:
    """Stamp ``verdicts.report_generated_at``.

    This is the only write Stage 6 performs. It records that a report was
    produced; it does not alter any verdict value.
    """
    verdict_row = db.scalars(
        select(Verdict).where(Verdict.claim_id == claim_id)
    ).one_or_none()
    if verdict_row is not None:
        verdict_row.report_generated_at = datetime.now(timezone.utc)
        db.flush()


def record_stage6_run(
    db: Session,
    claim_id: uuid.UUID,
    *,
    status: str,
    error: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    attempt: int | None = None,
) -> Run:
    """Write the Stage 6 ``runs`` row.

    Token counts are zero unless the optional narrative call actually ran.
    Cost stays zero: the free tier reports no price this code can trust.

    ``attempt`` defaults to one more than the highest attempt already recorded
    for this claim and stage. Stage 6 is safe to re-run, and
    ``UNIQUE(claim_id, stage, attempt)`` means a fixed attempt number would
    collide on the second run. Each generation is its own attempt, which is also
    the honest audit trail: two reports were produced, at two times.
    """
    if attempt is None:
        highest = db.scalar(
            select(func.max(Run.attempt)).where(
                Run.claim_id == claim_id, Run.stage == STAGE_REPORT
            )
        )
        attempt = (highest or 0) + 1

    run = Run(
        claim_id=claim_id,
        stage=STAGE_REPORT,
        attempt=attempt,
        status=status,
        finished_at=datetime.now(timezone.utc),
        error=error[:2000] if error else None,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=Decimal("0"),
        cache_hit=False,
    )
    db.add(run)
    return run


__all__ = [
    "NARRATIVE_JSON_SCHEMA",
    "PROMPT_VERSION",
    "REPORT_VERSION",
    "STAGE_REPORT",
    "NarrativeResponse",
    "Stage6Error",
    "Stage6InputError",
    "build_points",
    "build_report",
    "build_subclaim_statements",
    "build_verdict_statement",
    "collect_limitations",
    "generate_narrative",
    "load_prompt",
    "mark_report_generated",
    "record_stage6_run",
]

