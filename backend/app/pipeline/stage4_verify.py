"""Stage 4 — Claim Verification (deterministic aggregation, ``agg-1.0.0``).

Consumes subclaims, Stage 3 evidence links and Stage 5 source scores, and
produces one ``subclaim_verdicts`` row per subclaim plus one ``verdicts`` row per
claim.

**No LLM is imported by this module.** That is structural, not a convention: the
verdict must be reproducible from stored rows alone, byte for byte. If this file
ever needs a model call, the design has failed.

The algorithm is a total function: every input produces exactly one verdict.
There is no "unknown" branch and no path that raises because the evidence is
awkward.

Frozen constants (``agg-1.0.0``) — do not tune these to make a test pass::

    MIN_LINK_WEIGHT        = 0.15
    MIN_SUBCLAIM_EVIDENCE  = 0.25
    DOMINANCE_MARGIN       = 0.30
    COVERAGE_FLOOR         = 0.50
    VERDICT_THRESHOLD      = 0.60
    MISLEADING_SCORE_FLOOR = 0.20
    MISLEADING_MATERIALITY = 0.20
    ROUND_DP               = 4
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
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

logger = logging.getLogger(__name__)

AGGREGATION_VERSION = "agg-1.0.0"
STAGE_VERIFY = 4

# --- Frozen constants ------------------------------------------------------
MIN_LINK_WEIGHT = 0.15
MIN_SUBCLAIM_EVIDENCE = 0.25
DOMINANCE_MARGIN = 0.30
COVERAGE_FLOOR = 0.50
VERDICT_THRESHOLD = 0.60
MISLEADING_SCORE_FLOOR = 0.20
MISLEADING_MATERIALITY = 0.20
ROUND_DP = 4

RELATIONSHIPS = ("SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT")

SUB_VERDICTS = ("SUPPORTED", "CONTRADICTED", "MIXED", "INSUFFICIENT")
OVERALL_VERDICTS = (
    "SUPPORTED",
    "CONTRADICTED",
    "MIXED",
    "INSUFFICIENT_EVIDENCE",
    "MISLEADING",
)


class Stage4Error(RuntimeError):
    """Stage 4 could not compute a verdict."""


class Stage4InputError(Stage4Error):
    """The claim has no subclaims to verify."""


def _round(value: float) -> float:
    """Round for persistence and for every comparison.

    All threshold comparisons use rounded values, so a score of 0.59999 and one
    of 0.60001 cannot straddle a boundary by floating-point accident.
    """
    return round(value, ROUND_DP)


# ---------------------------------------------------------------------------
# Pure aggregation — no database, no I/O
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LinkContribution:
    """One evidence link reduced to the numbers aggregation needs."""

    relationship: str
    weight: float
    counted: bool
    reason: str = ""


def link_weight(
    authority: float | None,
    recency: float | None,
    relevance: float,
    evidence_quality: float,
    stance_confidence: float,
) -> float:
    """The five-factor link weight, in [0, 1].

    A missing source score is treated as 0.0, not as a neutral 1.0. An
    unassessed source must not receive artificial authority: a NULL authority
    means Stage 5 has not run, and silently assuming "average" would let
    unassessed evidence move a verdict. Zero weight means the link falls below
    ``MIN_LINK_WEIGHT`` and is excluded, which is the safe direction.
    """
    a = 0.0 if authority is None else float(authority)
    r = 0.0 if recency is None else float(recency)
    return a * r * float(relevance) * float(evidence_quality) * float(stance_confidence)


def build_contributions(
    links: list[tuple[str, float | None, float | None, float, float, float]],
) -> list[LinkContribution]:
    """Reduce raw links to contributions, applying the minimum-weight rule.

    Args:
        links: tuples of ``(relationship, authority, recency, relevance,
            evidence_quality, stance_confidence)``.

    Returns:
        One LinkContribution per input link, in input order.
    """
    contributions: list[LinkContribution] = []

    for relationship, authority, recency, relevance, quality, stance in links:
        if relationship not in RELATIONSHIPS:
            raise Stage4Error(f"invalid stored relationship: {relationship!r}")

        weight = link_weight(authority, recency, relevance, quality, stance)

        if weight < MIN_LINK_WEIGHT:
            contributions.append(
                LinkContribution(
                    relationship=relationship,
                    weight=weight,
                    counted=False,
                    reason=f"weight {_round(weight)} below MIN_LINK_WEIGHT {MIN_LINK_WEIGHT}",
                )
            )
            continue

        if relationship == "INSUFFICIENT":
            # Stored, displayed, never moves a verdict.
            contributions.append(
                LinkContribution(
                    relationship=relationship,
                    weight=weight,
                    counted=False,
                    reason="INSUFFICIENT links never contribute weight",
                )
            )
            continue

        contributions.append(
            LinkContribution(relationship=relationship, weight=weight, counted=True)
        )

    return contributions


@dataclass(slots=True)
class SubclaimAggregate:
    """Deterministic result for one subclaim."""

    subclaim_id: uuid.UUID
    ordinal: int
    text: str
    weight: float
    support_weight: float
    contradict_weight: float
    mixed_weight: float
    total_weight: float
    link_count: int
    verdict: str
    score: float
    coverage: float
    rationale: str


def aggregate_subclaim(
    *,
    subclaim_id: uuid.UUID,
    ordinal: int,
    text: str,
    weight: float,
    contributions: list[LinkContribution],
) -> SubclaimAggregate:
    """Compute one subclaim's verdict from its link contributions.

    Verdict order (fixed, total):

    1. ``T < MIN_SUBCLAIM_EVIDENCE``          -> INSUFFICIENT
    2. ``C == 0 and M == 0``                  -> SUPPORTED
    3. ``S == 0 and M == 0``                  -> CONTRADICTED
    4. ``M > 0`` with support or contradiction -> MIXED
    5. support and contradiction both present -> dominance margin decides
    6. otherwise                              -> INSUFFICIENT

    The dominance test is written as ``(S - C) / T >= DOMINANCE_MARGIN``. With
    ``M == 0`` this is algebraically identical to ``S / (S + C) >= 0.5 +
    DOMINANCE_MARGIN / 2``; the ``/T`` form is used because it stays defined
    when mixed weight is present.
    """
    support = sum(c.weight for c in contributions if c.counted and c.relationship == "SUPPORTS")
    contradict = sum(c.weight for c in contributions if c.counted and c.relationship == "CONTRADICTS")
    mixed = sum(c.weight for c in contributions if c.counted and c.relationship == "MIXED")

    total = support + contradict + mixed
    link_count = sum(1 for c in contributions if c.counted)

    score = (support - contradict) / total if total > 0 else 0.0
    score = max(-1.0, min(1.0, score))
    coverage = min(1.0, total)

    # --- verdict ----------------------------------------------------------
    if total < MIN_SUBCLAIM_EVIDENCE:
        verdict = "INSUFFICIENT"
        rationale = (
            f"Total contributing evidence weight {_round(total)} is below the "
            f"minimum {MIN_SUBCLAIM_EVIDENCE}; no verdict can be reached."
        )
    elif contradict == 0 and mixed == 0:
        verdict = "SUPPORTED"
        rationale = (
            f"All {link_count} contributing link(s) support the subclaim "
            f"(support weight {_round(support)})."
        )
    elif support == 0 and mixed == 0:
        verdict = "CONTRADICTED"
        rationale = (
            f"All {link_count} contributing link(s) contradict the subclaim "
            f"(contradict weight {_round(contradict)})."
        )
    elif mixed > 0 and (support > 0 or contradict > 0):
        verdict = "MIXED"
        rationale = (
            f"Evidence points in more than one direction: support "
            f"{_round(support)}, contradict {_round(contradict)}, mixed "
            f"{_round(mixed)}."
        )
    elif support > 0 and contradict > 0:
        margin = (support - contradict) / total
        if _round(margin) >= DOMINANCE_MARGIN:
            verdict = "SUPPORTED"
            rationale = (
                f"Support outweighs contradiction by {_round(margin)} of total "
                f"weight, at or above the dominance margin {DOMINANCE_MARGIN}."
            )
        elif _round(-margin) >= DOMINANCE_MARGIN:
            verdict = "CONTRADICTED"
            rationale = (
                f"Contradiction outweighs support by {_round(-margin)} of total "
                f"weight, at or above the dominance margin {DOMINANCE_MARGIN}."
            )
        else:
            verdict = "MIXED"
            rationale = (
                f"Support {_round(support)} and contradiction "
                f"{_round(contradict)} are within the dominance margin "
                f"{DOMINANCE_MARGIN}; neither side prevails."
            )
    else:
        verdict = "INSUFFICIENT"
        rationale = "No contributing evidence established a direction."

    return SubclaimAggregate(
        subclaim_id=subclaim_id,
        ordinal=ordinal,
        text=text,
        weight=weight,
        support_weight=_round(support),
        contradict_weight=_round(contradict),
        mixed_weight=_round(mixed),
        total_weight=_round(total),
        link_count=link_count,
        verdict=verdict,
        score=_round(score),
        coverage=_round(coverage),
        rationale=rationale,
    )


@dataclass(slots=True)
class OverallAggregate:
    """Deterministic overall verdict for a claim."""

    verdict: str
    score: float
    coverage: float
    contradicted_weight_ratio: float
    subclaim_counts: dict[str, int]
    summary: str
    limitations: list[str] = field(default_factory=list)


def aggregate_overall(subclaims: list[SubclaimAggregate]) -> OverallAggregate:
    """Compute the claim-level verdict from subclaim aggregates.

    Decision order is fixed and must not be reordered:

    1. ``coverage < COVERAGE_FLOOR``                     -> INSUFFICIENT_EVIDENCE
    2. ``score > MISLEADING_SCORE_FLOOR`` and ratio >=
       ``MISLEADING_MATERIALITY``                        -> MISLEADING
    3. ``score >= VERDICT_THRESHOLD``                    -> SUPPORTED
    4. ``score <= -VERDICT_THRESHOLD``                   -> CONTRADICTED
    5. otherwise                                         -> MIXED
    """
    if not subclaims:
        return OverallAggregate(
            verdict="INSUFFICIENT_EVIDENCE",
            score=0.0,
            coverage=0.0,
            contradicted_weight_ratio=0.0,
            subclaim_counts={},
            summary="No subclaims were available to verify.",
            limitations=["No subclaims were available to verify."],
        )

    total_weight = sum(s.weight for s in subclaims)
    covered = [s for s in subclaims if s.verdict != "INSUFFICIENT"]
    covered_weight = sum(s.weight for s in covered)

    coverage = covered_weight / total_weight if total_weight > 0 else 0.0

    if covered_weight > 0:
        score = sum(s.weight * s.score for s in covered) / covered_weight
    else:
        score = 0.0
    score = max(-1.0, min(1.0, score))

    contradicted_weight = sum(
        s.weight for s in covered if s.verdict == "CONTRADICTED"
    )
    ratio = contradicted_weight / covered_weight if covered_weight > 0 else 0.0

    counts: dict[str, int] = {}
    for s in subclaims:
        counts[s.verdict] = counts.get(s.verdict, 0) + 1

    r_score = _round(score)
    r_coverage = _round(coverage)
    r_ratio = _round(ratio)

    limitations: list[str] = []
    insufficient = [s for s in subclaims if s.verdict == "INSUFFICIENT"]
    if insufficient:
        limitations.append(
            f"{len(insufficient)} of {len(subclaims)} subclaim(s) lacked adequate "
            "evidence and were excluded from the aggregate score."
        )

    # --- verdict ----------------------------------------------------------
    if r_coverage < COVERAGE_FLOOR:
        verdict = "INSUFFICIENT_EVIDENCE"
        summary = (
            f"Only {r_coverage} of subclaim weight is sufficiently evidenced, "
            f"below the coverage floor {COVERAGE_FLOOR}. No overall verdict is "
            "asserted."
        )
        limitations.append(
            f"Evidence coverage {r_coverage} is below the floor {COVERAGE_FLOOR}."
        )
    elif r_score > MISLEADING_SCORE_FLOOR and r_ratio >= MISLEADING_MATERIALITY:
        verdict = "MISLEADING"
        summary = (
            f"Overall score {r_score} is positive but "
            f"{r_ratio} of evidenced subclaim weight is contradicted, at or above "
            f"the materiality threshold {MISLEADING_MATERIALITY}. The claim is "
            "partly supported and materially overstated."
        )
    elif r_score >= VERDICT_THRESHOLD:
        verdict = "SUPPORTED"
        summary = (
            f"Overall score {r_score} meets the support threshold "
            f"{VERDICT_THRESHOLD} with coverage {r_coverage}."
        )
    elif r_score <= -VERDICT_THRESHOLD:
        verdict = "CONTRADICTED"
        summary = (
            f"Overall score {r_score} meets the contradiction threshold "
            f"-{VERDICT_THRESHOLD} with coverage {r_coverage}."
        )
    else:
        verdict = "MIXED"
        summary = (
            f"Overall score {r_score} falls between the support and contradiction "
            f"thresholds with coverage {r_coverage}; evidence does not settle the "
            "claim either way."
        )

    return OverallAggregate(
        verdict=verdict,
        score=r_score,
        coverage=r_coverage,
        contradicted_weight_ratio=r_ratio,
        subclaim_counts=counts,
        summary=summary,
        limitations=limitations,
    )


# ---------------------------------------------------------------------------
# Database-facing stage
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Stage4Result:
    """Outcome of a Stage 4 run."""

    claim_id: uuid.UUID
    subclaims: list[SubclaimAggregate]
    overall: OverallAggregate
    links_total: int = 0
    links_counted: int = 0
    links_below_threshold: int = 0
    links_insufficient: int = 0


def _load_links(
    db: Session, subclaim_id: uuid.UUID
) -> list[tuple[str, float | None, float | None, float, float, float]]:
    """Load one subclaim's links joined through passages to sources."""
    rows = db.execute(
        select(
            EvidenceLink.relationship,
            Source.authority_score,
            Source.recency_score,
            EvidenceLink.relevance_score,
            EvidenceLink.evidence_quality_score,
            EvidenceLink.stance_confidence,
        )
        .join(Passage, Passage.id == EvidenceLink.passage_id)
        .join(Source, Source.id == Passage.source_id)
        .where(EvidenceLink.subclaim_id == subclaim_id)
        .order_by(EvidenceLink.id)
    ).all()

    return [
        (
            relationship,
            None if authority is None else float(authority),
            None if recency is None else float(recency),
            float(relevance),
            float(quality),
            float(stance),
        )
        for relationship, authority, recency, relevance, quality, stance in rows
    ]


def run_stage4(claim_id: uuid.UUID, db: Session) -> Stage4Result:
    """Verify a claim deterministically.

    Args:
        claim_id: the claim to verify.
        db: an open session; the caller owns the transaction.

    Returns:
        Stage4Result with per-subclaim and overall aggregates.

    Raises:
        Stage4Error: the claim does not exist, or stored data is invalid.
        Stage4InputError: the claim has no subclaims.
    """
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise Stage4Error(f"claim {claim_id} not found")

    subclaims = db.scalars(
        select(Subclaim)
        .where(Subclaim.claim_id == claim_id)
        .order_by(Subclaim.ordinal)
    ).all()

    if not subclaims:
        raise Stage4InputError(
            f"claim {claim_id} has no subclaims; run Stage 1 first"
        )

    aggregates: list[SubclaimAggregate] = []
    links_total = links_counted = below = insufficient = 0

    for subclaim in subclaims:
        raw_links = _load_links(db, subclaim.id)
        contributions = build_contributions(raw_links)

        links_total += len(contributions)
        links_counted += sum(1 for c in contributions if c.counted)
        insufficient += sum(
            1 for c in contributions if c.relationship == "INSUFFICIENT"
        )
        below += sum(
            1
            for c in contributions
            if not c.counted and c.relationship != "INSUFFICIENT"
        )

        aggregates.append(
            aggregate_subclaim(
                subclaim_id=subclaim.id,
                ordinal=subclaim.ordinal,
                text=subclaim.text,
                weight=float(subclaim.weight),
                contributions=contributions,
            )
        )

    overall = aggregate_overall(aggregates)

    if links_total > 0 and links_counted == 0:
        overall.limitations.append(
            "No evidence link met the minimum weight; all were excluded from "
            "aggregation."
        )

    return Stage4Result(
        claim_id=claim_id,
        subclaims=aggregates,
        overall=overall,
        links_total=links_total,
        links_counted=links_counted,
        links_below_threshold=below,
        links_insufficient=insufficient,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_stage4(db: Session, result: Stage4Result) -> None:
    """Write subclaim verdicts and the overall verdict, idempotently.

    Existing rows are updated in place rather than duplicated, respecting
    ``UNIQUE(subclaim_id)`` and ``UNIQUE(claim_id)``.
    """
    now = datetime.now(timezone.utc)

    for agg in result.subclaims:
        existing = db.scalars(
            select(SubclaimVerdict).where(
                SubclaimVerdict.subclaim_id == agg.subclaim_id
            )
        ).one_or_none()

        row = existing or SubclaimVerdict(
            subclaim_id=agg.subclaim_id,
            verdict=agg.verdict,
            score=Decimal("0"),
            coverage=Decimal("0"),
            aggregation_version=AGGREGATION_VERSION,
        )

        row.verdict = agg.verdict
        row.score = Decimal(str(agg.score))
        row.coverage = Decimal(str(agg.coverage))
        row.support_weight = Decimal(str(agg.support_weight))
        row.contradict_weight = Decimal(str(agg.contradict_weight))
        row.mixed_weight = Decimal(str(agg.mixed_weight))
        row.link_count = agg.link_count
        row.rationale = agg.rationale
        row.aggregation_version = AGGREGATION_VERSION
        row.computed_at = now

        if existing is None:
            db.add(row)

    overall = result.overall
    existing_verdict = db.scalars(
        select(Verdict).where(Verdict.claim_id == result.claim_id)
    ).one_or_none()

    verdict_row = existing_verdict or Verdict(
        claim_id=result.claim_id,
        verdict=overall.verdict,
        score=Decimal("0"),
        coverage=Decimal("0"),
        aggregation_version=AGGREGATION_VERSION,
    )

    verdict_row.verdict = overall.verdict
    verdict_row.score = Decimal(str(overall.score))
    verdict_row.coverage = Decimal(str(overall.coverage))
    verdict_row.contradicted_weight_ratio = Decimal(
        str(overall.contradicted_weight_ratio)
    )
    verdict_row.subclaim_counts = overall.subclaim_counts
    verdict_row.summary = overall.summary
    verdict_row.limitations = overall.limitations
    verdict_row.aggregation_version = AGGREGATION_VERSION
    verdict_row.computed_at = now
    # Left NULL: Stage 6 sets this when it actually generates the report.
    if existing_verdict is None:
        verdict_row.report_generated_at = None
        db.add(verdict_row)

    db.flush()


def record_stage4_run(
    db: Session,
    claim_id: uuid.UUID,
    *,
    status: str,
    error: str | None = None,
    attempt: int = 1,
) -> Run:
    """Write the Stage 4 ``runs`` row.

    Stage 4 is deterministic and makes no model calls, so token and cost fields
    are zero by construction rather than by measurement.
    """
    run = Run(
        claim_id=claim_id,
        stage=STAGE_VERIFY,
        attempt=attempt,
        status=status,
        finished_at=datetime.now(timezone.utc),
        error=error[:2000] if error else None,
        tokens_in=0,
        tokens_out=0,
        cost_usd=Decimal("0"),
        cache_hit=False,
    )
    db.add(run)
    return run


__all__ = [
    "AGGREGATION_VERSION",
    "COVERAGE_FLOOR",
    "DOMINANCE_MARGIN",
    "MIN_LINK_WEIGHT",
    "MIN_SUBCLAIM_EVIDENCE",
    "MISLEADING_MATERIALITY",
    "MISLEADING_SCORE_FLOOR",
    "OVERALL_VERDICTS",
    "ROUND_DP",
    "STAGE_VERIFY",
    "SUB_VERDICTS",
    "VERDICT_THRESHOLD",
    "LinkContribution",
    "OverallAggregate",
    "Stage4Error",
    "Stage4InputError",
    "Stage4Result",
    "SubclaimAggregate",
    "aggregate_overall",
    "aggregate_subclaim",
    "build_contributions",
    "link_weight",
    "persist_stage4",
    "record_stage4_run",
    "run_stage4",
]
