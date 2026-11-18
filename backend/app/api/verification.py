"""Stage 4 trigger endpoint.

Thin wrapper over ``run_stage4`` and ``persist_stage4``. No aggregation logic
lives here.

``POST /api/claims/{id}/verify`` computes subclaim verdicts and the overall
verdict deterministically. No model is called.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Claim
from app.db.session import get_db
from app.pipeline.stage4_verify import (
    AGGREGATION_VERSION,
    Stage4Error,
    Stage4InputError,
    Stage4Result,
    persist_stage4,
    record_stage4_run,
    run_stage4,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["verification"])


class SubclaimVerdictResponse(BaseModel):
    subclaim_id: uuid.UUID
    ordinal: int
    verdict: str
    score: float
    coverage: float
    support_weight: float
    contradict_weight: float
    mixed_weight: float
    link_count: int
    rationale: str


class OverallVerdictResponse(BaseModel):
    verdict: str
    score: float
    coverage: float
    contradicted_weight_ratio: float
    subclaim_counts: dict[str, int] = Field(default_factory=dict)
    summary: str
    limitations: list[str] = Field(default_factory=list)


class Stage4Response(BaseModel):
    claim_id: uuid.UUID
    aggregation_version: str
    links_total: int
    links_counted: int
    links_below_threshold: int
    links_insufficient: int
    subclaims: list[SubclaimVerdictResponse] = Field(default_factory=list)
    overall: OverallVerdictResponse


def _to_response(result: Stage4Result) -> Stage4Response:
    return Stage4Response(
        claim_id=result.claim_id,
        aggregation_version=AGGREGATION_VERSION,
        links_total=result.links_total,
        links_counted=result.links_counted,
        links_below_threshold=result.links_below_threshold,
        links_insufficient=result.links_insufficient,
        subclaims=[
            SubclaimVerdictResponse(
                subclaim_id=s.subclaim_id,
                ordinal=s.ordinal,
                verdict=s.verdict,
                score=s.score,
                coverage=s.coverage,
                support_weight=s.support_weight,
                contradict_weight=s.contradict_weight,
                mixed_weight=s.mixed_weight,
                link_count=s.link_count,
                rationale=s.rationale,
            )
            for s in result.subclaims
        ],
        overall=OverallVerdictResponse(
            verdict=result.overall.verdict,
            score=result.overall.score,
            coverage=result.overall.coverage,
            contradicted_weight_ratio=result.overall.contradicted_weight_ratio,
            subclaim_counts=result.overall.subclaim_counts,
            summary=result.overall.summary,
            limitations=result.overall.limitations,
        ),
    )


@router.post(
    "/{claim_id}/verify",
    response_model=Stage4Response,
    summary="Run Stage 4 deterministic verification for a claim",
)
def verify_claim(
    claim_id: uuid.UUID, db: Session = Depends(get_db)
) -> Stage4Response:
    """Compute subclaim and overall verdicts.

    Returns 404 if the claim does not exist, 409 if Stage 1 has not produced
    subclaims, 500 if stored data is invalid.

    A claim with no evidence links is not an error: every subclaim resolves to
    INSUFFICIENT and the claim resolves to INSUFFICIENT_EVIDENCE. That is the
    required abstention path, not a failure.
    """
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found"
        )

    try:
        result = run_stage4(claim_id, db)
        persist_stage4(db, result)
    except Stage4InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except Stage4Error as exc:
        db.rollback()
        record_stage4_run(db, claim_id, status="failed", error=str(exc))
        db.commit()
        logger.error("Stage 4 failed for claim %s: %s", claim_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed.",
        ) from exc

    record_stage4_run(db, claim_id, status="complete")
    db.commit()

    return _to_response(result)
