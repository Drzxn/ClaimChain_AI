"""Stage 5 trigger endpoint.

Thin wrapper over ``run_stage5``. No scoring logic lives here.

``POST /api/claims/{id}/assess`` scores every source of a claim for authority and
recency, classifies its type, and records the assessment. It produces no
verdicts: those are Stage 4.
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
from app.pipeline.stage5_assess import (
    Stage5Error,
    Stage5InputError,
    Stage5Result,
    record_stage5_run,
    run_stage5,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["assessment"])


class SourceAssessmentResponse(BaseModel):
    source_id: uuid.UUID
    domain: str
    authority_score: float
    recency_score: float
    source_type: str
    cites_primary_sources: bool | None = None
    rationale: str
    used_llm: bool


class Stage5Response(BaseModel):
    """Summary of a Stage 5 run. No secrets, no prompt contents."""

    claim_id: uuid.UUID
    assessed_count: int
    llm_calls: int
    llm_failures: int
    sources: list[SourceAssessmentResponse] = Field(default_factory=list)


def _to_response(result: Stage5Result) -> Stage5Response:
    return Stage5Response(
        claim_id=result.claim_id,
        assessed_count=result.assessed_count,
        llm_calls=result.llm_calls,
        llm_failures=result.llm_failures,
        sources=[
            SourceAssessmentResponse(
                source_id=s.source_id,
                domain=s.domain,
                authority_score=s.authority_score,
                recency_score=s.recency_score,
                source_type=s.source_type,
                cites_primary_sources=s.cites_primary_sources,
                rationale=s.rationale,
                used_llm=s.used_llm,
            )
            for s in result.sources
        ],
    )


@router.post(
    "/{claim_id}/assess",
    response_model=Stage5Response,
    summary="Run Stage 5 source assessment for a claim",
)
def assess_sources(
    claim_id: uuid.UUID, db: Session = Depends(get_db)
) -> Stage5Response:
    """Assess every source of a claim.

    Returns 404 if the claim does not exist, 409 if Stage 2 has not produced any
    sources. LLM failures degrade to deterministic classification rather than
    failing the request, so a provider outage does not block assessment.
    """
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found"
        )

    try:
        result = run_stage5(claim_id, db)
    except Stage5InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except Stage5Error as exc:
        db.rollback()
        record_stage5_run(db, claim_id, status="failed", error=str(exc))
        db.commit()
        logger.error("Stage 5 failed for claim %s: %s", claim_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Source assessment failed.",
        ) from exc

    record_stage5_run(db, claim_id, status="complete", result=result)
    db.commit()

    return _to_response(result)
