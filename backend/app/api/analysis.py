"""Stage 3 trigger endpoint.

Thin wrapper over ``run_stage3``. No analysis logic lives here.

``POST /api/claims/{id}/analyze`` classifies every candidate passage against its
subclaim and persists ``evidence_links``. It produces no verdicts: those are
Stage 4.
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
from app.pipeline.stage3_analyze import (
    Stage3Error,
    Stage3InputError,
    Stage3Result,
    record_stage3_run,
    run_stage3,
)
from app.services.llm import LLMNotConfiguredError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["analysis"])


class SubclaimAnalysisResponse(BaseModel):
    subclaim_id: uuid.UUID
    subclaim_text: str
    passages_analyzed: int
    links_created: int
    links_updated: int
    quotes_failed: int
    missing_judgments: int
    relationship_counts: dict[str, int] = Field(default_factory=dict)


class Stage3Response(BaseModel):
    """Summary of a Stage 3 run. No secrets, no prompt contents."""

    claim_id: uuid.UUID
    links_created: int
    links_updated: int
    total_links: int
    subclaims: list[SubclaimAnalysisResponse] = Field(default_factory=list)


def _to_response(result: Stage3Result) -> Stage3Response:
    return Stage3Response(
        claim_id=result.claim_id,
        links_created=result.links_created,
        links_updated=result.links_updated,
        total_links=result.total_links,
        subclaims=[
            SubclaimAnalysisResponse(
                subclaim_id=s.subclaim_id,
                subclaim_text=s.subclaim_text,
                passages_analyzed=s.passages_analyzed,
                links_created=s.links_created,
                links_updated=s.links_updated,
                quotes_failed=s.quotes_failed,
                missing_judgments=s.missing_judgments,
                relationship_counts=s.relationship_counts,
            )
            for s in result.subclaims
        ],
    )


@router.post(
    "/{claim_id}/analyze",
    response_model=Stage3Response,
    summary="Run Stage 3 evidence analysis for a claim",
)
def analyze_evidence(
    claim_id: uuid.UUID, db: Session = Depends(get_db)
) -> Stage3Response:
    """Classify every candidate passage against its subclaim.

    Returns 404 if the claim does not exist, 409 if Stage 1 or Stage 2 has not
    produced subclaims or passages, 503 if the LLM is not configured, 502 if the
    model failed or its output could not be validated.
    """
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found"
        )

    try:
        result = run_stage3(claim_id, db)
    except Stage3InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except LLMNotConfiguredError as exc:
        record_stage3_run(db, claim_id, status="failed", error="LLM not configured")
        db.commit()
        logger.error("Stage 3 could not run: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not configured. Set GEMINI_API_KEY.",
        ) from exc
    except Stage3Error as exc:
        db.rollback()
        record_stage3_run(db, claim_id, status="failed", error=str(exc))
        db.commit()
        logger.error("Stage 3 failed for claim %s: %s", claim_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Evidence analysis failed.",
        ) from exc

    record_stage3_run(db, claim_id, status="complete", result=result)
    db.commit()

    return _to_response(result)
