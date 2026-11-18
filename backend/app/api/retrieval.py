"""Stage 2 trigger endpoint.

Deliberately minimal and scoped: it exists so Stage 2 can be exercised and
demonstrated on its own, without pretending the full 1->6 pipeline exists.

``POST /api/claims/{id}/retrieve`` runs Stage 2 for an already-decomposed
claim, persists sources and passages, records the run, and returns a summary.

It does NOT create evidence links, assess sources, or produce a verdict. Those
are later stages.
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
from app.pipeline.stage2_retrieve import (
    Stage2Error,
    Stage2InputError,
    Stage2Result,
    record_stage2_run,
    run_stage2,
)
from app.services.exa import SearchNotConfiguredError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["retrieval"])


class SkippedUrlResponse(BaseModel):
    url: str
    reason: str


class SubclaimCandidatesResponse(BaseModel):
    subclaim_id: uuid.UUID
    subclaim_text: str
    passage_count: int


class Stage2Response(BaseModel):
    """Summary of a Stage 2 run. No secrets, no prompt contents."""

    claim_id: uuid.UUID
    sources_created: int
    passages_created: int
    duplicates_linked: int
    urls_considered: int
    urls_skipped: list[SkippedUrlResponse] = Field(default_factory=list)
    candidates: list[SubclaimCandidatesResponse] = Field(default_factory=list)
    fixture_mode: bool


def _to_response(result: Stage2Result) -> Stage2Response:
    return Stage2Response(
        claim_id=result.claim_id,
        sources_created=result.sources_created,
        passages_created=result.passages_created,
        duplicates_linked=result.duplicates_linked,
        urls_considered=result.urls_considered,
        urls_skipped=[
            SkippedUrlResponse(url=s.url, reason=s.reason) for s in result.urls_skipped
        ],
        candidates=[
            SubclaimCandidatesResponse(
                subclaim_id=c.subclaim_id,
                subclaim_text=c.subclaim_text,
                passage_count=len(c.passage_ids),
            )
            for c in result.candidates
        ],
        fixture_mode=result.fixture_mode,
    )


@router.post(
    "/{claim_id}/retrieve",
    response_model=Stage2Response,
    summary="Run Stage 2 evidence collection for a claim",
)
def retrieve_evidence(
    claim_id: uuid.UUID, db: Session = Depends(get_db)
) -> Stage2Response:
    """Collect evidence for every subclaim of an already-decomposed claim.

    Returns 404 if the claim does not exist, 409 if it has no subclaims
    (Stage 1 has not run or failed), 503 if no search provider is configured.
    """
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found"
        )

    try:
        result = run_stage2(claim_id, db)
    except Stage2InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except SearchNotConfiguredError as exc:
        record_stage2_run(db, claim_id, status="failed", error="search not configured")
        db.commit()
        logger.error("Stage 2 could not run: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search provider is not configured. Set EXA_API_KEY or enable fixture mode.",
        ) from exc
    except Stage2Error as exc:
        record_stage2_run(db, claim_id, status="failed", error=str(exc))
        db.commit()
        logger.error("Stage 2 failed for claim %s: %s", claim_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Evidence collection failed.",
        ) from exc

    record_stage2_run(db, claim_id, status="complete", result=result)
    db.commit()

    return _to_response(result)
