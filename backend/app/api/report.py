"""Stage 6 report endpoints.

Thin wrappers over ``build_report``. No report logic lives here.

``POST /api/claims/{id}/report`` builds the report, stamps
``verdicts.report_generated_at``, records the Stage 6 run, and returns it.
``GET  /api/claims/{id}/report`` returns the same report without writing
anything, for cheap frontend reads.

Both are read-only with respect to verdicts: Stage 4 owns those values.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Claim
from app.db.session import get_db
from app.pipeline.stage6_report import (
    Stage6Error,
    Stage6InputError,
    build_report,
    mark_report_generated,
    record_stage6_run,
)
from app.schemas.report import ClaimReport
from app.services.llm import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["report"])


def _require_claim(db: Session, claim_id: uuid.UUID) -> Claim:
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found"
        )
    return claim


@router.post(
    "/{claim_id}/report",
    response_model=ClaimReport,
    summary="Generate the Stage 6 explainable report for a claim",
)
def generate_report(
    claim_id: uuid.UUID,
    db: Session = Depends(get_db),
    narrative: bool = Query(
        default=False,
        description=(
            "Ask the model for narrative prose. Off by default: the report is "
            "fully deterministic without it. Prose can never change the verdict."
        ),
    ),
) -> ClaimReport:
    """Build and return the report, recording the Stage 6 run.

    Returns 404 if the claim does not exist, 409 if Stage 4 has not produced a
    verdict for it.
    """
    _require_claim(db, claim_id)

    llm = GeminiService() if narrative else None

    try:
        report, tokens_in, tokens_out = build_report(
            claim_id, db, llm=llm, use_llm=narrative
        )
    except Stage6InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except Stage6Error as exc:
        db.rollback()
        record_stage6_run(db, claim_id, status="failed", error=str(exc))
        db.commit()
        logger.error("Stage 6 failed for claim %s: %s", claim_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed.",
        ) from exc

    mark_report_generated(db, claim_id)
    record_stage6_run(
        db, claim_id, status="complete", tokens_in=tokens_in, tokens_out=tokens_out
    )
    db.commit()

    return report


@router.get(
    "/{claim_id}/report",
    response_model=ClaimReport,
    summary="Read the Stage 6 report without recording a run",
)
def read_report(
    claim_id: uuid.UUID, db: Session = Depends(get_db)
) -> ClaimReport:
    """Build and return the report without writing anything.

    The report is a projection of persisted rows, so a GET is safe and always
    reflects current data.
    """
    _require_claim(db, claim_id)

    try:
        report, _, _ = build_report(claim_id, db, use_llm=False)
    except Stage6InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except Stage6Error as exc:
        logger.error("Stage 6 read failed for claim %s: %s", claim_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed.",
        ) from exc

    return report
