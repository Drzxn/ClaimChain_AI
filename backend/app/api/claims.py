"""Claims API — Stage 1 only.

``POST /api/claims`` runs the claim through Stage 1 (decomposition) and returns
the persisted result. ``GET /api/claims/{id}`` reads it back.

Transaction policy
------------------
The claim row, its subclaims and the Stage 1 run rows are written in one
transaction. If Stage 1 fails, the subclaims are never written and the claim is
marked ``failed`` with an error message, in a *separate* committed transaction
so the failure is durable and debuggable.

Concretely:

1. Commit the claim row immediately (status ``queued``) so a crash mid-call
   still leaves a record of what was attempted.
2. Run Stage 1 outside any open transaction. No database locks are held across
   a network call.
3. On success: insert subclaims, update the claim, insert run rows, commit once.
4. On failure: insert the failed run rows, set the claim to ``failed``, commit.

A failed claim therefore keeps its original row and its failed run records, and
has zero subclaims. There is no partial decomposition.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Claim, Run, Subclaim
from app.db.session import get_db
from app.pipeline.stage1_decompose import (
    PROMPT_VERSION,
    Stage1Error,
    Stage1InputError,
    Stage1Result,
    run_stage1,
)
from app.schemas.claim import (
    ClaimResponse,
    CreateClaimRequest,
    CreateClaimResponse,
)
from app.services.llm import (
    GeminiService,
    LLMNotConfiguredError,
    LLMTransportError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["claims"])

# Stage numbers are conceptual labels. Stage 1 is the only stage implemented.
STAGE_DECOMPOSE = 1


def get_llm_service() -> GeminiService:
    """Dependency so tests can override the LLM with a fake."""
    return GeminiService()


def _record_runs(db: Session, claim: Claim, result: Stage1Result, model: str) -> None:
    """Persist one ``runs`` row per Stage 1 attempt."""
    for attempt in result.attempts:
        db.add(
            Run(
                claim_id=claim.id,
                stage=STAGE_DECOMPOSE,
                attempt=attempt.attempt,
                status=attempt.status,
                finished_at=datetime.now(timezone.utc),
                error=attempt.error,
                tokens_in=attempt.tokens_in,
                tokens_out=attempt.tokens_out,
                # Free tier reports no reliable price; 0.0 means "unknown".
                cost_usd=Decimal(str(attempt.cost_usd)),
                cache_hit=attempt.cache_hit,
            )
        )


def _record_failed_run(db: Session, claim: Claim, attempt: int, error: str) -> None:
    """Persist a failed run row when no Stage1Result was produced."""
    db.add(
        Run(
            claim_id=claim.id,
            stage=STAGE_DECOMPOSE,
            attempt=attempt,
            status="failed",
            finished_at=datetime.now(timezone.utc),
            error=error[:2000],
        )
    )


@router.post(
    "",
    response_model=CreateClaimResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a claim for decomposition (Stage 1)",
)
def create_claim(
    payload: CreateClaimRequest,
    db: Session = Depends(get_db),
    llm: GeminiService = Depends(get_llm_service),
) -> CreateClaimResponse:
    """Create a claim and run Stage 1 decomposition.

    Returns 201 only when decomposition succeeded. Any Stage 1 failure returns
    an error status and leaves no subclaims behind.
    """
    # --- 1. persist the claim so the attempt is auditable -------------------
    claim = Claim(
        raw_text=payload.raw_text,
        domain="general",  # provisional; Stage 1 assigns the real domain
        status="queued",
        model_config={
            "model": llm.model,
            "temperature": 0.0,
            "prompt_version": PROMPT_VERSION,
        },
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)

    claim.status = "running"
    db.commit()

    # --- 2. run Stage 1 outside a transaction -------------------------------
    try:
        result = run_stage1(payload.raw_text, service=llm)
    except Stage1InputError as exc:
        # Should already be caught by request validation; belt and braces.
        _fail_claim(db, claim, str(exc), attempt=1)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LLMNotConfiguredError as exc:
        _fail_claim(db, claim, "LLM is not configured", attempt=1)
        logger.error("Stage 1 could not run: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not configured. Set GEMINI_API_KEY.",
        ) from exc
    except LLMTransportError as exc:
        _fail_claim(db, claim, "LLM call failed", attempt=1)
        logger.error("Stage 1 transport failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model could not be reached. Try again.",
        ) from exc
    except Stage1Error as exc:
        # Both the initial attempt and the repair failed validation.
        _fail_claim(db, claim, str(exc), attempt=2)
        logger.warning("Stage 1 failed for claim %s: %s", claim.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model did not return a valid decomposition after one repair retry.",
        ) from exc

    # --- 3. persist the decomposition atomically ---------------------------
    try:
        claim.domain = result.response.domain
        for sub in result.response.subclaims:
            db.add(
                Subclaim(
                    claim_id=claim.id,
                    ordinal=sub.ordinal,
                    text=sub.text,
                    claim_type=sub.claim_type,
                    # Weight is forced to 1.0 by the schema layer.
                    weight=sub.weight,
                    decomposition_note=sub.decomposition_note,
                )
            )
        _record_runs(db, claim, result, model=result.response.model)

        # Stage 1 is done. The claim waits for Stage 2 (evidence collection).
        claim.status = "queued"
        claim.error = None
        db.commit()
    except Exception as exc:  # noqa: BLE001 - never leave a partial write
        db.rollback()
        _fail_claim(db, claim, "failed to persist decomposition", attempt=1)
        logger.exception("Failed to persist Stage 1 result for claim %s", claim.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Decomposition succeeded but could not be saved.",
        ) from exc

    return CreateClaimResponse(
        id=claim.id,
        status=claim.status,
        domain=result.response.domain,
        subclaim_count=len(result.response.subclaims),
    )


def _fail_claim(db: Session, claim: Claim, error: str, attempt: int) -> None:
    """Mark a claim failed and record the failed run, in its own transaction."""
    try:
        db.rollback()
        _record_failed_run(db, claim, attempt=attempt, error=error)
        claim.status = "failed"
        claim.error = error[:2000]
        claim.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:  # noqa: BLE001 - failure reporting must not mask the cause
        db.rollback()
        logger.exception("Could not record failure for claim %s", claim.id)


@router.get(
    "/{claim_id}",
    response_model=ClaimResponse,
    summary="Read a claim and its subclaims",
)
def get_claim(claim_id: uuid.UUID, db: Session = Depends(get_db)) -> ClaimResponse:
    """Return a claim with its subclaims ordered by ordinal."""
    claim = db.scalars(
        select(Claim)
        .where(Claim.id == claim_id)
        .options(selectinload(Claim.subclaims))
    ).one_or_none()

    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found"
        )

    return ClaimResponse.model_validate(claim)
