"""Manual smoke test for Stage 3 against the real Gemini API.

Excluded from the default run; needs a live key and spends quota.

    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke -q -m smoke

Seeds a claim with the canonical subclaims and real passage text, runs Stage 3
against Gemini, and asserts the persisted evidence links are well formed: valid
relationships, verbatim quotes, scores in range, and no verdicts of any kind.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import (
    Claim,
    EvidenceLink,
    Passage,
    Source,
    Subclaim,
    SubclaimVerdict,
    Verdict,
)
from app.db.session import SessionLocal
from app.pipeline.stage3_analyze import PROMPT_VERSION, record_stage3_run, run_stage3

pytestmark = pytest.mark.smoke

SUBCLAIMS = [
    "HTTPS encrypts data in transit between the browser and the server.",
    "HTTPS makes a website completely secure against all attacks.",
]

PASSAGES = [
    (
        "HTTPS encrypts traffic between the browser and the server using TLS. "
        "An observer on the network cannot read the request or the response."
    ),
    (
        "HTTPS does not prevent cross-site scripting or SQL injection. A site "
        "can serve malicious content over a perfectly valid TLS connection."
    ),
    (
        "Transport encryption protects data while it is moving between two "
        "parties. It provides confidentiality and integrity for that channel only."
    ),
]


def _require_key() -> None:
    settings = get_settings()
    if not (settings.gemini_api_key or os.getenv("GEMINI_API_KEY")):
        pytest.skip("GEMINI_API_KEY is not set; skipping live Stage 3 smoke test")


def _seed(db) -> Claim:
    claim = Claim(
        raw_text="Using HTTPS makes a website completely secure.",
        domain="technology",
        status="queued",
    )
    db.add(claim)
    db.flush()

    for index, text in enumerate(SUBCLAIMS, start=1):
        db.add(
            Subclaim(
                claim_id=claim.id, ordinal=index, text=text, claim_type="factual"
            )
        )
    db.flush()

    source = Source(
        claim_id=claim.id,
        url="https://example.org/https-explained",
        url_hash="c" * 64,
        domain="example.org",
        title="How HTTPS works",
        content_hash="d" * 64,
    )
    db.add(source)
    db.flush()

    for index, text in enumerate(PASSAGES):
        db.add(
            Passage(
                source_id=source.id,
                text=text,
                chunk_index=index,
                quote_start=0,
                quote_end=len(text),
                token_count=max(1, len(text) // 4),
            )
        )
    db.flush()
    return claim


def test_real_gemini_stage3() -> None:
    """Stage 3 end to end against live Gemini."""
    _require_key()

    db = SessionLocal()
    claim_id: uuid.UUID | None = None
    try:
        claim = _seed(db)
        claim_id = claim.id
        db.commit()

        result = run_stage3(claim_id, db)
        record_stage3_run(db, claim_id, status="complete", result=result)
        db.commit()

        print(f"\nlinks created: {result.links_created}")
        print(f"tokens: in={result.tokens_in} out={result.tokens_out}")
        for summary in result.subclaims:
            print(
                f"  {summary.subclaim_text[:55]!r}: "
                f"{summary.relationship_counts} quotes_failed={summary.quotes_failed}"
            )

        links = db.scalars(select(EvidenceLink)).all()
        assert links, "no evidence links persisted"

        passages_by_id = {
            p.id: p for p in db.scalars(select(Passage)).all()
        }

        valid = {"SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT"}
        for link in links:
            assert link.relationship in valid
            assert 0 <= float(link.relevance_score) <= 1
            assert 0 <= float(link.evidence_quality_score) <= 1
            assert 0 <= float(link.stance_confidence) <= 1
            assert link.prompt_version == PROMPT_VERSION
            assert link.model

            if link.quote:
                assert link.quote in passages_by_id[link.passage_id].text, (
                    f"quote not a verbatim substring: {link.quote[:60]!r}"
                )

        # Stage 3 must not produce verdicts.
        assert db.scalar(select(func.count()).select_from(SubclaimVerdict)) == 0
        assert db.scalar(select(func.count()).select_from(Verdict)) == 0

        print("\nSMOKE OK")
    finally:
        try:
            leftover = (
                db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
                if claim_id
                else None
            )
            if leftover is not None:
                db.delete(leftover)
                db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
        db.close()


if __name__ == "__main__":
    _require_key()
    test_real_gemini_stage3()
    sys.exit(0)
