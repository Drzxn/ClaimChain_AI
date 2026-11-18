"""Manual smoke test for Stage 5 against the real Gemini API.

Excluded from the default run; needs a live key and spends quota.

    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke -q -m smoke

Verifies that live classification produces valid source types, that the
deterministic scores are still computed by the backend (never by the model), and
that no verdicts are created.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import (
    Claim,
    Passage,
    Source,
    SubclaimVerdict,
    Verdict,
)
from app.db.session import SessionLocal
from app.pipeline.stage5_assess import (
    SOURCE_TYPES,
    record_stage5_run,
    run_stage5,
)

pytestmark = pytest.mark.smoke

REF = date(2026, 11, 18)

SOURCES = [
    ("https://www.nist.gov/topics/cybersecurity", "nist.gov", 30),
    ("https://en.wikipedia.org/wiki/HTTPS", "wikipedia.org", 200),
    ("https://medium.com/some-post-about-https", "medium.com", 900),
]


def _require_key() -> None:
    settings = get_settings()
    if not (settings.gemini_api_key or os.getenv("GEMINI_API_KEY")):
        pytest.skip("GEMINI_API_KEY is not set; skipping live Stage 5 smoke test")


def _seed(db) -> Claim:
    claim = Claim(
        raw_text="Using HTTPS makes a website completely secure.",
        domain="technology",
        status="queued",
    )
    db.add(claim)
    db.flush()

    for index, (url, domain, age_days) in enumerate(SOURCES):
        source = Source(
            claim_id=claim.id,
            url=url,
            url_hash=f"{index:064d}",
            domain=domain,
            title=f"Source {index}",
            published_at=date.fromordinal(REF.toordinal() - age_days),
            retrieved_at=datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc),
            content_hash=f"k{index:063d}",
        )
        db.add(source)
        db.flush()

        db.add(
            Passage(
                source_id=source.id,
                text=(
                    "HTTPS encrypts traffic in transit between the browser and "
                    "the server. This reference summarises the protocol and cites "
                    "the relevant standards documents."
                ),
                chunk_index=0,
                quote_start=0,
                quote_end=150,
                token_count=37,
            )
        )

    db.flush()
    return claim


def test_real_gemini_stage5() -> None:
    """Stage 5 end to end against live Gemini."""
    _require_key()

    db = SessionLocal()
    claim_id: uuid.UUID | None = None
    try:
        claim = _seed(db)
        claim_id = claim.id
        db.commit()

        result = run_stage5(claim_id, db, reference_date=REF)
        record_stage5_run(db, claim_id, status="complete", result=result)
        db.commit()

        print(f"\nassessed: {result.assessed_count}")
        print(f"llm calls: {result.llm_calls}  failures: {result.llm_failures}")
        print(f"tokens: in={result.tokens_in} out={result.tokens_out}")

        sources = db.scalars(select(Source)).all()
        assert sources

        for source in sources:
            print(
                f"  {source.domain:20} auth={source.authority_score} "
                f"rec={source.recency_score} type={source.source_type} "
                f"cites={source.cites_primary_sources}"
            )
            # Deterministic scores must be present and bounded.
            assert source.authority_score is not None
            assert 0 <= float(source.authority_score) <= 1
            assert source.recency_score is not None
            assert 0 <= float(source.recency_score) <= 1
            assert source.source_type in SOURCE_TYPES
            assert source.assessment_rationale
            assert source.assessed_at is not None

        # nist.gov must score 1.0 regardless of what the model said.
        nist = next(s for s in sources if s.domain == "nist.gov")
        assert float(nist.authority_score) == 1.0

        # Stage 5 must not produce verdicts.
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
    test_real_gemini_stage5()
    sys.exit(0)
