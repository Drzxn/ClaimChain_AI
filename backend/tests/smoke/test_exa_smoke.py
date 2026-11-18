"""Manual smoke test against the real Exa API.

Excluded from the default run. Needs a live key and spends quota.

    $env:EXA_API_KEY = "..."
    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke -q -m smoke

Verifies the full Stage 2 chain against live search results: search, dedupe,
SSRF rejection, fetch, extract, chunk, persist. It asserts that no evidence
links and no verdicts are created, because Stage 2 must not classify anything.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import Claim, EvidenceLink, Passage, Source, Subclaim, Verdict
from app.db.session import SessionLocal
from app.pipeline.stage2_retrieve import record_stage2_run, run_stage2
from app.services.exa import ExaSearchService

CANONICAL = "Using HTTPS makes a website completely secure."

pytestmark = pytest.mark.smoke


def _require_key() -> None:
    settings = get_settings()
    if not (settings.exa_api_key or os.getenv("EXA_API_KEY")):
        pytest.skip("EXA_API_KEY is not set; skipping live Exa smoke test")


def _seed_claim(db) -> Claim:
    claim = Claim(raw_text=CANONICAL, domain="technology", status="queued")
    db.add(claim)
    db.flush()
    for index, text in enumerate(
        [
            "HTTPS encrypts data in transit between the browser and the server.",
            "HTTPS makes a website completely secure against all attacks.",
        ],
        start=1,
    ):
        db.add(
            Subclaim(
                claim_id=claim.id, ordinal=index, text=text, claim_type="factual"
            )
        )
    db.flush()
    return claim


def test_real_exa_stage2() -> None:
    """Full Stage 2 against live Exa results."""
    _require_key()

    service = ExaSearchService()
    assert service.configured

    # 1. Exa responds at all.
    probe = service.search("https encryption data in transit", num_results=3)
    assert probe, "Exa returned no results for a basic query"
    assert all(r.url for r in probe)

    db = SessionLocal()
    claim_id = uuid.uuid4()
    try:
        claim = _seed_claim(db)
        claim_id = claim.id
        db.commit()

        # 2. Stage 2 runs end to end.
        result = run_stage2(claim.id, db, search=service)
        record_stage2_run(db, claim.id, status="complete", result=result)
        db.commit()

        print(f"\nurls considered: {result.urls_considered}")
        print(f"sources created: {result.sources_created}")
        print(f"passages created: {result.passages_created}")
        print(f"duplicates linked: {result.duplicates_linked}")
        print(f"urls skipped: {len(result.urls_skipped)}")
        for skipped in result.urls_skipped[:5]:
            print(f"  skipped {skipped.url}: {skipped.reason}")
        for candidate in result.candidates:
            print(f"  subclaim {candidate.subclaim_text[:50]!r}: {len(candidate.passage_ids)} passages")

        # 3. Sources and passages persisted.
        sources = db.scalars(select(Source).where(Source.claim_id == claim.id)).all()
        assert sources, "no sources persisted from live search"
        assert result.passages_created > 0, "no passages persisted"

        # 4. URL dedupe held.
        hashes = [s.url_hash for s in sources]
        assert len(hashes) == len(set(hashes))

        # 5. Passages are well formed.
        for passage in db.scalars(select(Passage)).all():
            assert passage.text.strip()
            assert passage.quote_end >= passage.quote_start
            assert passage.token_count > 0

        # 6. Stage 2 classified nothing.
        assert db.scalar(select(func.count()).select_from(EvidenceLink)) == 0
        assert db.scalar(select(func.count()).select_from(Verdict)) == 0

        print("\nSMOKE OK")
    finally:
        # Leave the database as found.
        try:
            leftover = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
            if leftover is not None:
                db.delete(leftover)
                db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
        db.close()


if __name__ == "__main__":
    _require_key()
    test_real_exa_stage2()
    sys.exit(0)
