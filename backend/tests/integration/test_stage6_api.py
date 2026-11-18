"""Stage 6 API tests through the real HTTP layer and test database."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Claim,
    EvidenceLink,
    Passage,
    Run,
    Source,
    Subclaim,
    Verdict,
)
from app.main import create_app
from app.pipeline.stage4_verify import persist_stage4, run_stage4

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def api(engine: Engine):
    """TestClient bound to the test database."""

    def _override_get_db():
        session = Session(bind=engine, expire_on_commit=False)
        try:
            yield session
        finally:
            session.close()

    from app.db.session import get_db

    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_tables(engine: Engine):
    yield
    with Session(bind=engine) as session:
        for claim in session.scalars(select(Claim)).all():
            session.delete(claim)
        session.commit()


def seed(engine: Engine, *, misleading: bool = True) -> uuid.UUID:
    """Seed a verified claim directly, returning its id."""
    with Session(bind=engine) as db:
        claim = Claim(
            raw_text="Using HTTPS makes a website completely secure.",
            domain="technology",
            status="queued",
        )
        db.add(claim)
        db.flush()

        specs = (
            [("A.", "SUPPORTS"), ("B.", "SUPPORTS"), ("C.", "CONTRADICTS")]
            if misleading
            else [("A.", "SUPPORTS"), ("B.", "SUPPORTS")]
        )

        for ordinal, (text, relationship) in enumerate(specs, start=1):
            sub = Subclaim(
                claim_id=claim.id, ordinal=ordinal, text=text, claim_type="factual"
            )
            db.add(sub)
            db.flush()

            source = Source(
                claim_id=claim.id,
                url=f"https://example.org/{ordinal}",
                url_hash=f"{ordinal:064d}",
                domain="example.org",
                title=f"Source {ordinal}",
                source_type="reference",
                published_at=date(2026, 1, 1),
                retrieved_at=NOW,
                content_hash=f"h{ordinal:063d}",
                authority_score=Decimal("1.0"),
                recency_score=Decimal("1.0"),
                assessment_rationale="assessed",
                assessed_at=NOW,
            )
            db.add(source)
            db.flush()

            passage = Passage(
                source_id=source.id,
                text="HTTPS encrypts traffic in transit.",
                chunk_index=0,
                quote_start=0,
                quote_end=32,
                token_count=8,
            )
            db.add(passage)
            db.flush()

            db.add(
                EvidenceLink(
                    subclaim_id=sub.id,
                    passage_id=passage.id,
                    relationship=relationship,
                    relevance_score=Decimal("1.0"),
                    evidence_quality_score=Decimal("1.0"),
                    stance_confidence=Decimal("1.0"),
                    quote="HTTPS encrypts traffic in transit.",
                    rationale="Direct.",
                    quote_validation="passed",
                )
            )
            db.flush()

        persist_stage4(db, run_stage4(claim.id, db))
        db.commit()
        return claim.id


# ---------------------------------------------------------------------------
# POST /report
# ---------------------------------------------------------------------------

def test_post_report_returns_200(api, engine: Engine) -> None:
    claim_id = seed(engine)
    response = api.post(f"/api/claims/{claim_id}/report")

    assert response.status_code == 200, response.text


def test_post_report_returns_misleading(api, engine: Engine) -> None:
    """The critical invariant, through HTTP."""
    claim_id = seed(engine, misleading=True)
    body = api.post(f"/api/claims/{claim_id}/report").json()

    assert body["verdict"]["verdict"] == "MISLEADING"
    assert body["verdict"]["score"] == 0.3333
    assert body["verdict"]["coverage"] == 1.0
    assert body["verdict"]["contradicted_weight_ratio"] == 0.3333


def test_post_report_contains_all_sections(api, engine: Engine) -> None:
    claim_id = seed(engine)
    body = api.post(f"/api/claims/{claim_id}/report").json()

    assert set(body) >= {
        "claim",
        "verdict",
        "subclaims",
        "sources",
        "explanation",
        "limitations",
        "report_version",
        "generated_at",
    }
    assert body["claim"]["id"] == str(claim_id)
    assert len(body["subclaims"]) == 3
    assert body["sources"]
    assert body["explanation"]["verdict_statement"]


def test_post_report_evidence_is_traceable(api, engine: Engine) -> None:
    claim_id = seed(engine)
    body = api.post(f"/api/claims/{claim_id}/report").json()

    for sub in body["subclaims"]:
        for ev in sub["evidence"]:
            assert ev["evidence_link_id"]
            assert ev["passage_id"]
            assert ev["source_id"]
            assert ev["source_url"].startswith("https://")
            assert ev["quote"] in ev["passage_text"]


def test_post_report_records_run(api, engine: Engine) -> None:
    claim_id = seed(engine)
    api.post(f"/api/claims/{claim_id}/report")

    with Session(bind=engine) as db:
        run = db.scalars(
            select(Run).where(Run.claim_id == claim_id, Run.stage == 6)
        ).one()
        assert run.status == "complete"
        assert run.finished_at is not None


def test_post_report_stamps_report_generated_at(api, engine: Engine) -> None:
    claim_id = seed(engine)
    api.post(f"/api/claims/{claim_id}/report")

    with Session(bind=engine) as db:
        verdict = db.scalars(
            select(Verdict).where(Verdict.claim_id == claim_id)
        ).one()
        assert verdict.report_generated_at is not None
        assert verdict.verdict == "MISLEADING"


def test_post_report_is_idempotent(api, engine: Engine) -> None:
    claim_id = seed(engine)
    first = api.post(f"/api/claims/{claim_id}/report").json()
    second = api.post(f"/api/claims/{claim_id}/report").json()

    assert first["verdict"] == second["verdict"]
    assert len(first["subclaims"]) == len(second["subclaims"])

    with Session(bind=engine) as db:
        assert db.scalar(
            select(func.count()).select_from(Verdict).where(Verdict.claim_id == claim_id)
        ) == 1


def test_post_report_does_not_mutate_verdicts(api, engine: Engine) -> None:
    claim_id = seed(engine)

    with Session(bind=engine) as db:
        before = db.scalars(select(Verdict).where(Verdict.claim_id == claim_id)).one()
        snapshot = (before.verdict, before.score, before.coverage)

    api.post(f"/api/claims/{claim_id}/report")

    with Session(bind=engine) as db:
        after = db.scalars(select(Verdict).where(Verdict.claim_id == claim_id)).one()
        assert (after.verdict, after.score, after.coverage) == snapshot


def test_post_report_404_for_missing_claim(api) -> None:
    response = api.post(f"/api/claims/{uuid.uuid4()}/report")
    assert response.status_code == 404


def test_post_report_409_without_verdict(api, engine: Engine) -> None:
    with Session(bind=engine) as db:
        claim = Claim(raw_text="No verdict.", domain="general", status="queued")
        db.add(claim)
        db.commit()
        claim_id = claim.id

    response = api.post(f"/api/claims/{claim_id}/report")
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# GET /report
# ---------------------------------------------------------------------------

def test_get_report_returns_200(api, engine: Engine) -> None:
    claim_id = seed(engine)
    response = api.get(f"/api/claims/{claim_id}/report")
    assert response.status_code == 200


def test_get_report_matches_post(api, engine: Engine) -> None:
    claim_id = seed(engine)
    posted = api.post(f"/api/claims/{claim_id}/report").json()
    fetched = api.get(f"/api/claims/{claim_id}/report").json()

    assert posted["verdict"] == fetched["verdict"]
    assert len(posted["subclaims"]) == len(fetched["subclaims"])


def test_get_report_writes_nothing(api, engine: Engine) -> None:
    claim_id = seed(engine)
    api.get(f"/api/claims/{claim_id}/report")

    with Session(bind=engine) as db:
        assert db.scalar(
            select(func.count()).select_from(Run).where(Run.stage == 6)
        ) == 0
        verdict = db.scalars(select(Verdict).where(Verdict.claim_id == claim_id)).one()
        assert verdict.report_generated_at is None


def test_get_report_404_for_missing_claim(api) -> None:
    assert api.get(f"/api/claims/{uuid.uuid4()}/report").status_code == 404


def test_get_report_409_without_verdict(api, engine: Engine) -> None:
    with Session(bind=engine) as db:
        claim = Claim(raw_text="No verdict.", domain="general", status="queued")
        db.add(claim)
        db.commit()
        claim_id = claim.id

    assert api.get(f"/api/claims/{claim_id}/report").status_code == 409


# ---------------------------------------------------------------------------
# Non-misleading case through HTTP
# ---------------------------------------------------------------------------

def test_supported_claim_reports_supported(api, engine: Engine) -> None:
    claim_id = seed(engine, misleading=False)
    body = api.post(f"/api/claims/{claim_id}/report").json()

    assert body["verdict"]["verdict"] == "SUPPORTED"
    assert body["verdict"]["score"] == 1.0
