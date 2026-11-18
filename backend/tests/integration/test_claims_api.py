"""Claims API tests.

These run against the isolated ``claimchain_test`` database with the real
migration applied. The LLM is replaced by a scripted fake, so no network call
is made and no API quota is consumed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.api.claims import get_llm_service
from app.db.models import Claim, Run, Subclaim
from app.main import create_app
from app.services.llm import LLMNotConfiguredError, LLMTransportError
from tests.fakes import FakeGemini, valid_decomposition

CANONICAL = "Using HTTPS makes a website completely secure."


@pytest.fixture
def api(engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """TestClient whose DB dependency points at the test database.

    ``get_db`` is overridden to yield a session bound to the test engine, and
    ``get_llm_service`` is overridden per test with a scripted fake.
    """

    def _override_get_db():
        session = Session(bind=engine, expire_on_commit=False)
        try:
            yield session
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[_db_dependency()] = _override_get_db
    client = TestClient(app)

    def _with_llm(fake: FakeGemini) -> TestClient:
        app.dependency_overrides[get_llm_service] = lambda: fake
        return client

    yield _with_llm

    app.dependency_overrides.clear()


def _db_dependency():
    """The exact ``get_db`` callable used by the router."""
    from app.db.session import get_db

    return get_db


@pytest.fixture(autouse=True)
def _clean_tables(engine: Engine):
    """Remove rows created by each API test.

    API tests commit, so the rollback fixture used by schema tests does not
    apply here. Deleting claims cascades to every child table.
    """
    yield
    with Session(bind=engine) as session:
        for claim in session.scalars(select(Claim)).all():
            session.delete(claim)
        session.commit()


# ---------------------------------------------------------------------------
# POST /api/claims — success
# ---------------------------------------------------------------------------

def test_create_claim_success(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)

    response = client.post("/api/claims", json={"raw_text": CANONICAL})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["domain"] == "technology"
    assert body["subclaim_count"] == 2
    assert body["id"]


def test_create_claim_persists_subclaims(api, engine: Engine) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)

    claim_id = client.post("/api/claims", json={"raw_text": CANONICAL}).json()["id"]

    with Session(bind=engine) as session:
        subs = session.scalars(
            select(Subclaim).where(Subclaim.claim_id == claim_id).order_by(Subclaim.ordinal)
        ).all()
        assert len(subs) == 2
        assert [s.ordinal for s in subs] == [1, 2]


def test_all_stored_weights_are_exactly_one(api, engine: Engine) -> None:
    """The model proposes varied weights; the database must hold 1.0."""
    payload = valid_decomposition(
        subclaims=[
            {"text": "First distinct claim.", "claim_type": "factual", "weight": 2.5},
            {"text": "Second distinct claim.", "claim_type": "causal", "weight": 0.2},
        ]
    )
    fake = FakeGemini([payload])
    client = api(fake)

    claim_id = client.post("/api/claims", json={"raw_text": CANONICAL}).json()["id"]

    with Session(bind=engine) as session:
        weights = session.scalars(
            select(Subclaim.weight).where(Subclaim.claim_id == claim_id)
        ).all()
        assert len(weights) == 2
        for weight in weights:
            assert float(weight) == 1.0, f"stored weight was {weight}, expected 1.0"


def test_create_claim_records_one_run(api, engine: Engine) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)

    claim_id = client.post("/api/claims", json={"raw_text": CANONICAL}).json()["id"]

    with Session(bind=engine) as session:
        runs = session.scalars(select(Run).where(Run.claim_id == claim_id)).all()
        assert len(runs) == 1
        assert runs[0].stage == 1
        assert runs[0].attempt == 1
        assert runs[0].status == "complete"
        assert runs[0].tokens_in == 100
        assert float(runs[0].cost_usd) == 0.0


# ---------------------------------------------------------------------------
# POST /api/claims — input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_or_whitespace_rejected(api, text: str) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    response = client.post("/api/claims", json={"raw_text": text})
    assert response.status_code == 422
    assert fake.call_count == 0, "no LLM call should happen for invalid input"


def test_over_2000_chars_rejected(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    response = client.post("/api/claims", json={"raw_text": "x" * 2001})
    assert response.status_code == 422
    assert fake.call_count == 0


def test_missing_field_rejected(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    assert client.post("/api/claims", json={}).status_code == 422


def test_extra_field_rejected(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    response = client.post(
        "/api/claims", json={"raw_text": CANONICAL, "domain": "technology"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/claims — repair retry at the API layer
# ---------------------------------------------------------------------------

def test_repair_retry_success_records_two_runs(api, engine: Engine) -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, valid_decomposition()])
    client = api(fake)

    response = client.post("/api/claims", json={"raw_text": CANONICAL})
    assert response.status_code == 201
    claim_id = response.json()["id"]

    with Session(bind=engine) as session:
        runs = session.scalars(
            select(Run).where(Run.claim_id == claim_id).order_by(Run.attempt)
        ).all()
        assert len(runs) == 2
        assert runs[0].attempt == 1 and runs[0].status == "failed"
        assert runs[1].attempt == 2 and runs[1].status == "complete"
        assert runs[0].error is not None


def test_repair_failure_returns_error_and_leaves_no_subclaims(api, engine: Engine) -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, bad])
    client = api(fake)

    response = client.post("/api/claims", json={"raw_text": CANONICAL})
    assert response.status_code == 502
    assert fake.call_count == 2, "exactly one repair retry, never a third attempt"

    with Session(bind=engine) as session:
        claim = session.scalars(select(Claim)).one()
        assert claim.status == "failed"
        assert claim.error
        assert session.scalar(
            select(func.count()).select_from(Subclaim).where(Subclaim.claim_id == claim.id)
        ) == 0


def test_failed_claim_preserves_run_record(api, engine: Engine) -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, bad])
    client = api(fake)

    client.post("/api/claims", json={"raw_text": CANONICAL})

    with Session(bind=engine) as session:
        runs = session.scalars(select(Run).order_by(Run.attempt)).all()
        assert len(runs) >= 1
        assert any(r.status == "failed" for r in runs)


# ---------------------------------------------------------------------------
# POST /api/claims — infrastructure failures
# ---------------------------------------------------------------------------

def test_llm_not_configured_returns_503(api, engine: Engine) -> None:
    fake = FakeGemini([LLMNotConfiguredError("no key")])
    client = api(fake)

    response = client.post("/api/claims", json={"raw_text": CANONICAL})
    assert response.status_code == 503

    with Session(bind=engine) as session:
        claim = session.scalars(select(Claim)).one()
        assert claim.status == "failed"


def test_transport_error_returns_502(api, engine: Engine) -> None:
    fake = FakeGemini([LLMTransportError("network down")])
    client = api(fake)

    response = client.post("/api/claims", json={"raw_text": CANONICAL})
    assert response.status_code == 502

    with Session(bind=engine) as session:
        claim = session.scalars(select(Claim)).one()
        assert claim.status == "failed"


def test_error_response_never_leaks_secrets(api) -> None:
    fake = FakeGemini([LLMTransportError("network down")])
    client = api(fake)
    body = client.post("/api/claims", json={"raw_text": CANONICAL}).text
    assert "GEMINI_API_KEY" not in body
    assert "AIza" not in body
    assert "fake-key-for-tests" not in body


# ---------------------------------------------------------------------------
# GET /api/claims/{id}
# ---------------------------------------------------------------------------

def test_get_claim_returns_subclaims(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)

    claim_id = client.post("/api/claims", json={"raw_text": CANONICAL}).json()["id"]
    response = client.get(f"/api/claims/{claim_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == claim_id
    assert body["domain"] == "technology"
    assert body["status"] == "queued"
    assert len(body["subclaims"]) == 2

    first = body["subclaims"][0]
    assert set(first) == {"id", "ordinal", "text", "claim_type", "weight"}
    assert first["ordinal"] == 1
    assert float(first["weight"]) == 1.0


def test_get_claim_orders_subclaims_by_ordinal(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    claim_id = client.post("/api/claims", json={"raw_text": CANONICAL}).json()["id"]

    body = client.get(f"/api/claims/{claim_id}").json()
    assert [s["ordinal"] for s in body["subclaims"]] == [1, 2]


def test_get_nonexistent_claim_returns_404(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    response = client.get("/api/claims/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_get_claim_rejects_malformed_uuid(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    assert client.get("/api/claims/not-a-uuid").status_code == 422


def test_get_claim_does_not_expose_internal_config(api) -> None:
    fake = FakeGemini([valid_decomposition()])
    client = api(fake)
    claim_id = client.post("/api/claims", json={"raw_text": CANONICAL}).json()["id"]

    body = client.get(f"/api/claims/{claim_id}").text
    assert "model_config" not in body
    assert "prompt_version" not in body
    assert "GEMINI_API_KEY" not in body
