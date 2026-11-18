"""Stage 2 integration tests against the real test database.

Fixture mode is used throughout, so no network call is made and no API key is
needed. The database is the isolated ``claimchain_test`` instance.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.db.models import Claim, EvidenceLink, Passage, Run, Source, Subclaim
from app.pipeline.stage2_retrieve import (
    MAX_TOTAL_URLS_PER_CLAIM,
    FixtureSearchProvider,
    Stage2Error,
    Stage2InputError,
    fallback_queries,
    generate_queries,
    record_stage2_run,
    run_stage2,
)
from app.services.exa import SearchError
from app.services.fixtures import CANONICAL_CLAIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_claim_with_subclaims(db: Session, subclaim_texts: list[str]) -> Claim:
    claim = Claim(raw_text=CANONICAL_CLAIM, domain="technology", status="queued")
    db.add(claim)
    db.flush()

    for index, text in enumerate(subclaim_texts, start=1):
        db.add(
            Subclaim(
                claim_id=claim.id,
                ordinal=index,
                text=text,
                claim_type="factual",
            )
        )
    db.flush()
    return claim


CANONICAL_SUBCLAIMS = [
    "HTTPS encrypts data in transit between the browser and the server.",
    "HTTPS makes a website completely secure against all attacks.",
]


@pytest.fixture
def clean(db: Session):
    """Stage 2 commits nothing itself, but tests may flush; isolate per test."""
    yield db
    db.rollback()


# ---------------------------------------------------------------------------
# Fixture mode
# ---------------------------------------------------------------------------

def test_fixture_mode_persists_sources_and_passages(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)

    result = run_stage2(claim.id, db, fixture_mode=True)

    assert result.fixture_mode is True
    assert result.sources_created > 0
    assert result.passages_created > 0

    sources = db.scalars(select(Source).where(Source.claim_id == claim.id)).all()
    assert len(sources) == result.sources_created
    assert all(s.claim_id == claim.id for s in sources)

    passages = db.scalars(select(Passage)).all()
    assert len(passages) == result.passages_created


def test_fixture_mode_needs_no_api_key(db: Session, monkeypatch) -> None:
    """Fixture mode must work with no EXA_API_KEY configured at all."""
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)
    assert result.sources_created > 0


def test_fixture_mode_is_deterministic(db: Session) -> None:
    claim_a = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result_a = run_stage2(claim_a.id, db, fixture_mode=True)
    db.rollback()

    claim_b = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result_b = run_stage2(claim_b.id, db, fixture_mode=True)

    assert result_a.sources_created == result_b.sources_created
    assert result_a.passages_created == result_b.passages_created


def test_fixture_mode_creates_no_evidence_links(db: Session) -> None:
    """Stage 2 must never classify evidence."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)

    assert db.scalar(select(func.count()).select_from(EvidenceLink)) == 0


def test_fixture_mode_leaves_source_assessment_unset(db: Session) -> None:
    """Authority and recency belong to Stage 5, not here."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)

    for source in db.scalars(select(Source)).all():
        assert source.authority_score is None
        assert source.recency_score is None
        assert source.assessed_at is None
        assert source.source_type == "unknown"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_url_creates_one_source(db: Session) -> None:
    """The fixture set returns the same URLs for every query."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)

    hashes = [s.url_hash for s in db.scalars(select(Source)).all()]
    assert len(hashes) == len(set(hashes)), "duplicate url_hash rows were created"
    assert result.duplicates_linked >= 1


def test_duplicate_content_is_linked_not_duplicated(db: Session) -> None:
    """The mirror page has identical text; it must point at the original."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)

    sources = db.scalars(select(Source)).all()
    duplicates = [s for s in sources if s.duplicate_of is not None]
    assert duplicates, "expected the mirror page to be recorded as a duplicate"

    for dup in duplicates:
        assert dup.duplicate_of in {s.id for s in sources}
        # A duplicate stores metadata but no passages of its own.
        assert db.scalar(
            select(func.count()).select_from(Passage).where(Passage.source_id == dup.id)
        ) == 0


def test_content_hash_is_populated(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)

    for source in db.scalars(select(Source)).all():
        assert source.content_hash is not None
        assert len(source.content_hash) == 64


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------

def test_passages_have_valid_offsets(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)

    for passage in db.scalars(select(Passage)).all():
        assert passage.quote_start >= 0
        assert passage.quote_end >= passage.quote_start
        assert passage.token_count > 0
        assert passage.text.strip()


def test_chunk_indices_are_unique_per_source(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)

    for source in db.scalars(select(Source)).all():
        indices = [
            p.chunk_index
            for p in db.scalars(
                select(Passage).where(Passage.source_id == source.id)
            ).all()
        ]
        assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def test_every_subclaim_gets_a_candidate_entry(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)

    assert len(result.candidates) == len(CANONICAL_SUBCLAIMS)
    for candidate in result.candidates:
        assert candidate.subclaim_id is not None
        assert isinstance(candidate.passage_ids, list)


def test_candidates_reference_real_passages(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)

    real_ids = {p.id for p in db.scalars(select(Passage)).all()}
    for candidate in result.candidates:
        for passage_id in candidate.passage_ids:
            assert passage_id in real_ids


def test_relevant_candidates_are_selected(db: Session) -> None:
    """The canonical claim's subclaims should match the fixture pages."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)

    assert any(c.passage_ids for c in result.candidates), (
        "expected at least one subclaim to have candidate passages"
    )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_claim_without_subclaims_raises(db: Session) -> None:
    claim = Claim(raw_text="No subclaims here.", domain="general", status="queued")
    db.add(claim)
    db.flush()

    with pytest.raises(Stage2InputError):
        run_stage2(claim.id, db, fixture_mode=True)


def test_missing_claim_raises(db: Session) -> None:
    with pytest.raises(Stage2Error, match="not found"):
        run_stage2(uuid.uuid4(), db, fixture_mode=True)


class FailingSearch:
    """A provider that always fails."""

    def search(self, query, *, num_results=None):
        raise SearchError("provider down")


def test_search_failure_does_not_crash_the_stage(db: Session) -> None:
    """A failed query is recorded; the run continues with zero sources."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, search=FailingSearch(), fixture_mode=True)

    assert result.sources_created == 0
    assert result.urls_skipped, "the failure must be recorded"
    # Every subclaim still gets an entry, with no candidates.
    assert len(result.candidates) == len(CANONICAL_SUBCLAIMS)
    assert all(c.passage_ids == [] for c in result.candidates)


def test_zero_results_is_not_an_error(db: Session) -> None:
    """No evidence found is a valid outcome, not a failure."""

    class EmptySearch:
        def search(self, query, *, num_results=None):
            return []

    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, search=EmptySearch(), fixture_mode=True)

    assert result.sources_created == 0
    assert result.passages_created == 0
    assert len(result.candidates) == len(CANONICAL_SUBCLAIMS)


def test_unfetchable_url_is_skipped(db: Session) -> None:
    """A page that cannot be fetched must not stop the others."""
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)

    def failing_fetch(url: str) -> str:
        from app.services.fetcher import FetchError

        raise FetchError("simulated failure")

    result = run_stage2(
        claim.id, db, fixture_mode=True, fetch_html=failing_fetch
    )
    assert result.sources_created == 0
    assert result.urls_skipped


def test_url_cap_is_enforced(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)
    assert result.sources_created <= MAX_TOTAL_URLS_PER_CLAIM


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------

def test_run_record_is_written(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    result = run_stage2(claim.id, db, fixture_mode=True)
    record_stage2_run(db, claim.id, status="complete", result=result)
    db.flush()

    run = db.scalars(select(Run).where(Run.claim_id == claim.id)).one()
    assert run.stage == 2
    assert run.attempt == 1
    assert run.status == "complete"
    assert run.finished_at is not None
    # Fixture runs are flagged so provenance is never ambiguous.
    assert run.cache_hit is True
    # Cost is never invented.
    assert float(run.cost_usd) == 0.0


def test_failed_run_record_is_written(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    record_stage2_run(db, claim.id, status="failed", error="search unavailable")
    db.flush()

    run = db.scalars(select(Run).where(Run.claim_id == claim.id)).one()
    assert run.status == "failed"
    assert "search unavailable" in run.error


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def test_fallback_queries_are_derived_from_subclaim() -> None:
    queries = fallback_queries("HTTPS encrypts data in transit")
    assert queries
    assert all("HTTPS encrypts data in transit" in q for q in queries)
    assert len(queries) <= 3


def test_fallback_queries_never_invent_terms() -> None:
    queries = fallback_queries("A short claim.")
    joined = " ".join(queries).lower()
    assert "short claim" in joined


def test_generate_queries_without_llm_uses_fallback() -> None:
    queries, t_in, t_out = generate_queries("Some subclaim text", llm=None)
    assert queries
    assert (t_in, t_out) == (0, 0)


def test_generate_queries_survives_llm_failure() -> None:
    class BrokenLLM:
        def generate_json(self, **kwargs):
            raise RuntimeError("model unavailable")

    queries, _, _ = generate_queries("Some subclaim text", llm=BrokenLLM())
    assert queries, "must fall back rather than raise"


# ---------------------------------------------------------------------------
# Cascade integrity
# ---------------------------------------------------------------------------

def test_deleting_claim_cascades_sources_and_passages(db: Session) -> None:
    claim = make_claim_with_subclaims(db, CANONICAL_SUBCLAIMS)
    run_stage2(claim.id, db, fixture_mode=True)
    db.flush()

    assert db.scalar(select(func.count()).select_from(Source)) > 0

    db.delete(claim)
    db.flush()

    assert db.scalar(select(func.count()).select_from(Source)) == 0
    assert db.scalar(select(func.count()).select_from(Passage)) == 0
