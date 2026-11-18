"""Stage 4 integration tests against the real test database.

No network, no model. Every fixture is seeded directly so the aggregation is
exercised end to end through the database join.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Claim,
    EvidenceLink,
    Passage,
    Run,
    Source,
    Subclaim,
    SubclaimVerdict,
    Verdict,
)
from app.pipeline.stage4_verify import (
    AGGREGATION_VERSION,
    Stage4Error,
    Stage4InputError,
    persist_stage4,
    record_stage4_run,
    run_stage4,
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def make_claim(db: Session, *, domain: str = "technology") -> Claim:
    claim = Claim(
        raw_text="Using HTTPS makes a website completely secure.",
        domain=domain,
        status="queued",
    )
    db.add(claim)
    db.flush()
    return claim


def make_subclaim(db: Session, claim: Claim, ordinal: int, text: str) -> Subclaim:
    sub = Subclaim(
        claim_id=claim.id, ordinal=ordinal, text=text, claim_type="factual"
    )
    db.add(sub)
    db.flush()
    return sub


def make_source(
    db: Session,
    claim: Claim,
    index: int,
    *,
    authority: float | None = 1.0,
    recency: float | None = 1.0,
) -> Source:
    source = Source(
        claim_id=claim.id,
        url=f"https://example.org/{index}",
        url_hash=f"{index:064d}",
        domain="example.org",
        published_at=date(2026, 1, 1),
        retrieved_at=datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc),
        content_hash=f"h{index:063d}",
        authority_score=None if authority is None else Decimal(str(authority)),
        recency_score=None if recency is None else Decimal(str(recency)),
    )
    db.add(source)
    db.flush()
    return source


def make_passage(db: Session, source: Source, index: int) -> Passage:
    passage = Passage(
        source_id=source.id,
        text=f"Passage text number {index} about HTTPS behaviour.",
        chunk_index=index,
        quote_start=0,
        quote_end=40,
        token_count=10,
    )
    db.add(passage)
    db.flush()
    return passage


def make_link(
    db: Session,
    subclaim: Subclaim,
    passage: Passage,
    relationship: str,
    *,
    relevance: float = 1.0,
    quality: float = 1.0,
    stance: float = 1.0,
) -> EvidenceLink:
    link = EvidenceLink(
        subclaim_id=subclaim.id,
        passage_id=passage.id,
        relationship=relationship,
        relevance_score=Decimal(str(relevance)),
        evidence_quality_score=Decimal(str(quality)),
        stance_confidence=Decimal(str(stance)),
        quote="quote",
        rationale="rationale",
    )
    db.add(link)
    db.flush()
    return link


def add_evidence(
    db: Session,
    subclaim: Subclaim,
    relationship: str,
    *,
    index: int,
    authority: float | None = 1.0,
    recency: float | None = 1.0,
    relevance: float = 1.0,
    quality: float = 1.0,
    stance: float = 1.0,
) -> EvidenceLink:
    """One source + passage + link, all at full strength by default."""
    claim_id = subclaim.claim_id
    claim = db.scalars(select(Claim).where(Claim.id == claim_id)).one()
    source = make_source(
        db, claim, index, authority=authority, recency=recency
    )
    passage = make_passage(db, source, index)
    return make_link(
        db,
        subclaim,
        passage,
        relationship,
        relevance=relevance,
        quality=quality,
        stance=stance,
    )


# ---------------------------------------------------------------------------
# Canonical HTTPS example
# ---------------------------------------------------------------------------

def test_canonical_https_example(db: Session) -> None:
    """The deck's claim must reproduce exactly: 2 SUPPORTED, 1 CONTRADICTED,
    overall MISLEADING with score 0.3333, coverage 1.0000, ratio 0.3333."""
    claim = make_claim(db)
    s1 = make_subclaim(db, claim, 1, "HTTPS encrypts data in transit.")
    s2 = make_subclaim(db, claim, 2, "HTTPS protects against some network attacks.")
    s3 = make_subclaim(db, claim, 3, "HTTPS makes a website completely secure.")

    add_evidence(db, s1, "SUPPORTS", index=1)
    add_evidence(db, s2, "SUPPORTS", index=2)
    add_evidence(db, s3, "CONTRADICTS", index=3)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    verdicts = {v.subclaim_id: v for v in db.scalars(select(SubclaimVerdict)).all()}
    assert verdicts[s1.id].verdict == "SUPPORTED"
    assert verdicts[s2.id].verdict == "SUPPORTED"
    assert verdicts[s3.id].verdict == "CONTRADICTED"

    overall = db.scalars(select(Verdict)).one()
    assert overall.verdict == "MISLEADING"
    assert float(overall.score) == 0.3333
    assert float(overall.coverage) == 1.0
    assert float(overall.contradicted_weight_ratio) == 0.3333
    assert overall.subclaim_counts == {"SUPPORTED": 2, "CONTRADICTED": 1}


# ---------------------------------------------------------------------------
# Subclaim verdict persistence
# ---------------------------------------------------------------------------

def test_support_only_subclaim_persisted(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A supported subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "SUPPORTED"
    assert float(row.score) == 1.0
    assert float(row.coverage) == 1.0
    assert float(row.support_weight) == 1.0
    assert float(row.contradict_weight) == 0.0
    assert row.link_count == 1
    assert row.aggregation_version == AGGREGATION_VERSION
    assert row.rationale
    assert row.computed_at is not None


def test_contradiction_only_subclaim_persisted(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A contradicted subclaim.")
    add_evidence(db, sub, "CONTRADICTS", index=1)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "CONTRADICTED"
    assert float(row.score) == -1.0


def test_mixed_subclaim_persisted(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A mixed subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)
    add_evidence(db, sub, "CONTRADICTS", index=2)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "MIXED"
    assert float(row.score) == 0.0


def test_insufficient_subclaim_persisted(db: Session) -> None:
    claim = make_claim(db)
    make_subclaim(db, claim, 1, "A subclaim with no evidence.")

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "INSUFFICIENT"
    assert float(row.coverage) == 0.0
    assert row.link_count == 0


def test_weak_links_do_not_contribute(db: Session) -> None:
    """A link below MIN_LINK_WEIGHT is stored but excluded from the sums."""
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim with only weak evidence.")
    # 0.5*0.5*0.5*0.5*0.5 = 0.03125 < 0.15
    add_evidence(
        db, sub, "SUPPORTS", index=1, authority=0.5, recency=0.5, relevance=0.5,
        quality=0.5, stance=0.5,
    )

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    assert result.links_below_threshold == 1
    assert result.links_counted == 0

    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "INSUFFICIENT"
    assert float(row.support_weight) == 0.0

    # The link itself is untouched.
    assert db.scalar(select(func.count()).select_from(EvidenceLink)) == 1


def test_insufficient_link_never_contributes(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "INSUFFICIENT", index=1)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    assert result.links_insufficient == 1
    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Overall verdict persistence
# ---------------------------------------------------------------------------

def test_all_supported_claim(db: Session) -> None:
    claim = make_claim(db)
    for i in (1, 2, 3):
        sub = make_subclaim(db, claim, i, f"Supported subclaim {i}.")
        add_evidence(db, sub, "SUPPORTS", index=i)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert overall.verdict == "SUPPORTED"
    assert float(overall.score) == 1.0
    assert float(overall.coverage) == 1.0


def test_all_contradicted_claim(db: Session) -> None:
    claim = make_claim(db)
    for i in (1, 2, 3):
        sub = make_subclaim(db, claim, i, f"Contradicted subclaim {i}.")
        add_evidence(db, sub, "CONTRADICTS", index=i)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert overall.verdict == "CONTRADICTED"
    assert float(overall.score) == -1.0


def test_insufficient_coverage_claim(db: Session) -> None:
    """1 of 3 subclaims evidenced -> coverage 0.3333 < 0.50."""
    claim = make_claim(db)
    s1 = make_subclaim(db, claim, 1, "Evidenced subclaim.")
    make_subclaim(db, claim, 2, "Unevidenced subclaim A.")
    make_subclaim(db, claim, 3, "Unevidenced subclaim B.")
    add_evidence(db, s1, "SUPPORTS", index=1)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert overall.verdict == "INSUFFICIENT_EVIDENCE"
    assert float(overall.coverage) == pytest.approx(0.3333, abs=1e-4)
    assert overall.limitations


def test_claim_with_no_evidence_at_all(db: Session) -> None:
    """Abstention, not an error: no links means INSUFFICIENT_EVIDENCE."""
    claim = make_claim(db)
    make_subclaim(db, claim, 1, "No evidence here.")

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert overall.verdict == "INSUFFICIENT_EVIDENCE"
    assert float(overall.coverage) == 0.0


def test_report_generated_at_stays_null(db: Session) -> None:
    """Stage 6 sets this, not Stage 4."""
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert overall.report_generated_at is None


def test_limitations_are_json_list(db: Session) -> None:
    claim = make_claim(db)
    make_subclaim(db, claim, 1, "Unevidenced.")
    make_subclaim(db, claim, 2, "Unevidenced too.")

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert isinstance(overall.limitations, list)
    assert all(isinstance(item, str) for item in overall.limitations)


def test_subclaim_counts_are_jsonb(db: Session) -> None:
    claim = make_claim(db)
    s1 = make_subclaim(db, claim, 1, "Supported.")
    s2 = make_subclaim(db, claim, 2, "Contradicted.")
    add_evidence(db, s1, "SUPPORTS", index=1)
    add_evidence(db, s2, "CONTRADICTS", index=2)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    overall = db.scalars(select(Verdict)).one()
    assert overall.subclaim_counts == {"SUPPORTED": 1, "CONTRADICTED": 1}


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_rerun_updates_in_place(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    first_sub_count = db.scalar(select(func.count()).select_from(SubclaimVerdict))
    first_verdict_count = db.scalar(select(func.count()).select_from(Verdict))

    result2 = run_stage4(claim.id, db)
    persist_stage4(db, result2)
    db.flush()

    assert db.scalar(select(func.count()).select_from(SubclaimVerdict)) == first_sub_count
    assert db.scalar(select(func.count()).select_from(Verdict)) == first_verdict_count


def test_rerun_reflects_new_evidence(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()
    assert db.scalars(select(SubclaimVerdict)).one().verdict == "SUPPORTED"

    # Add contradicting evidence and re-run.
    add_evidence(db, sub, "CONTRADICTS", index=2)
    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()

    assert db.scalars(select(SubclaimVerdict)).one().verdict == "MIXED"


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def test_does_not_modify_sources(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    before = [
        (s.id, s.authority_score, s.recency_score)
        for s in db.scalars(select(Source)).all()
    ]

    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()

    after = [
        (s.id, s.authority_score, s.recency_score)
        for s in db.scalars(select(Source)).all()
    ]
    assert before == after


def test_does_not_modify_evidence_links(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    before = [
        (l.id, l.relationship, l.relevance_score, l.evidence_quality_score)
        for l in db.scalars(select(EvidenceLink)).all()
    ]

    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()

    after = [
        (l.id, l.relationship, l.relevance_score, l.evidence_quality_score)
        for l in db.scalars(select(EvidenceLink)).all()
    ]
    assert before == after


def test_null_authority_does_not_grant_authority(db: Session) -> None:
    """An unassessed source must not move a verdict."""
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1, authority=None, recency=None)

    result = run_stage4(claim.id, db)
    persist_stage4(db, result)
    db.flush()

    row = db.scalars(select(SubclaimVerdict)).one()
    assert row.verdict == "INSUFFICIENT"
    assert float(row.support_weight) == 0.0


def test_invalid_relationship_is_rejected_before_aggregation() -> None:
    """Corrupt relationship values are rejected by the aggregation guard.

    The database CHECK constraint makes this unreachable through the ORM, which
    is the first line of defence. This test exercises the second line: the
    guard inside ``build_contributions`` that protects against a value arriving
    from a restored dump, a manual UPDATE, or a future schema change.
    """
    from app.pipeline.stage4_verify import build_contributions

    with pytest.raises(Stage4Error, match="invalid stored relationship"):
        build_contributions([("REFUTES", 1.0, 1.0, 1.0, 1.0, 1.0)])


def test_database_rejects_invalid_relationship(db: Session) -> None:
    """The CHECK constraint prevents corrupt data being stored in the first place."""
    from sqlalchemy.exc import IntegrityError

    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    link = add_evidence(db, sub, "SUPPORTS", index=1)
    link.relationship = "REFUTES"

    with pytest.raises(IntegrityError):
        db.flush()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_claim_raises(db: Session) -> None:
    with pytest.raises(Stage4Error, match="not found"):
        run_stage4(uuid.uuid4(), db)


def test_claim_without_subclaims_raises(db: Session) -> None:
    claim = make_claim(db)
    with pytest.raises(Stage4InputError, match="no subclaims"):
        run_stage4(claim.id, db)


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

def test_run_record_is_written(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    persist_stage4(db, run_stage4(claim.id, db))
    record_stage4_run(db, claim.id, status="complete")
    db.flush()

    run = db.scalars(select(Run).where(Run.stage == 4)).one()
    assert run.stage == 4
    assert run.status == "complete"
    assert run.finished_at is not None
    # Deterministic stage: no model, so no tokens and no cost.
    assert run.tokens_in == 0
    assert run.tokens_out == 0
    assert float(run.cost_usd) == 0.0
    assert run.cache_hit is False


def test_failed_run_record_is_written(db: Session) -> None:
    claim = make_claim(db)
    record_stage4_run(db, claim.id, status="failed", error="aggregation error")
    db.flush()

    run = db.scalars(select(Run).where(Run.stage == 4)).one()
    assert run.status == "failed"
    assert "aggregation error" in run.error


# ---------------------------------------------------------------------------
# Stage boundaries
# ---------------------------------------------------------------------------

def test_stage4_creates_only_verdict_rows(db: Session) -> None:
    """No sources, passages or links are created by Stage 4."""
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)

    sources_before = db.scalar(select(func.count()).select_from(Source))
    passages_before = db.scalar(select(func.count()).select_from(Passage))
    links_before = db.scalar(select(func.count()).select_from(EvidenceLink))

    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()

    assert db.scalar(select(func.count()).select_from(Source)) == sources_before
    assert db.scalar(select(func.count()).select_from(Passage)) == passages_before
    assert db.scalar(select(func.count()).select_from(EvidenceLink)) == links_before
    assert db.scalar(select(func.count()).select_from(SubclaimVerdict)) == 1
    assert db.scalar(select(func.count()).select_from(Verdict)) == 1


def test_cascade_deletes_verdicts(db: Session) -> None:
    claim = make_claim(db)
    sub = make_subclaim(db, claim, 1, "A subclaim.")
    add_evidence(db, sub, "SUPPORTS", index=1)
    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()

    assert db.scalar(select(func.count()).select_from(Verdict)) == 1

    db.delete(claim)
    db.flush()

    assert db.scalar(select(func.count()).select_from(Verdict)) == 0
    assert db.scalar(select(func.count()).select_from(SubclaimVerdict)) == 0
