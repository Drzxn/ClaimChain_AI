"""Stage 6 integration tests against the real test database.

Stage 4 is run for real to produce the verdict, then Stage 6 reports on it.
No network, no model.
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
from app.pipeline.stage4_verify import persist_stage4, run_stage4
from app.pipeline.stage6_report import (
    REPORT_VERSION,
    Stage6Error,
    Stage6InputError,
    build_report,
    mark_report_generated,
    record_stage6_run,
)

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_verified_claim(
    db: Session,
    *,
    subclaim_specs: list[tuple[str, list[str]]],
    authority: float | None = 1.0,
    recency: float | None = 1.0,
    published: bool = True,
) -> Claim:
    """Create a claim, evidence, run Stage 4, and return the claim.

    ``subclaim_specs`` is a list of ``(subclaim_text, [relationships])``.
    """
    claim = Claim(
        raw_text="Using HTTPS makes a website completely secure.",
        domain="technology",
        status="queued",
    )
    db.add(claim)
    db.flush()

    index = 0
    for ordinal, (text, relationships) in enumerate(subclaim_specs, start=1):
        sub = Subclaim(
            claim_id=claim.id, ordinal=ordinal, text=text, claim_type="factual"
        )
        db.add(sub)
        db.flush()

        for relationship in relationships:
            index += 1
            source = Source(
                claim_id=claim.id,
                url=f"https://example.org/{index}",
                url_hash=f"{index:064d}",
                domain="example.org",
                title=f"Source {index}",
                publisher="Example Org",
                source_type="reference",
                published_at=date(2026, 1, 1) if published else None,
                retrieved_at=NOW,
                content_hash=f"h{index:063d}",
                authority_score=None if authority is None else Decimal(str(authority)),
                recency_score=None if recency is None else Decimal(str(recency)),
                assessment_rationale="assessed",
                assessed_at=NOW,
                cites_primary_sources=True,
            )
            db.add(source)
            db.flush()

            passage = Passage(
                source_id=source.id,
                text=f"Passage {index}: HTTPS encrypts traffic in transit.",
                chunk_index=0,
                quote_start=0,
                quote_end=44,
                token_count=11,
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
                    rationale="Directly states the property.",
                    quote_validation="passed",
                )
            )
            db.flush()

    persist_stage4(db, run_stage4(claim.id, db))
    db.flush()
    return claim


CANONICAL = [
    ("HTTPS encrypts data in transit.", ["SUPPORTS"]),
    ("HTTPS protects against some network attacks.", ["SUPPORTS"]),
    ("HTTPS makes a website completely secure.", ["CONTRADICTS"]),
]


# ---------------------------------------------------------------------------
# Critical invariant: the verdict is never reinterpreted
# ---------------------------------------------------------------------------

def test_misleading_verdict_is_reported_exactly(db: Session) -> None:
    """The canonical case: Stage 4 says MISLEADING, Stage 6 must say MISLEADING."""
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    stored = db.scalars(select(Verdict)).one()
    assert stored.verdict == "MISLEADING"

    report, _, _ = build_report(claim.id, db)

    assert report.verdict.verdict == "MISLEADING"
    assert report.verdict.score == 0.3333
    assert report.verdict.coverage == 1.0
    assert report.verdict.contradicted_weight_ratio == 0.3333
    assert report.verdict.subclaim_counts == {"SUPPORTED": 2, "CONTRADICTED": 1}


def test_report_verdict_matches_database_for_every_verdict(db: Session) -> None:
    """Whatever Stage 4 stored is what the report shows, verbatim."""
    cases = {
        "SUPPORTED": [("A.", ["SUPPORTS"]), ("B.", ["SUPPORTS"])],
        "CONTRADICTED": [("A.", ["CONTRADICTS"]), ("B.", ["CONTRADICTS"])],
        "MIXED": [("A.", ["SUPPORTS"]), ("B.", ["CONTRADICTS"])],
        "INSUFFICIENT_EVIDENCE": [("A.", []), ("B.", [])],
    }

    for expected, specs in cases.items():
        claim = seed_verified_claim(db, subclaim_specs=specs)
        stored = db.scalars(
            select(Verdict).where(Verdict.claim_id == claim.id)
        ).one()
        report, _, _ = build_report(claim.id, db)

        assert report.verdict.verdict == stored.verdict
        assert report.verdict.verdict == expected
        assert report.verdict.score == float(stored.score)


def test_stage6_does_not_recalculate_the_verdict(db: Session) -> None:
    """Corrupt the stored verdict; the report must echo the corruption, not fix it."""
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    stored = db.scalars(select(Verdict)).one()
    stored.verdict = "SUPPORTED"
    stored.score = Decimal("0.9999")
    db.flush()

    report, _, _ = build_report(claim.id, db)

    assert report.verdict.verdict == "SUPPORTED"
    assert report.verdict.score == 0.9999


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_report_has_all_sections(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    assert report.claim.id == claim.id
    assert report.claim.raw_text == claim.raw_text
    assert report.claim.domain == "technology"
    assert report.verdict.verdict
    assert len(report.subclaims) == 3
    assert report.sources
    assert report.explanation.verdict_statement
    assert report.report_version == REPORT_VERSION
    assert report.generated_at is not None


def test_subclaims_are_ordered_by_ordinal(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)
    assert [s.ordinal for s in report.subclaims] == [1, 2, 3]


def test_subclaim_verdicts_propagate(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    verdicts = [s.verdict.verdict for s in report.subclaims]
    assert verdicts == ["SUPPORTED", "SUPPORTED", "CONTRADICTED"]


def test_subclaim_weights_and_types_propagate(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    for sub in report.subclaims:
        assert sub.weight == 1.0
        assert sub.claim_type == "factual"
        assert sub.text


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

def test_every_evidence_item_is_traceable(db: Session) -> None:
    """Each evidence item must resolve to a real link, passage and source."""
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    link_ids = {l.id for l in db.scalars(select(EvidenceLink)).all()}
    passage_ids = {p.id for p in db.scalars(select(Passage)).all()}
    source_ids = {s.id for s in db.scalars(select(Source)).all()}

    seen = 0
    for sub in report.subclaims:
        for ev in sub.evidence:
            seen += 1
            assert ev.evidence_link_id in link_ids
            assert ev.passage_id in passage_ids
            assert ev.source_id in source_ids
            assert ev.subclaim_id == sub.id

    assert seen == report.evidence_link_count
    assert seen == len(link_ids)


def test_quotes_match_stored_passages(db: Session) -> None:
    """No fabricated quotes: every quote is a substring of its passage."""
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    passages = {p.id: p for p in db.scalars(select(Passage)).all()}
    for sub in report.subclaims:
        for ev in sub.evidence:
            if ev.quote:
                assert ev.quote in passages[ev.passage_id].text


def test_source_urls_match_stored_sources(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    stored = {s.id: s.url for s in db.scalars(select(Source)).all()}
    for source in report.sources:
        assert source.url == stored[source.id]


def test_evidence_links_to_correct_subclaim(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    for sub in report.subclaims:
        for ev in sub.evidence:
            stored = db.scalars(
                select(EvidenceLink).where(EvidenceLink.id == ev.evidence_link_id)
            ).one()
            assert stored.subclaim_id == sub.id


# ---------------------------------------------------------------------------
# Source assessment propagation
# ---------------------------------------------------------------------------

def test_stage5_assessment_appears_in_report(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    assert report.sources
    for source in report.sources:
        assert source.authority_score == 1.0
        assert source.recency_score == 1.0
        assert source.source_type == "reference"
        assert source.assessment_rationale == "assessed"
        assert source.assessed_at is not None
        assert source.cites_primary_sources is True


def test_unassessed_sources_are_reported_as_null(db: Session) -> None:
    claim = seed_verified_claim(
        db, subclaim_specs=CANONICAL, authority=None, recency=None
    )
    report, _, _ = build_report(claim.id, db)

    for source in report.sources:
        assert source.authority_score is None
        assert source.recency_score is None


def test_duplicate_sources_are_flagged(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    # Mark one source as a duplicate of another.
    sources = db.scalars(select(Source).order_by(Source.id)).all()
    sources[1].duplicate_of = sources[0].id
    db.flush()

    report, _, _ = build_report(claim.id, db)
    flagged = [s for s in report.sources if s.is_duplicate]
    assert len(flagged) == 1
    assert flagged[0].duplicate_of == sources[0].id


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def test_explanation_is_deterministic(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    first, _, _ = build_report(claim.id, db)
    second, _, _ = build_report(claim.id, db)

    assert first.explanation.verdict_statement == second.explanation.verdict_statement
    assert first.explanation.subclaim_statements == second.explanation.subclaim_statements
    assert first.explanation.supporting_points == second.explanation.supporting_points
    assert first.explanation.contradicting_points == second.explanation.contradicting_points


def test_explanation_lists_supporting_and_contradicting_points(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)

    assert report.explanation.supporting_points
    assert report.explanation.contradicting_points
    assert report.explanation.narrative_source == "deterministic"
    assert report.explanation.narrative == ""


def test_explanation_names_the_verdict(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    report, _, _ = build_report(claim.id, db)
    assert "MISLEADING" in report.explanation.verdict_statement


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

def test_insufficient_evidence_limitation_is_reported(db: Session) -> None:
    claim = seed_verified_claim(
        db, subclaim_specs=[("A.", []), ("B.", [])]
    )
    report, _, _ = build_report(claim.id, db)

    codes = {l.code for l in report.limitations}
    assert "insufficient_evidence" in codes


def test_subclaims_without_evidence_are_reported(db: Session) -> None:
    claim = seed_verified_claim(
        db, subclaim_specs=[("A.", ["SUPPORTS"]), ("B.", [])]
    )
    report, _, _ = build_report(claim.id, db)

    codes = {l.code for l in report.limitations}
    assert "subclaims_without_evidence" in codes


def test_missing_publication_date_is_reported(db: Session) -> None:
    claim = seed_verified_claim(
        db, subclaim_specs=[("A.", ["SUPPORTS"])], published=False
    )
    report, _, _ = build_report(claim.id, db)

    codes = {l.code for l in report.limitations}
    assert "sources_without_publication_date" in codes


def test_unassessed_sources_limitation(db: Session) -> None:
    claim = seed_verified_claim(
        db, subclaim_specs=[("A.", ["SUPPORTS"])], authority=None, recency=None
    )
    report, _, _ = build_report(claim.id, db)

    codes = {l.code for l in report.limitations}
    assert "sources_unassessed" in codes


def test_limitations_are_never_invented(db: Session) -> None:
    """A clean claim with assessed, dated, non-duplicate sources has no
    evidence or source limitations."""
    claim = seed_verified_claim(
        db, subclaim_specs=[("A.", ["SUPPORTS"]), ("B.", ["SUPPORTS"])]
    )
    report, _, _ = build_report(claim.id, db)

    codes = {l.code for l in report.limitations}
    assert "insufficient_evidence" not in codes
    assert "sources_without_publication_date" not in codes
    assert "sources_unassessed" not in codes
    assert "duplicate_sources" not in codes


def test_limitations_have_provenance(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=[("A.", []), ("B.", [])])
    report, _, _ = build_report(claim.id, db)

    for limitation in report.limitations:
        assert limitation.source in {"verdict", "evidence", "sources", "report"}
        assert limitation.message


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_claim_with_no_evidence_links(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=[("A.", []), ("B.", [])])
    report, _, _ = build_report(claim.id, db)

    assert report.evidence_link_count == 0
    assert report.sources == []
    assert all(sub.evidence == [] for sub in report.subclaims)
    assert report.verdict.verdict == "INSUFFICIENT_EVIDENCE"


def test_missing_claim_raises(db: Session) -> None:
    with pytest.raises(Stage6Error, match="not found"):
        build_report(uuid.uuid4(), db)


def test_claim_without_verdict_raises(db: Session) -> None:
    claim = Claim(raw_text="No verdict.", domain="general", status="queued")
    db.add(claim)
    db.flush()

    with pytest.raises(Stage6InputError, match="no verdict"):
        build_report(claim.id, db)


# ---------------------------------------------------------------------------
# Idempotency and non-mutation
# ---------------------------------------------------------------------------

def test_repeated_report_generation_is_safe(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    first, _, _ = build_report(claim.id, db)
    second, _, _ = build_report(claim.id, db)

    assert first.verdict.verdict == second.verdict.verdict
    assert first.evidence_link_count == second.evidence_link_count
    assert len(first.subclaims) == len(second.subclaims)


def test_report_generation_creates_no_rows(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    counts_before = {
        "verdicts": db.scalar(select(func.count()).select_from(Verdict)),
        "subclaim_verdicts": db.scalar(select(func.count()).select_from(SubclaimVerdict)),
        "sources": db.scalar(select(func.count()).select_from(Source)),
        "evidence_links": db.scalar(select(func.count()).select_from(EvidenceLink)),
    }

    build_report(claim.id, db)
    db.flush()

    assert db.scalar(select(func.count()).select_from(Verdict)) == counts_before["verdicts"]
    assert db.scalar(select(func.count()).select_from(SubclaimVerdict)) == counts_before["subclaim_verdicts"]
    assert db.scalar(select(func.count()).select_from(Source)) == counts_before["sources"]
    assert db.scalar(select(func.count()).select_from(EvidenceLink)) == counts_before["evidence_links"]


def test_stage4_values_are_not_modified(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    before_verdict = db.scalars(select(Verdict)).one()
    snapshot = (
        before_verdict.verdict,
        before_verdict.score,
        before_verdict.coverage,
        before_verdict.contradicted_weight_ratio,
        dict(before_verdict.subclaim_counts),
    )
    before_subs = [
        (v.subclaim_id, v.verdict, v.score, v.support_weight)
        for v in db.scalars(select(SubclaimVerdict)).all()
    ]

    build_report(claim.id, db)
    mark_report_generated(db, claim.id)
    db.flush()

    after_verdict = db.scalars(select(Verdict)).one()
    assert (
        after_verdict.verdict,
        after_verdict.score,
        after_verdict.coverage,
        after_verdict.contradicted_weight_ratio,
        dict(after_verdict.subclaim_counts),
    ) == snapshot

    after_subs = [
        (v.subclaim_id, v.verdict, v.score, v.support_weight)
        for v in db.scalars(select(SubclaimVerdict)).all()
    ]
    assert after_subs == before_subs


def test_stage5_values_are_not_modified(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)

    before = [
        (s.id, s.authority_score, s.recency_score, s.source_type, s.assessed_at)
        for s in db.scalars(select(Source)).all()
    ]

    build_report(claim.id, db)
    mark_report_generated(db, claim.id)
    db.flush()

    after = [
        (s.id, s.authority_score, s.recency_score, s.source_type, s.assessed_at)
        for s in db.scalars(select(Source)).all()
    ]
    assert before == after


def test_mark_report_generated_stamps_only_that_field(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    verdict_row = db.scalars(select(Verdict)).one()
    assert verdict_row.report_generated_at is None

    mark_report_generated(db, claim.id)
    db.flush()

    assert verdict_row.report_generated_at is not None
    assert verdict_row.verdict == "MISLEADING"


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

def test_run_record_is_written(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    build_report(claim.id, db)
    record_stage6_run(db, claim.id, status="complete")
    db.flush()

    run = db.scalars(select(Run).where(Run.stage == 6)).one()
    assert run.stage == 6
    assert run.status == "complete"
    assert run.finished_at is not None
    # Deterministic path: no model, so no tokens and no cost.
    assert run.tokens_in == 0
    assert run.tokens_out == 0
    assert float(run.cost_usd) == 0.0


def test_failed_run_record_is_written(db: Session) -> None:
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    record_stage6_run(db, claim.id, status="failed", error="report error")
    db.flush()

    run = db.scalars(select(Run).where(Run.stage == 6)).one()
    assert run.status == "failed"
    assert "report error" in run.error


def test_repeated_runs_are_allowed(db: Session) -> None:
    """Stage 6 may run more than once; each attempt is its own run row."""
    claim = seed_verified_claim(db, subclaim_specs=CANONICAL)
    record_stage6_run(db, claim.id, status="complete", attempt=1)
    record_stage6_run(db, claim.id, status="complete", attempt=2)
    db.flush()

    runs = db.scalars(select(Run).where(Run.stage == 6)).all()
    assert len(runs) == 2
