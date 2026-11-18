"""Database round-trip check.

A standalone utility, not a migration and not demo data. It proves the ORM,
the migration, and the cascade rules work end to end against the live
development database, then removes everything it created.

    python -m app.db.seed_roundtrip

Exit code 0 means every check passed. Non-zero means the schema or the ORM is
broken and the next phase should not start.

Nothing is left behind: the claim is deleted at the end, and the cascade
removes every descendant row.
"""

from __future__ import annotations

import sys
import uuid
from decimal import Decimal

from sqlalchemy import func, select

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
from app.db.session import SessionLocal

CLAIM_TEXT = (
    "Using HTTPS makes a website completely secure, and it prevents all "
    "forms of web attack."
)


class CheckFailed(Exception):
    """Raised when a round-trip assertion fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _count(db, model) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


def run() -> int:
    """Insert, read back, verify relationships, delete, verify cascade."""
    db = SessionLocal()
    claim_id: uuid.UUID | None = None

    try:
        # --- 1. insert a realistic claim ---------------------------------
        claim = Claim(
            raw_text=CLAIM_TEXT,
            domain="technology",
            status="queued",
            model_config={"model": "gemini-3.5-flash-lite", "temperature": 0},
        )
        db.add(claim)
        db.commit()
        db.refresh(claim)
        claim_id = claim.id
        print(f"[1/8] inserted claim {claim.id}")

        # --- 2. verify the UUID and fields --------------------------------
        _require(isinstance(claim.id, uuid.UUID), "id is not a UUID")
        _require(claim.domain == "technology", "domain did not persist")
        _require(claim.status == "queued", "status did not persist")
        _require(claim.raw_text == CLAIM_TEXT, "raw_text did not persist")
        _require(claim.created_at is not None, "created_at not set")
        _require(claim.created_at.tzinfo is not None, "created_at is naive")
        print("[2/8] uuid, domain, status, raw_text, timestamptz all correct")

        # --- 3. read back through a fresh query ---------------------------
        fetched = db.scalars(select(Claim).where(Claim.id == claim_id)).one()
        _require(fetched.id == claim_id, "read-back returned a different row")
        print("[3/8] read back by UUID")

        # --- 4. build the relationship layer ------------------------------
        sub1 = Subclaim(
            claim_id=claim.id,
            ordinal=1,
            text="HTTPS encrypts data in transit.",
            claim_type="factual",
        )
        sub2 = Subclaim(
            claim_id=claim.id,
            ordinal=2,
            text="HTTPS prevents all forms of web attack.",
            claim_type="factual",
        )
        source = Source(
            claim_id=claim.id,
            url="https://example.org/https-explained",
            url_hash="a" * 64,
            domain="example.org",
            title="How HTTPS works",
            source_type="reference",
            published_at=None,
        )
        db.add_all([sub1, sub2, source])
        db.commit()

        passage = Passage(
            source_id=source.id,
            text=(
                "HTTPS encrypts traffic between the browser and the server, "
                "but it does not prevent cross-site scripting or SQL injection."
            ),
            chunk_index=0,
            token_count=24,
        )
        db.add(passage)
        db.commit()

        link = EvidenceLink(
            subclaim_id=sub1.id,
            passage_id=passage.id,
            relationship="SUPPORTS",
            relevance_score=Decimal("0.900"),
            evidence_quality_score=Decimal("0.850"),
            stance_confidence=Decimal("0.880"),
            quote="HTTPS encrypts traffic between the browser and the server",
            rationale="States the encryption property directly.",
            model="gemini-3.5-flash-lite",
            prompt_version="analyze.v1",
        )
        db.add(link)
        db.commit()
        print("[4/8] built subclaims, source, passage, evidence link")

        # --- 5. verify the relationship layer loads ----------------------
        db.refresh(claim)
        _require(len(claim.subclaims) == 2, "claim did not load its subclaims")
        _require(len(claim.sources) == 1, "claim did not load its sources")
        _require(
            claim.subclaims[0].ordinal == 1,
            "subclaims are not ordered by ordinal",
        )
        _require(
            claim.sources[0].passages[0].id == passage.id,
            "source did not load its passages",
        )
        _require(
            claim.subclaims[0].evidence_links[0].id == link.id,
            "subclaim did not load its evidence links",
        )
        print("[5/8] relationship layer loads claim -> subclaims/sources -> passages -> links")

        # --- 6. exercise the one-to-one verdict tables -------------------
        db.add(
            SubclaimVerdict(
                subclaim_id=sub1.id,
                verdict="SUPPORTED",
                score=Decimal("0.8500"),
                coverage=Decimal("1.0000"),
                support_weight=Decimal("0.6732"),
                link_count=1,
                rationale="One supporting passage.",
                aggregation_version="agg-1.0.0",
            )
        )
        db.add(
            Verdict(
                claim_id=claim.id,
                verdict="MIXED",
                score=Decimal("0.0000"),
                coverage=Decimal("0.5000"),
                contradicted_weight_ratio=Decimal("0.0000"),
                subclaim_counts={"SUPPORTED": 1, "INSUFFICIENT": 1},
                limitations=["Subclaim weights fixed at 1.0 in v1."],
                aggregation_version="agg-1.0.0",
            )
        )
        db.add(
            Run(
                claim_id=claim.id,
                stage=1,
                attempt=1,
                status="complete",
                tokens_in=120,
                tokens_out=64,
                cost_usd=Decimal("0.000120"),
            )
        )
        db.commit()

        db.refresh(claim)
        _require(claim.verdict is not None, "claim did not load its verdict")
        _require(
            claim.subclaims[0].verdict is not None,
            "subclaim did not load its verdict",
        )
        _require(len(claim.runs) == 1, "claim did not load its runs")
        print("[6/8] one-to-one verdicts and run log load correctly")

        # --- 7. delete the claim -----------------------------------------
        db.delete(claim)
        db.commit()
        print("[7/8] deleted claim")

        # --- 8. verify the cascade ---------------------------------------
        remaining = {
            "claims": _count(db, Claim),
            "subclaims": _count(db, Subclaim),
            "sources": _count(db, Source),
            "passages": _count(db, Passage),
            "evidence_links": _count(db, EvidenceLink),
            "subclaim_verdicts": _count(db, SubclaimVerdict),
            "verdicts": _count(db, Verdict),
            "runs": _count(db, Run),
        }
        leftovers = {k: v for k, v in remaining.items() if v != 0}
        _require(not leftovers, f"cascade left rows behind: {leftovers}")
        print("[8/8] cascade deleted every descendant row")

        print("\nROUND TRIP OK — schema, ORM and cascade all verified.")
        return 0

    except CheckFailed as exc:
        print(f"\nROUND TRIP FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        print(f"\nROUND TRIP ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        # Leave the database exactly as we found it, even on failure.
        if claim_id is not None:
            try:
                leftover = db.scalars(select(Claim).where(Claim.id == claim_id)).one_or_none()
                if leftover is not None:
                    db.delete(leftover)
                    db.commit()
            except Exception:  # noqa: BLE001 - cleanup is best effort
                db.rollback()
        db.close()


if __name__ == "__main__":
    raise SystemExit(run())
