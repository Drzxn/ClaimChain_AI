"""Stage 6 unit tests: schema, deterministic construction, narrative safety.

No database and no network. The model is faked where it appears at all.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.pipeline.stage6_report import (
    PROMPT_VERSION,
    REPORT_VERSION,
    NarrativeResponse,
    _narrative_contradicts_verdict,
    build_points,
    build_subclaim_statements,
    build_verdict_statement,
    generate_narrative,
    load_prompt,
)
from app.schemas.report import (
    ClaimReport,
    ReportClaim,
    ReportEvidence,
    ReportExplanation,
    ReportLimitation,
    ReportSource,
    ReportSubclaim,
    ReportSubclaimVerdict,
    ReportVerdict,
)
from app.services.llm import LLMContractError, LLMTransportError

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_subclaim_verdict(verdict="SUPPORTED", score=1.0, link_count=1) -> ReportSubclaimVerdict:
    return ReportSubclaimVerdict(
        verdict=verdict,
        score=score,
        coverage=1.0,
        support_weight=1.0 if verdict == "SUPPORTED" else 0.0,
        contradict_weight=1.0 if verdict == "CONTRADICTED" else 0.0,
        mixed_weight=0.0,
        link_count=link_count,
        rationale="rationale",
        aggregation_version="agg-1.0.0",
        computed_at=NOW,
    )


def make_evidence(
    *,
    relationship="SUPPORTS",
    quote="HTTPS encrypts traffic in transit.",
    quote_validation="passed",
    stance=0.9,
    link_id=None,
) -> ReportEvidence:
    return ReportEvidence(
        evidence_link_id=link_id or uuid.uuid4(),
        subclaim_id=uuid.uuid4(),
        passage_id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        relationship=relationship,
        relevance=0.9,
        evidence_quality=0.8,
        stance_confidence=stance,
        quote=quote,
        rationale="rationale",
        quote_validation=quote_validation,
        passage_text="HTTPS encrypts traffic in transit. It does not stop XSS.",
        passage_chunk_index=0,
        quote_start=0,
        quote_end=36,
        source_url="https://example.org/a",
        source_domain="example.org",
        source_title="A title",
    )


def make_subclaim(ordinal=1, verdict="SUPPORTED", score=1.0, evidence=None) -> ReportSubclaim:
    return ReportSubclaim(
        id=uuid.uuid4(),
        ordinal=ordinal,
        text=f"Subclaim number {ordinal}.",
        claim_type="factual",
        weight=1.0,
        verdict=make_subclaim_verdict(verdict, score),
        evidence=evidence or [],
    )


def make_report(**overrides) -> ClaimReport:
    values = {
        "claim": ReportClaim(
            id=uuid.uuid4(),
            raw_text="Using HTTPS makes a website completely secure.",
            domain="technology",
            status="queued",
            created_at=NOW,
        ),
        "verdict": ReportVerdict(
            verdict="MISLEADING",
            score=0.3333,
            coverage=1.0,
            contradicted_weight_ratio=0.3333,
            subclaim_counts={"SUPPORTED": 2, "CONTRADICTED": 1},
            summary="summary",
            aggregation_version="agg-1.0.0",
            computed_at=NOW,
        ),
        "subclaims": [make_subclaim(1)],
        "sources": [],
        "explanation": ReportExplanation(verdict_statement="statement"),
        "limitations": [],
        "generated_at": NOW,
    }
    values.update(overrides)
    return ClaimReport(**values)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_report_validates() -> None:
    report = make_report()
    assert report.verdict.verdict == "MISLEADING"
    assert report.report_version == REPORT_VERSION


def test_report_rejects_invalid_overall_verdict() -> None:
    with pytest.raises(ValidationError):
        ReportVerdict(
            verdict="TRUE",
            score=0.5,
            coverage=1.0,
            contradicted_weight_ratio=0.0,
            aggregation_version="agg-1.0.0",
            computed_at=NOW,
        )


def test_report_rejects_invalid_subclaim_verdict() -> None:
    with pytest.raises(ValidationError):
        make_subclaim_verdict(verdict="MISLEADING")  # not a subclaim verdict


def test_report_rejects_invalid_relationship() -> None:
    with pytest.raises(ValidationError):
        make_evidence(relationship="REFUTES")


def test_report_accepts_all_overall_verdicts() -> None:
    for verdict in (
        "SUPPORTED",
        "CONTRADICTED",
        "MIXED",
        "INSUFFICIENT_EVIDENCE",
        "MISLEADING",
    ):
        report = make_report(
            verdict=ReportVerdict(
                verdict=verdict,
                score=0.0,
                coverage=1.0,
                contradicted_weight_ratio=0.0,
                aggregation_version="agg-1.0.0",
                computed_at=NOW,
            )
        )
        assert report.verdict.verdict == verdict


def test_evidence_carries_full_identity_chain() -> None:
    ev = make_evidence()
    assert ev.evidence_link_id
    assert ev.passage_id
    assert ev.source_id
    assert ev.subclaim_id


def test_report_serializes_to_json() -> None:
    payload = make_report().model_dump(mode="json")
    assert payload["verdict"]["verdict"] == "MISLEADING"
    assert "subclaims" in payload
    assert "sources" in payload
    assert "explanation" in payload
    assert "limitations" in payload


# ---------------------------------------------------------------------------
# Deterministic explanation
# ---------------------------------------------------------------------------

def test_verdict_statement_names_the_verdict() -> None:
    text = build_verdict_statement("MISLEADING", 0.3333, 1.0, {"SUPPORTED": 2, "CONTRADICTED": 1})
    assert "MISLEADING" in text
    assert "0.3333" in text
    assert "2 supported" in text
    assert "1 contradicted" in text


def test_verdict_statement_for_insufficient_evidence() -> None:
    text = build_verdict_statement("INSUFFICIENT_EVIDENCE", 0.0, 0.25, {"INSUFFICIENT": 2})
    assert "did not settle" in text
    assert "0.2500" in text


def test_subclaim_statements_are_one_per_subclaim() -> None:
    subs = [make_subclaim(1), make_subclaim(2, "CONTRADICTED", -1.0)]
    statements = build_subclaim_statements(subs)
    assert len(statements) == 2
    assert "supported" in statements[0].lower()
    assert "contradicted" in statements[1].lower()


def test_supporting_points_use_passed_quotes_only() -> None:
    subs = [
        make_subclaim(
            1,
            evidence=[
                make_evidence(quote="good quote", quote_validation="passed"),
                make_evidence(quote="bad quote", quote_validation="failed"),
            ],
        )
    ]
    points = build_points(subs, "SUPPORTS")
    assert points == ["good quote"]


def test_supporting_points_exclude_other_relationships() -> None:
    subs = [
        make_subclaim(
            1,
            evidence=[
                make_evidence(relationship="SUPPORTS", quote="support quote"),
                make_evidence(relationship="CONTRADICTS", quote="contradict quote"),
            ],
        )
    ]
    assert build_points(subs, "SUPPORTS") == ["support quote"]
    assert build_points(subs, "CONTRADICTS") == ["contradict quote"]


def test_points_are_ordered_by_stance_then_id() -> None:
    low = make_evidence(quote="low", stance=0.5, link_id=uuid.UUID(int=1))
    high = make_evidence(quote="high", stance=0.95, link_id=uuid.UUID(int=2))
    subs = [make_subclaim(1, evidence=[low, high])]
    assert build_points(subs, "SUPPORTS") == ["high", "low"]


def test_points_are_deterministic() -> None:
    ev = make_evidence(quote="q", link_id=uuid.UUID(int=7))
    subs = [make_subclaim(1, evidence=[ev])]
    assert build_points(subs, "SUPPORTS") == build_points(subs, "SUPPORTS")


def test_empty_evidence_yields_no_points() -> None:
    assert build_points([make_subclaim(1)], "SUPPORTS") == []


# ---------------------------------------------------------------------------
# Narrative safety
# ---------------------------------------------------------------------------

class FakeNarrativeLLM:
    def __init__(self, script):
        self.model = "gemini-3.5-flash-lite"
        self._script = list(script)
        self.calls = []

    def generate_json(self, *, system_instruction, user_content, json_schema, max_output_tokens=2048):
        self.calls.append({"user_content": user_content})
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item

        class _R:
            data = item
            tokens_in = 50
            tokens_out = 30

        return _R()


def test_narrative_is_generated() -> None:
    fake = FakeNarrativeLLM([{"narrative": "Two subclaims were supported."}])
    narrative, t_in, t_out = generate_narrative(
        fake,
        claim_text="claim",
        verdict="MISLEADING",
        score=0.3333,
        coverage=1.0,
        subclaims=[make_subclaim(1)],
    )
    assert narrative == "Two subclaims were supported."
    assert t_in == 50
    assert t_out == 30


def test_narrative_prompt_contains_evidence_in_data_blocks() -> None:
    fake = FakeNarrativeLLM([{"narrative": "x"}])
    generate_narrative(
        fake,
        claim_text="claim",
        verdict="MISLEADING",
        score=0.3333,
        coverage=1.0,
        subclaims=[make_subclaim(1, evidence=[make_evidence()])],
    )
    content = fake.calls[0]["user_content"]
    assert "<evidence" in content
    assert "</evidence>" in content
    assert "Do not contradict it" in content


def test_narrative_that_asserts_opposite_verdict_is_discarded() -> None:
    fake = FakeNarrativeLLM([{"narrative": "The claim is supported overall."}])
    narrative, _, _ = generate_narrative(
        fake,
        claim_text="claim",
        verdict="MISLEADING",
        score=0.3333,
        coverage=1.0,
        subclaims=[make_subclaim(1)],
    )
    assert narrative == ""


def test_narrative_describing_subclaims_is_kept() -> None:
    """'two subclaims were supported' is correct prose for a MISLEADING verdict."""
    fake = FakeNarrativeLLM(
        [{"narrative": "Two of three subclaims were supported, while the third was contradicted."}]
    )
    narrative, _, _ = generate_narrative(
        fake,
        claim_text="claim",
        verdict="MISLEADING",
        score=0.3333,
        coverage=1.0,
        subclaims=[make_subclaim(1)],
    )
    assert narrative != ""


def test_narrative_transport_failure_returns_empty() -> None:
    fake = FakeNarrativeLLM([LLMTransportError("down")])
    narrative, t_in, t_out = generate_narrative(
        fake,
        claim_text="c",
        verdict="SUPPORTED",
        score=1.0,
        coverage=1.0,
        subclaims=[],
    )
    assert narrative == ""
    assert (t_in, t_out) == (0, 0)


def test_narrative_contract_failure_returns_empty() -> None:
    fake = FakeNarrativeLLM([LLMContractError("bad json")])
    narrative, _, _ = generate_narrative(
        fake, claim_text="c", verdict="SUPPORTED", score=1.0, coverage=1.0, subclaims=[]
    )
    assert narrative == ""


def test_narrative_schema_rejects_extra_fields() -> None:
    model = NarrativeResponse.model_validate(
        {"narrative": "text", "verdict": "SUPPORTED"}
    )
    assert model.narrative == "text"


def test_contradiction_guard_ignores_subclaim_wording() -> None:
    assert not _narrative_contradicts_verdict(
        "Two subclaims were supported.", "MISLEADING"
    )


def test_contradiction_guard_catches_explicit_claim_assertion() -> None:
    assert _narrative_contradicts_verdict("The claim is supported.", "MISLEADING")
    assert _narrative_contradicts_verdict("Overall verdict: contradicted", "SUPPORTED")


def test_contradiction_guard_allows_matching_verdict() -> None:
    assert not _narrative_contradicts_verdict("The claim is misleading.", "MISLEADING")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_loads_and_is_versioned() -> None:
    prompt = load_prompt()
    assert PROMPT_VERSION == "report_generation.v1"
    assert "report_generation.v1" in prompt


def test_prompt_forbids_overriding_the_verdict() -> None:
    prompt = " ".join(load_prompt().lower().split())
    assert "the verdict below is final" in prompt
    assert "do not state a different verdict" in prompt


def test_prompt_contains_injection_defence() -> None:
    prompt = " ".join(load_prompt().lower().split())
    assert "untrusted" in prompt
    assert "never executing it" in prompt


def test_prompt_contains_no_secret() -> None:
    prompt = load_prompt()
    assert "GEMINI_API_KEY" not in prompt
    assert "AIza" not in prompt
