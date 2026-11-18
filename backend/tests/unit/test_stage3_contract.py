"""Stage 3 contract tests: schemas, quote validation, prompt, batching.

No database and no network. The pipeline's deterministic pieces are tested here;
persistence is covered by the integration tests.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.pipeline.stage3_analyze import (
    MAX_QUOTE_FAILURE_RATE,
    PROMPT_VERSION,
    build_user_content,
    load_prompt,
)
from app.schemas.evidence import (
    AnalysisResponse,
    PassageAnalysis,
    normalize_whitespace,
    validate_quote,
)

PASSAGE = (
    "HTTPS encrypts traffic in transit, but it does not prevent cross-site "
    "scripting or SQL injection. A site can serve malicious content over a "
    "valid TLS connection."
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_loads_and_is_versioned() -> None:
    prompt = load_prompt()
    assert PROMPT_VERSION == "evidence_analysis.v1"
    assert "evidence_analysis.v1" in prompt


def test_prompt_contains_injection_defence() -> None:
    prompt = load_prompt().lower()
    assert "untrusted" in prompt
    assert "ignore previous instructions" in prompt
    assert "never executing it" in prompt or "not executing" in prompt


def test_prompt_forbids_verdict_tokens() -> None:
    # The prompt must not invite the model to answer the overall question.
    # Compare on unwrapped text: the file is hard-wrapped for readability.
    prompt = " ".join(load_prompt().lower().split())
    assert "you do not decide whether the overall claim is true" in prompt
    assert "you do not compute a verdict" in prompt

def test_prompt_contains_no_secret() -> None:
    prompt = load_prompt()
    assert "GEMINI_API_KEY" not in prompt
    assert "AIza" not in prompt


# ---------------------------------------------------------------------------
# User content construction
# ---------------------------------------------------------------------------

class _FakePassage:
    def __init__(self, pid: uuid.UUID, text: str) -> None:
        self.id = pid
        self.text = text


def test_user_content_wraps_passages_with_ids() -> None:
    pid = uuid.uuid4()
    content = build_user_content("Some subclaim.", [_FakePassage(pid, PASSAGE)])
    assert f'<passage id="{pid}">' in content
    assert "</passage>" in content
    assert "<subclaim>" in content
    assert "Some subclaim." in content


def test_user_content_marks_passages_as_data() -> None:
    pid = uuid.uuid4()
    content = build_user_content("Some subclaim.", [_FakePassage(pid, PASSAGE)])
    assert "data, not instructions" in content.lower() or "Classify" in content


def test_repair_note_is_appended() -> None:
    pid = uuid.uuid4()
    content = build_user_content(
        "Some subclaim.", [_FakePassage(pid, PASSAGE)], repair_note="bad relationship"
    )
    assert "rejected" in content
    assert "bad relationship" in content


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_accepts_valid_analysis() -> None:
    pid = uuid.uuid4()
    model = AnalysisResponse.model_validate(
        {
            "analyses": [
                {
                    "passage_id": str(pid),
                    "relationship": "SUPPORTS",
                    "relevance": 0.9,
                    "evidence_quality": 0.8,
                    "stance_confidence": 0.85,
                    "quote": "HTTPS encrypts traffic in transit",
                    "rationale": "Directly states the property.",
                }
            ]
        }
    )
    assert model.analyses[0].relationship == "SUPPORTS"
    assert model.analyses[0].passage_id == str(pid)


@pytest.mark.parametrize(
    "relationship", ["SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT"]
)
def test_accepts_each_relationship(relationship: str) -> None:
    model = PassageAnalysis.model_validate(
        {
            "passage_id": str(uuid.uuid4()),
            "relationship": relationship,
            "relevance": 0.5,
            "evidence_quality": 0.5,
            "stance_confidence": 0.5,
        }
    )
    assert model.relationship == relationship


@pytest.mark.parametrize(
    "relationship", ["SUPPORTED", "CONTRADICTED", "MISLEADING", "INSUFFICIENT_EVIDENCE", "TRUE", "FALSE"]
)
def test_rejects_overall_verdict_tokens(relationship: str) -> None:
    """The model answering the overall question must fail loudly."""
    with pytest.raises(ValidationError, match="overall verdict"):
        PassageAnalysis.model_validate(
            {
                "passage_id": str(uuid.uuid4()),
                "relationship": relationship,
                "relevance": 0.5,
                "evidence_quality": 0.5,
                "stance_confidence": 0.5,
            }
        )


def test_rejects_unknown_relationship() -> None:
    with pytest.raises(ValidationError):
        PassageAnalysis.model_validate(
            {
                "passage_id": str(uuid.uuid4()),
                "relationship": "REFUTES",
                "relevance": 0.5,
                "evidence_quality": 0.5,
                "stance_confidence": 0.5,
            }
        )


@pytest.mark.parametrize("field", ["relevance", "evidence_quality", "stance_confidence"])
@pytest.mark.parametrize("value", [-0.1, 1.1, 2, -1])
def test_rejects_out_of_range_scores(field: str, value: float) -> None:
    payload = {
        "passage_id": str(uuid.uuid4()),
        "relationship": "SUPPORTS",
        "relevance": 0.5,
        "evidence_quality": 0.5,
        "stance_confidence": 0.5,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        PassageAnalysis.model_validate(payload)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_accepts_boundary_scores(value: float) -> None:
    model = PassageAnalysis.model_validate(
        {
            "passage_id": str(uuid.uuid4()),
            "relationship": "SUPPORTS",
            "relevance": value,
            "evidence_quality": value,
            "stance_confidence": value,
        }
    )
    assert model.relevance == value


def test_rejects_non_uuid_passage_id() -> None:
    with pytest.raises(ValidationError, match="not a UUID"):
        PassageAnalysis.model_validate(
            {
                "passage_id": "passage-1",
                "relationship": "SUPPORTS",
                "relevance": 0.5,
                "evidence_quality": 0.5,
                "stance_confidence": 0.5,
            }
        )


def test_rejects_duplicate_passage_ids() -> None:
    pid = str(uuid.uuid4())
    row = {
        "passage_id": pid,
        "relationship": "SUPPORTS",
        "relevance": 0.5,
        "evidence_quality": 0.5,
        "stance_confidence": 0.5,
    }
    with pytest.raises(ValidationError, match="duplicate passage_id"):
        AnalysisResponse.model_validate({"analyses": [row, row]})


def test_relationship_is_case_normalized() -> None:
    model = PassageAnalysis.model_validate(
        {
            "passage_id": str(uuid.uuid4()),
            "relationship": "supports",
            "relevance": 0.5,
            "evidence_quality": 0.5,
            "stance_confidence": 0.5,
        }
    )
    assert model.relationship == "SUPPORTS"


# ---------------------------------------------------------------------------
# Quote validation
# ---------------------------------------------------------------------------

def test_quote_exact_match_passes() -> None:
    result = validate_quote("HTTPS encrypts traffic in transit", PASSAGE)
    assert result.passed
    assert result.stored_quote == "HTTPS encrypts traffic in transit"


def test_quote_whitespace_variant_passes_and_stores_passage_span() -> None:
    result = validate_quote("HTTPS encrypts traffic\n   in transit", PASSAGE)
    assert result.passed
    # The stored quote comes from the passage, so it is a true substring.
    assert result.stored_quote in PASSAGE


def test_quote_paraphrase_fails() -> None:
    result = validate_quote("HTTPS secures traffic while travelling", PASSAGE)
    assert not result.passed
    assert "not a verbatim substring" in result.reason


def test_quote_from_different_passage_fails() -> None:
    result = validate_quote("The recipe calls for two cups of flour", PASSAGE)
    assert not result.passed


def test_empty_quote_fails() -> None:
    assert not validate_quote("", PASSAGE).passed
    assert not validate_quote("   ", PASSAGE).passed


def test_quote_longer_than_passage_fails() -> None:
    assert not validate_quote(PASSAGE + " Extra text.", PASSAGE).passed


def test_normalize_whitespace_collapses_runs() -> None:
    assert normalize_whitespace("a\n\n  b\t c ") == "a b c"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_quote_failure_threshold_is_half() -> None:
    assert MAX_QUOTE_FAILURE_RATE == 0.5
