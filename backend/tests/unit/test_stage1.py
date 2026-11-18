"""Stage 1 contract tests: validation, retry behaviour, weight pinning.

All LLM calls are scripted; nothing here touches the network.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.pipeline.stage1_decompose import (
    PROMPT_VERSION,
    Stage1Error,
    Stage1InputError,
    load_prompt,
    run_stage1,
    validate_claim_text,
)
from pydantic import ValidationError

from app.schemas.claim import (
    FIXED_SUBCLAIM_WEIGHT,
    PersistedSubclaim,
    Stage1Decomposition,
)
from app.services.llm import LLMContractError, LLMNotConfiguredError, LLMTransportError
from tests.fakes import FakeGemini, valid_decomposition

CANONICAL = "Using HTTPS makes a website completely secure."


# ---------------------------------------------------------------------------
# Input validation (no LLM call)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_rejects_empty_or_whitespace(text: str) -> None:
    with pytest.raises(Stage1InputError):
        validate_claim_text(text)


def test_rejects_over_2000_chars() -> None:
    with pytest.raises(Stage1InputError):
        validate_claim_text("x" * 2001)


def test_rejects_text_without_letters() -> None:
    with pytest.raises(Stage1InputError):
        validate_claim_text("12345 !!! ???")


def test_accepts_valid_claim_and_strips() -> None:
    assert validate_claim_text("  " + CANONICAL + "  ") == CANONICAL


def test_accepts_exactly_2000_chars() -> None:
    text = "a" * 2000
    assert len(validate_claim_text(text)) == 2000


def test_invalid_input_makes_no_llm_call() -> None:
    fake = FakeGemini([valid_decomposition()])
    with pytest.raises(Stage1InputError):
        run_stage1("   ", service=fake)
    assert fake.call_count == 0


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_loads_and_is_versioned() -> None:
    prompt = load_prompt()
    assert PROMPT_VERSION == "decompose.v1"
    assert "decompose.v1" in prompt
    assert "<claim>" in prompt
    # Prompt-injection defence must be present in the prompt itself.
    assert "never an instruction" in prompt.lower() or "data" in prompt.lower()


def test_prompt_contains_no_secret() -> None:
    prompt = load_prompt()
    assert "GEMINI_API_KEY" not in prompt
    assert "AIza" not in prompt  # Google API key prefix


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_response_accepted() -> None:
    fake = FakeGemini([valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)

    assert result.response.domain == "technology"
    assert len(result.response.subclaims) == 2
    assert result.response.prompt_version == PROMPT_VERSION
    assert fake.call_count == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "complete"


def test_ordinals_are_sequential_from_one() -> None:
    fake = FakeGemini([valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert [s.ordinal for s in result.response.subclaims] == [1, 2]


def test_claim_is_wrapped_as_data_in_user_content() -> None:
    fake = FakeGemini([valid_decomposition()])
    run_stage1(CANONICAL, service=fake)
    sent = fake.calls[0]["user_content"]
    assert "<claim>" in sent and "</claim>" in sent
    assert CANONICAL in sent


# ---------------------------------------------------------------------------
# Weight pinning â€” the model cannot control stored weight
# ---------------------------------------------------------------------------

def test_model_proposed_weights_are_overwritten_to_one() -> None:
    payload = valid_decomposition(
        subclaims=[
            {"text": "First claim here.", "claim_type": "factual", "weight": 2.5},
            {"text": "Second claim here.", "claim_type": "causal", "weight": 0.4},
            {"text": "Third claim here.", "claim_type": "quantitative", "weight": 0.1},
        ]
    )
    fake = FakeGemini([payload])
    result = run_stage1(CANONICAL, service=fake)

    assert all(s.weight == FIXED_SUBCLAIM_WEIGHT for s in result.response.subclaims)
    assert [s.weight for s in result.response.subclaims] == [
        Decimal("1.0"),
        Decimal("1.0"),
        Decimal("1.0"),
    ]


def test_persisted_subclaim_rejects_non_unit_weight() -> None:
    """The stored form is structurally incapable of holding a non-1.0 weight."""
    with pytest.raises(ValidationError):
        PersistedSubclaim(
            ordinal=1,
            text="Some claim.",
            claim_type="factual",
            weight=Decimal("0.5"),
        )


# ---------------------------------------------------------------------------
# Contract violations -> repair retry
# ---------------------------------------------------------------------------

def test_invalid_domain_triggers_repair_then_succeeds() -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)

    assert fake.call_count == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].status == "failed"
    assert result.attempts[1].status == "complete"
    assert result.response.domain == "technology"


def test_invalid_claim_type_triggers_repair() -> None:
    bad = valid_decomposition(
        subclaims=[{"text": "A claim.", "claim_type": "opinion", "weight": 1.0}]
    )
    fake = FakeGemini([bad, valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 2
    assert result.attempts[0].status == "failed"


def test_duplicate_subclaims_rejected_then_repaired() -> None:
    dupes = valid_decomposition(
        subclaims=[
            {"text": "HTTPS encrypts traffic.", "claim_type": "factual", "weight": 1.0},
            {"text": "https  ENCRYPTS traffic.", "claim_type": "factual", "weight": 1.0},
        ]
    )
    fake = FakeGemini([dupes, valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 2
    assert "duplicate" in (result.attempts[0].error or "").lower()


def test_zero_subclaims_rejected() -> None:
    fake = FakeGemini([valid_decomposition(subclaims=[]), valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 2
    assert result.attempts[0].status == "failed"


def test_more_than_eight_subclaims_rejected() -> None:
    nine = valid_decomposition(
        subclaims=[
            {"text": f"Distinct claim number {i}.", "claim_type": "factual", "weight": 1.0}
            for i in range(9)
        ]
    )
    fake = FakeGemini([nine, valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 2
    assert result.attempts[0].status == "failed"


def test_subclaim_over_300_chars_rejected() -> None:
    long_sub = valid_decomposition(
        subclaims=[{"text": "x" * 301, "claim_type": "factual", "weight": 1.0}]
    )
    fake = FakeGemini([long_sub, valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 2
    assert result.attempts[0].status == "failed"


def test_non_json_response_triggers_repair() -> None:
    fake = FakeGemini(["this is not json", valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 2
    assert result.attempts[1].status == "complete"


# ---------------------------------------------------------------------------
# Retry exhaustion â€” exactly one repair, never a third attempt
# ---------------------------------------------------------------------------

def test_two_failures_raise_and_make_exactly_two_calls() -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, bad])

    with pytest.raises(Stage1Error):
        run_stage1(CANONICAL, service=fake)

    assert fake.call_count == 2, "must not make a third attempt"


def test_repair_prompt_includes_validation_error() -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, valid_decomposition()])
    run_stage1(CANONICAL, service=fake)

    repair_content = fake.calls[1]["user_content"]
    assert "did not satisfy" in repair_content
    assert "Validation error" in repair_content
    assert "<previous_response>" in repair_content


def test_repair_prompt_treats_previous_response_as_data() -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, valid_decomposition()])
    run_stage1(CANONICAL, service=fake)
    assert "data, not instructions" in fake.calls[1]["user_content"]


def test_attempts_are_numbered_one_and_two() -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert [a.attempt for a in result.attempts] == [1, 2]


# ---------------------------------------------------------------------------
# Transport errors are not retried
# ---------------------------------------------------------------------------

def test_transport_error_propagates_without_retry() -> None:
    fake = FakeGemini([LLMTransportError("network down")])
    with pytest.raises(LLMTransportError):
        run_stage1(CANONICAL, service=fake)
    assert fake.call_count == 1


def test_not_configured_propagates() -> None:
    fake = FakeGemini([LLMNotConfiguredError("no key")])
    with pytest.raises(LLMNotConfiguredError):
        run_stage1(CANONICAL, service=fake)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def test_token_counts_are_summed_across_attempts() -> None:
    bad = valid_decomposition(domain="sports")
    fake = FakeGemini([bad, valid_decomposition()], tokens_in=100, tokens_out=50)
    result = run_stage1(CANONICAL, service=fake)

    assert result.attempts[0].tokens_in == 100
    assert result.attempts[1].tokens_in == 100
    assert result.response.tokens_in == 200
    assert result.response.tokens_out == 100


def test_cost_is_zero_not_invented() -> None:
    fake = FakeGemini([valid_decomposition()])
    result = run_stage1(CANONICAL, service=fake)
    assert result.attempts[0].cost_usd == 0.0


# ---------------------------------------------------------------------------
# Schema-level validation
# ---------------------------------------------------------------------------

def test_decomposition_schema_rejects_unknown_domain() -> None:
    with pytest.raises(Exception):
        Stage1Decomposition.model_validate(
            {"domain": "history", "subclaims": [{"text": "x", "claim_type": "factual", "weight": 1.0}]}
        )


def test_decomposition_schema_ignores_extra_top_level_fields() -> None:
    payload = valid_decomposition()
    payload["unexpected"] = "ignored"
    model = Stage1Decomposition.model_validate(payload)
    assert model.domain == "technology"

