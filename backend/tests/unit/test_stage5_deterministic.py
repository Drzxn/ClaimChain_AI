"""Stage 5 deterministic unit tests: authority tiers, recency decay, fallbacks.

No database, no network, no model.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.pipeline.stage5_assess import (
    PROMPT_VERSION,
    SOURCE_TYPES,
    SourceAssessment,
    build_rationale,
    guess_source_type,
    load_prompt,
)
from app.services.recency import (
    DEFAULT_FULL_CREDIT_DAYS,
    OLD_TIER_FLOOR,
    TECHNOLOGY_FULL_CREDIT_DAYS,
    UNKNOWN_DATE_SCORE,
    full_credit_window,
    score_recency,
)
from app.services.source_tiers import (
    TIER_ACADEMIC,
    TIER_GENERAL,
    TIER_GOVERNMENT,
    TIER_LOW,
    TIER_NEWS,
    TIER_REFERENCE,
    lookup_authority,
)

REF = date(2026, 11, 18)


# ---------------------------------------------------------------------------
# Authority: tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("nist.gov", TIER_GOVERNMENT),
        ("www.nist.gov", TIER_GOVERNMENT),
        ("ietf.org", TIER_GOVERNMENT),
        ("w3.org", TIER_GOVERNMENT),
        ("foo.gov.uk", TIER_GOVERNMENT),
        ("mit.edu", TIER_ACADEMIC),
        ("arxiv.org", TIER_ACADEMIC),
        ("nature.com", TIER_ACADEMIC),
        ("bar.ac.uk", TIER_ACADEMIC),
        ("wikipedia.org", TIER_REFERENCE),
        ("britannica.com", TIER_REFERENCE),
        ("reuters.com", TIER_NEWS),
        ("bbc.com", TIER_NEWS),
        ("owasp.org", TIER_NEWS),
        ("docs.python.org", TIER_NEWS),
        ("medium.com", TIER_LOW),
        ("unknown.xyz", TIER_LOW),
        ("randomsite.io", TIER_GENERAL),
        ("", TIER_GENERAL),
    ],
)
def test_authority_tiers(domain: str, expected: float) -> None:
    assert lookup_authority(domain).score == expected


def test_subdomain_inherits_parent_tier() -> None:
    assert lookup_authority("docs.python.org").score == lookup_authority("python.org").score


def test_authority_always_within_bounds() -> None:
    for domain in (
        "nist.gov",
        "mit.edu",
        "wikipedia.org",
        "reuters.com",
        "randomsite.io",
        "medium.com",
        "",
        "a.b.c.d.example",
        "....",
    ):
        score = lookup_authority(domain).score
        assert 0.0 <= score <= 1.0, f"{domain} produced {score}"


def test_authority_is_deterministic() -> None:
    first = lookup_authority("nist.gov").score
    second = lookup_authority("nist.gov").score
    assert first == second


def test_unknown_domain_uses_general_default() -> None:
    result = lookup_authority("totally-unknown-domain.example")
    assert result.score == TIER_GENERAL
    assert "general-web default" in result.matched


def test_publisher_hint_used_when_domain_unknown() -> None:
    result = lookup_authority(
        "some-unknown.example", "National Institute of Standards and Technology"
    )
    assert result.score == TIER_GOVERNMENT
    assert "publisher" in result.matched


def test_domain_beats_publisher_hint() -> None:
    """An explicit domain entry must not be overridden by a publisher string."""
    result = lookup_authority("medium.com", "University of Somewhere")
    assert result.score == TIER_LOW


def test_authority_reason_is_populated() -> None:
    result = lookup_authority("nist.gov")
    assert result.matched
    assert result.tier_label


# ---------------------------------------------------------------------------
# Recency: decay curve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("age_days", "expected"),
    [
        (0, 1.0),
        (100, 1.0),
        (365, 1.0),
    ],
)
def test_recent_sources_get_full_credit(age_days: int, expected: float) -> None:
    published = date.fromordinal(REF.toordinal() - age_days)
    assert score_recency(published, REF).score == expected


def test_one_day_past_window_starts_decaying() -> None:
    published = date.fromordinal(REF.toordinal() - 366)
    score = score_recency(published, REF).score
    assert 0.99 < score < 1.0


def test_mid_tier_end_is_06() -> None:
    published = date.fromordinal(REF.toordinal() - 1095)
    assert score_recency(published, REF).score == pytest.approx(0.6, abs=0.001)


def test_very_old_source_hits_floor() -> None:
    published = date.fromordinal(REF.toordinal() - 5000)
    assert score_recency(published, REF).score == OLD_TIER_FLOOR


def test_old_source_decays_between_06_and_floor() -> None:
    published = date.fromordinal(REF.toordinal() - 1500)
    score = score_recency(published, REF).score
    assert OLD_TIER_FLOOR <= score <= 0.6


def test_missing_date_is_neutral() -> None:
    result = score_recency(None, REF)
    assert result.score == UNKNOWN_DATE_SCORE
    assert result.age_days is None
    assert "no publication date" in result.reason


def test_future_date_clamps_to_one() -> None:
    future = date.fromordinal(REF.toordinal() + 30)
    result = score_recency(future, REF)
    assert result.score == 1.0
    assert "future" in result.reason


def test_recency_always_within_bounds() -> None:
    for age in (0, 1, 180, 365, 366, 1095, 1096, 2000, 5000, 20000):
        published = date.fromordinal(REF.toordinal() - age)
        score = score_recency(published, REF).score
        assert 0.0 <= score <= 1.0, f"age {age} produced {score}"


def test_recency_is_deterministic() -> None:
    published = date(2020, 1, 1)
    assert score_recency(published, REF).score == score_recency(published, REF).score


def test_recency_is_monotonic_non_increasing() -> None:
    """Older sources never score higher than newer ones."""
    previous = 1.0
    for age in range(0, 4000, 50):
        published = date.fromordinal(REF.toordinal() - age)
        score = score_recency(published, REF).score
        assert score <= previous + 1e-9, f"age {age} scored {score} > {previous}"
        previous = score


# ---------------------------------------------------------------------------
# Recency: domain sensitivity
# ---------------------------------------------------------------------------

def test_technology_uses_shorter_window() -> None:
    assert full_credit_window("technology") == TECHNOLOGY_FULL_CREDIT_DAYS
    assert full_credit_window("science") == DEFAULT_FULL_CREDIT_DAYS
    assert full_credit_window(None) == DEFAULT_FULL_CREDIT_DAYS


def test_technology_source_ages_faster() -> None:
    """A 200-day-old source is full credit for science but not for technology."""
    published = date.fromordinal(REF.toordinal() - 200)
    tech = score_recency(published, REF, domain="technology").score
    general = score_recency(published, REF, domain="general").score
    assert general == 1.0
    assert tech < 1.0


# ---------------------------------------------------------------------------
# Source-type fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("nist.gov", "primary"),
        ("mit.edu", "secondary"),
        ("wikipedia.org", "reference"),
        ("reuters.com", "news"),
        ("medium.com", "opinion"),
        ("randomsite.io", "unknown"),
    ],
)
def test_guess_source_type(domain: str, expected: str) -> None:
    assert guess_source_type(domain) == expected


def test_guess_source_type_always_valid() -> None:
    for domain in ("nist.gov", "mit.edu", "x.xyz", "", "a.b"):
        assert guess_source_type(domain) in SOURCE_TYPES


# ---------------------------------------------------------------------------
# Rationale
# ---------------------------------------------------------------------------

def test_rationale_names_the_signals() -> None:
    text = build_rationale(
        authority_reason="domain nist.gov",
        tier_label="government/standards",
        recency_reason="14 days old",
        source_type="primary",
        source_type_source="model classification",
        cites_primary=True,
    )
    assert "government/standards" in text
    assert "nist.gov" in text
    assert "14 days old" in text
    assert "primary" in text
    assert "cite primary" in text


def test_rationale_never_claims_correctness() -> None:
    text = build_rationale(
        authority_reason="domain nist.gov",
        tier_label="government/standards",
        recency_reason="recent",
        source_type="primary",
        source_type_source="model classification",
        cites_primary=None,
    ).lower()
    for forbidden in ("correct", "true", "reliable", "trustworthy", "accurate"):
        assert forbidden not in text


def test_rationale_handles_unknown_primary_status() -> None:
    text = build_rationale(
        authority_reason="x",
        tier_label="general web",
        recency_reason="y",
        source_type="unknown",
        source_type_source="deterministic fallback",
        cites_primary=None,
    )
    assert "undetermined" in text


# ---------------------------------------------------------------------------
# LLM contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_accepts_each_source_type(source_type: str) -> None:
    model = SourceAssessment.model_validate(
        {"source_type": source_type, "rationale": "because"}
    )
    assert model.source_type == source_type


def test_rejects_invalid_source_type() -> None:
    with pytest.raises(Exception):
        SourceAssessment.model_validate(
            {"source_type": "blog", "rationale": "because"}
        )


def test_source_type_is_case_normalized() -> None:
    model = SourceAssessment.model_validate(
        {"source_type": "PRIMARY", "rationale": "x"}
    )
    assert model.source_type == "primary"


def test_cites_primary_may_be_null() -> None:
    model = SourceAssessment.model_validate(
        {"source_type": "unknown", "rationale": "x", "cites_primary_sources": None}
    )
    assert model.cites_primary_sources is None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_loads_and_is_versioned() -> None:
    prompt = load_prompt()
    assert PROMPT_VERSION == "source_assessment.v1"
    assert "source_assessment.v1" in prompt


def test_prompt_contains_injection_defence() -> None:
    prompt = load_prompt().lower()
    assert "untrusted" in prompt
    assert "never executing it" in prompt


def test_prompt_forbids_numeric_scoring() -> None:
    prompt = load_prompt().lower()
    assert "do not score its quality numerically" in prompt


def test_prompt_contains_no_secret() -> None:
    prompt = load_prompt()
    assert "GEMINI_API_KEY" not in prompt
    assert "AIza" not in prompt
