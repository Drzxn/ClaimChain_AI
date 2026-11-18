"""Stage 4 pure-aggregation unit tests.

No database, no network, no model. These exercise the deterministic core
directly, including every boundary of the frozen constants.
"""

from __future__ import annotations

import uuid

import pytest

from app.pipeline.stage4_verify import (
    AGGREGATION_VERSION,
    COVERAGE_FLOOR,
    DOMINANCE_MARGIN,
    MIN_LINK_WEIGHT,
    MIN_SUBCLAIM_EVIDENCE,
    MISLEADING_MATERIALITY,
    MISLEADING_SCORE_FLOOR,
    ROUND_DP,
    VERDICT_THRESHOLD,
    LinkContribution,
    aggregate_overall,
    aggregate_subclaim,
    build_contributions,
    link_weight,
)


def sub(ordinal: int = 1, weight: float = 1.0, text: str = "s"):
    return {
        "subclaim_id": uuid.uuid4(),
        "ordinal": ordinal,
        "text": text,
        "weight": weight,
    }


def agg(contributions, **kwargs):
    return aggregate_subclaim(**sub(**kwargs), contributions=contributions)


def counted(relationship: str, weight: float = 1.0) -> LinkContribution:
    return LinkContribution(relationship=relationship, weight=weight, counted=True)


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

def test_frozen_constants_are_exact() -> None:
    """These are the approved agg-1.0.0 values. Changing one is a version bump."""
    assert MIN_LINK_WEIGHT == 0.15
    assert MIN_SUBCLAIM_EVIDENCE == 0.25
    assert DOMINANCE_MARGIN == 0.30
    assert COVERAGE_FLOOR == 0.50
    assert VERDICT_THRESHOLD == 0.60
    assert MISLEADING_SCORE_FLOOR == 0.20
    assert MISLEADING_MATERIALITY == 0.20
    assert ROUND_DP == 4
    assert AGGREGATION_VERSION == "agg-1.0.0"


# ---------------------------------------------------------------------------
# Link weighting
# ---------------------------------------------------------------------------

def test_link_weight_is_the_five_factor_product() -> None:
    w = link_weight(1.0, 1.0, 1.0, 1.0, 1.0)
    assert w == 1.0

    w = link_weight(0.8, 0.9, 0.7, 0.6, 0.5)
    assert w == pytest.approx(0.8 * 0.9 * 0.7 * 0.6 * 0.5)


def test_link_weight_zero_when_any_factor_zero() -> None:
    assert link_weight(0.0, 1.0, 1.0, 1.0, 1.0) == 0.0
    assert link_weight(1.0, 0.0, 1.0, 1.0, 1.0) == 0.0
    assert link_weight(1.0, 1.0, 0.0, 1.0, 1.0) == 0.0


def test_missing_authority_yields_zero_weight() -> None:
    """An unassessed source must not receive artificial authority."""
    assert link_weight(None, 1.0, 1.0, 1.0, 1.0) == 0.0


def test_missing_recency_yields_zero_weight() -> None:
    assert link_weight(1.0, None, 1.0, 1.0, 1.0) == 0.0


def test_link_weight_within_bounds() -> None:
    for a in (0.0, 0.3, 0.7, 1.0):
        for r in (0.0, 0.5, 1.0):
            w = link_weight(a, r, 0.9, 0.8, 0.7)
            assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Minimum link weight
# ---------------------------------------------------------------------------

def test_link_below_threshold_is_excluded() -> None:
    # 0.5 * 0.5 * 0.5 * 0.5 * 0.5 = 0.03125 < 0.15
    contributions = build_contributions([("SUPPORTS", 0.5, 0.5, 0.5, 0.5, 0.5)])
    assert contributions[0].counted is False
    assert "below MIN_LINK_WEIGHT" in contributions[0].reason


def test_link_at_threshold_is_included() -> None:
    # Construct exactly 0.15: 0.5*0.5*0.6*1.0*1.0 = 0.15
    contributions = build_contributions([("SUPPORTS", 0.5, 0.5, 0.6, 1.0, 1.0)])
    assert contributions[0].counted is True


def test_link_above_threshold_is_included() -> None:
    contributions = build_contributions([("SUPPORTS", 1.0, 1.0, 1.0, 1.0, 1.0)])
    assert contributions[0].counted is True


def test_insufficient_relationship_never_counts() -> None:
    """INSUFFICIENT is stored and displayed, but never moves a verdict."""
    contributions = build_contributions([("INSUFFICIENT", 1.0, 1.0, 1.0, 1.0, 1.0)])
    assert contributions[0].counted is False
    assert "INSUFFICIENT" in contributions[0].reason


def test_invalid_relationship_raises() -> None:
    with pytest.raises(Exception, match="invalid stored relationship"):
        build_contributions([("REFUTES", 1.0, 1.0, 1.0, 1.0, 1.0)])


def test_null_scores_excluded_from_aggregation() -> None:
    """A NULL-authority link is dropped, not treated as neutral."""
    contributions = build_contributions([("SUPPORTS", None, None, 1.0, 1.0, 1.0)])
    assert contributions[0].counted is False


# ---------------------------------------------------------------------------
# Subclaim verdicts
# ---------------------------------------------------------------------------

def test_support_only_is_supported() -> None:
    result = agg([counted("SUPPORTS", 1.0)])
    assert result.verdict == "SUPPORTED"
    assert result.score == 1.0


def test_contradiction_only_is_contradicted() -> None:
    result = agg([counted("CONTRADICTS", 1.0)])
    assert result.verdict == "CONTRADICTED"
    assert result.score == -1.0


def test_no_evidence_is_insufficient() -> None:
    result = agg([])
    assert result.verdict == "INSUFFICIENT"
    assert result.coverage == 0.0
    assert result.score == 0.0


def test_below_minimum_evidence_is_insufficient() -> None:
    """T < 0.25 must be INSUFFICIENT, never MIXED."""
    result = agg([counted("SUPPORTS", 0.2)])
    assert result.verdict == "INSUFFICIENT"
    assert result.total_weight == 0.2


def test_exactly_at_minimum_evidence_is_not_insufficient() -> None:
    result = agg([counted("SUPPORTS", MIN_SUBCLAIM_EVIDENCE)])
    assert result.verdict == "SUPPORTED"


def test_mixed_weight_alone_falls_through_to_insufficient() -> None:
    """Documented gap in the frozen agg-1.0.0 rule chain.

    A subclaim whose only contributing evidence is MIXED has T >= 0.25, so it is
    not short of evidence, yet it matches none of the four explicit branches and
    falls to the ``else: INSUFFICIENT`` tail. That is what the approved contract
    specifies, so it is implemented as written rather than "fixed" here. Real
    Stage 3 output rarely produces MIXED-only subclaims, because a MIXED passage
    usually sits alongside SUPPORTS or CONTRADICTS links.
    """
    result = agg([counted("MIXED", 1.0)])
    assert result.verdict == "INSUFFICIENT"
    assert result.total_weight == 1.0
    assert result.score == 0.0


def test_mixed_with_support_is_mixed() -> None:
    result = agg([counted("SUPPORTS", 0.5), counted("MIXED", 0.5)])
    assert result.verdict == "MIXED"


def test_mixed_with_contradiction_is_mixed() -> None:
    result = agg([counted("CONTRADICTS", 0.5), counted("MIXED", 0.5)])
    assert result.verdict == "MIXED"


# ---------------------------------------------------------------------------
# Dominance margin
# ---------------------------------------------------------------------------

def test_strong_support_dominates() -> None:
    """S=0.9, C=0.1 -> margin 0.8 >= 0.30 -> SUPPORTED."""
    result = agg([counted("SUPPORTS", 0.9), counted("CONTRADICTS", 0.1)])
    assert result.verdict == "SUPPORTED"


def test_strong_contradiction_dominates() -> None:
    result = agg([counted("SUPPORTS", 0.1), counted("CONTRADICTS", 0.9)])
    assert result.verdict == "CONTRADICTED"


def test_balanced_evidence_is_mixed() -> None:
    """S=C -> margin 0 < 0.30 -> MIXED."""
    result = agg([counted("SUPPORTS", 0.5), counted("CONTRADICTS", 0.5)])
    assert result.verdict == "MIXED"


def test_just_inside_dominance_margin_is_mixed() -> None:
    """margin 0.29 < 0.30 -> MIXED."""
    result = agg([counted("SUPPORTS", 0.645), counted("CONTRADICTS", 0.355)])
    assert result.verdict == "MIXED"


def test_exactly_at_dominance_margin_is_supported() -> None:
    """margin exactly 0.30 -> SUPPORTED (>= comparison)."""
    result = agg([counted("SUPPORTS", 0.65), counted("CONTRADICTS", 0.35)])
    assert result.verdict == "SUPPORTED"


def test_verdict_is_not_simply_sign_of_score() -> None:
    """A positive score alone must not force SUPPORTED.

    score = (0.55 - 0.45) / 1.0 = +0.10, positive, but margin 0.10 is below
    the dominance margin, so the correct verdict is MIXED.
    """
    result = agg([counted("SUPPORTS", 0.55), counted("CONTRADICTS", 0.45)])
    assert result.score > 0
    assert result.verdict == "MIXED"


# ---------------------------------------------------------------------------
# Score and coverage
# ---------------------------------------------------------------------------

def test_score_formula() -> None:
    result = agg([counted("SUPPORTS", 0.6), counted("CONTRADICTS", 0.2)])
    assert result.score == pytest.approx((0.6 - 0.2) / 0.8, abs=1e-4)


def test_score_includes_mixed_in_denominator() -> None:
    """M contributes to T but not to S or C."""
    result = agg(
        [counted("SUPPORTS", 0.5), counted("CONTRADICTS", 0.25), counted("MIXED", 0.25)]
    )
    assert result.score == pytest.approx((0.5 - 0.25) / 1.0, abs=1e-4)


def test_score_within_bounds() -> None:
    for s, c in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.9, 0.1)):
        result = agg([counted("SUPPORTS", s), counted("CONTRADICTS", c)])
        assert -1.0 <= result.score <= 1.0


def test_coverage_saturates_at_one() -> None:
    result = agg([counted("SUPPORTS", 1.0), counted("SUPPORTS", 1.0)])
    assert result.coverage == 1.0


def test_coverage_is_partial_below_one() -> None:
    result = agg([counted("SUPPORTS", 0.5)])
    assert result.coverage == 0.5


def test_coverage_zero_without_evidence() -> None:
    assert agg([]).coverage == 0.0


def test_link_count_reflects_counted_links() -> None:
    contributions = [
        counted("SUPPORTS", 1.0),
        counted("CONTRADICTS", 1.0),
        LinkContribution(relationship="SUPPORTS", weight=0.01, counted=False),
    ]
    assert agg(contributions).link_count == 2


# ---------------------------------------------------------------------------
# Overall aggregation
# ---------------------------------------------------------------------------

def make_sub(verdict: str, score: float, weight: float = 1.0, ordinal: int = 1):
    """Build a subclaim aggregate with a specific verdict and score.

    A MIXED subclaim is built from balanced support and contradiction, which is
    the realistic shape: a MIXED-only subclaim falls through the frozen rule
    chain to INSUFFICIENT (see the dedicated test for that).
    """
    if verdict == "SUPPORTED":
        contributions = [counted("SUPPORTS", 1.0)]
    elif verdict == "CONTRADICTED":
        contributions = [counted("CONTRADICTS", 1.0)]
    elif verdict == "MIXED":
        contributions = [counted("SUPPORTS", 0.5), counted("CONTRADICTS", 0.5)]
    else:
        contributions = []
    return agg(contributions, ordinal=ordinal, weight=weight)


def test_all_supported_is_supported() -> None:
    overall = aggregate_overall([make_sub("SUPPORTED", 1.0, ordinal=i) for i in (1, 2, 3)])
    assert overall.verdict == "SUPPORTED"
    assert overall.score == 1.0
    assert overall.coverage == 1.0


def test_all_contradicted_is_contradicted() -> None:
    overall = aggregate_overall([make_sub("CONTRADICTED", -1.0, ordinal=i) for i in (1, 2, 3)])
    assert overall.verdict == "CONTRADICTED"
    assert overall.score == -1.0


def test_balanced_mix_is_mixed() -> None:
    overall = aggregate_overall(
        [make_sub("SUPPORTED", 1.0, ordinal=1), make_sub("CONTRADICTED", -1.0, ordinal=2)]
    )
    assert overall.verdict == "MIXED"
    assert overall.score == 0.0


def test_insufficient_coverage_returns_insufficient_evidence() -> None:
    """Only 1 of 3 subclaims evidenced -> coverage 0.3333 < 0.50."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("INSUFFICIENT", 0.0, ordinal=2),
            make_sub("INSUFFICIENT", 0.0, ordinal=3),
        ]
    )
    assert overall.verdict == "INSUFFICIENT_EVIDENCE"
    assert overall.coverage == pytest.approx(1 / 3, abs=1e-4)


def test_coverage_exactly_at_floor_is_not_insufficient() -> None:
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("INSUFFICIENT", 0.0, ordinal=2),
        ]
    )
    assert overall.coverage == 0.5
    assert overall.verdict != "INSUFFICIENT_EVIDENCE"


def test_no_subclaims_returns_insufficient_evidence() -> None:
    overall = aggregate_overall([])
    assert overall.verdict == "INSUFFICIENT_EVIDENCE"
    assert overall.coverage == 0.0
    assert overall.limitations


def test_misleading_condition() -> None:
    """score 0.3333 > 0.20 and ratio 0.3333 >= 0.20 -> MISLEADING."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("SUPPORTED", 1.0, ordinal=2),
            make_sub("CONTRADICTED", -1.0, ordinal=3),
        ]
    )
    assert overall.verdict == "MISLEADING"
    assert overall.score == 0.3333
    assert overall.contradicted_weight_ratio == 0.3333


def test_misleading_takes_precedence_over_supported() -> None:
    """High score but material contradiction still yields MISLEADING."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("SUPPORTED", 1.0, ordinal=2),
            make_sub("SUPPORTED", 1.0, ordinal=3),
            make_sub("SUPPORTED", 1.0, ordinal=4),
            make_sub("CONTRADICTED", -1.0, ordinal=5),
        ]
    )
    # score = (4 - 1)/5 = 0.6, ratio = 1/5 = 0.2
    assert overall.score == 0.6
    assert overall.contradicted_weight_ratio == 0.2
    assert overall.verdict == "MISLEADING"


def test_high_score_without_material_contradiction_is_supported() -> None:
    """5 supported, 1 contradicted gives ratio 0.1667 < 0.20 -> SUPPORTED."""
    overall = aggregate_overall(
        [make_sub("SUPPORTED", 1.0, ordinal=i) for i in range(1, 6)]
        + [make_sub("CONTRADICTED", -1.0, ordinal=6)]
    )
    assert overall.contradicted_weight_ratio == pytest.approx(1 / 6, abs=1e-4)
    assert overall.verdict == "SUPPORTED"


def test_low_score_with_contradiction_is_mixed_not_misleading() -> None:
    """score must exceed 0.20 for MISLEADING; a low positive score is MIXED."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("CONTRADICTED", -1.0, ordinal=2),
            make_sub("CONTRADICTED", -1.0, ordinal=3),
            make_sub("CONTRADICTED", -1.0, ordinal=4),
        ]
    )
    # score = (1-3)/4 = -0.5 -> not > 0.20, and not <= -0.60 -> MIXED
    assert overall.score == -0.5
    assert overall.verdict == "MIXED"


def test_strong_contradiction_is_contradicted() -> None:
    overall = aggregate_overall(
        [make_sub("CONTRADICTED", -1.0, ordinal=i) for i in (1, 2, 3)]
        + [make_sub("SUPPORTED", 1.0, ordinal=4)]
    )
    # score = (1-3)/4 = -0.5 -> MIXED, not CONTRADICTED
    assert overall.verdict == "MIXED"


def test_verdict_threshold_boundary() -> None:
    """score exactly 0.60 with no contradiction -> SUPPORTED."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("SUPPORTED", 1.0, ordinal=2),
            make_sub("SUPPORTED", 1.0, ordinal=3),
            make_sub("MIXED", 0.0, ordinal=4),
            make_sub("MIXED", 0.0, ordinal=5),
        ]
    )
    # score = (1+1+1+0+0)/5 = 0.6
    assert overall.score == 0.6
    assert overall.verdict == "SUPPORTED"


def test_contradicted_weight_ratio_uses_covered_weight() -> None:
    """The denominator is evidenced subclaim weight, not total weight."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("CONTRADICTED", -1.0, ordinal=2),
            make_sub("INSUFFICIENT", 0.0, ordinal=3),
        ]
    )
    # covered = 2, contradicted = 1 -> 0.5
    assert overall.contradicted_weight_ratio == 0.5


def test_subclaim_weights_affect_overall_score() -> None:
    """A weight-2 subclaim moves the aggregate twice as far as a weight-1 one."""
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, weight=2.0, ordinal=1),
            make_sub("CONTRADICTED", -1.0, weight=1.0, ordinal=2),
        ]
    )
    # score = (2*1 + 1*-1)/3 = 0.3333
    assert overall.score == 0.3333


def test_subclaim_counts_are_reported() -> None:
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("SUPPORTED", 1.0, ordinal=2),
            make_sub("CONTRADICTED", -1.0, ordinal=3),
        ]
    )
    assert overall.subclaim_counts == {"SUPPORTED": 2, "CONTRADICTED": 1}


def test_limitations_mention_insufficient_subclaims() -> None:
    overall = aggregate_overall(
        [
            make_sub("SUPPORTED", 1.0, ordinal=1),
            make_sub("INSUFFICIENT", 0.0, ordinal=2),
        ]
    )
    assert any("lacked adequate evidence" in l for l in overall.limitations)


def test_overall_verdict_is_always_one_of_five() -> None:
    valid = {"SUPPORTED", "CONTRADICTED", "MIXED", "INSUFFICIENT_EVIDENCE", "MISLEADING"}
    combos = [
        [make_sub("SUPPORTED", 1.0, ordinal=1)],
        [make_sub("CONTRADICTED", -1.0, ordinal=1)],
        [make_sub("MIXED", 0.0, ordinal=1)],
        [make_sub("INSUFFICIENT", 0.0, ordinal=1)],
        [],
        [make_sub("SUPPORTED", 1.0, ordinal=1), make_sub("CONTRADICTED", -1.0, ordinal=2)],
    ]
    for combo in combos:
        assert aggregate_overall(combo).verdict in valid


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_aggregation_is_deterministic() -> None:
    subs = [
        make_sub("SUPPORTED", 1.0, ordinal=1),
        make_sub("CONTRADICTED", -1.0, ordinal=2),
        make_sub("SUPPORTED", 1.0, ordinal=3),
    ]
    first = aggregate_overall(subs)
    second = aggregate_overall(subs)
    assert (first.verdict, first.score, first.coverage, first.contradicted_weight_ratio) == (
        second.verdict,
        second.score,
        second.coverage,
        second.contradicted_weight_ratio,
    )


def test_no_llm_import_in_aggregation_module() -> None:
    """Structural guarantee: the verdict cannot depend on a model."""
    import inspect

    import app.pipeline.stage4_verify as module

    source = inspect.getsource(module)
    assert "google.genai" not in source
    assert "GeminiService" not in source
    assert "generate_json" not in source
