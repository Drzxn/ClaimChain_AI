"""Deterministic recency scoring.

Recency answers "how fresh is this source relative to when we read it?", not
"is this source good?". An old source can be excellent evidence for a stable
fact; that judgement belongs to Stage 3 and Stage 4, not here.

The decay is piecewise linear and fully documented, so any score can be
reproduced by hand:

    age_days = (retrieved_at.date() - published_at).days

    published_at is None          -> 0.5   (unknown; neutral, not punished)
    age_days <= full_credit_days  -> 1.0
    age_days <= 1095              -> 1.0 - 0.4 * (age - full) / (1095 - full)
    1095 < age_days               -> max(0.2, 0.6 - 0.4 * (age - 1095) / 1825)

    future publication dates clamp to 1.0

The full-credit window is 365 days by default and 180 days for technology
claims, because technology claims age faster than, say, historical ones. The
window is the only domain-sensitive parameter; the shape of the curve is not.

The floor of 0.2 exists so a genuinely ancient source is still usable as
evidence rather than being zeroed out. A floor of 0 would let the aggregation
silently discard old-but-relevant material.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# --- Constants -------------------------------------------------------------

DEFAULT_FULL_CREDIT_DAYS = 365
TECHNOLOGY_FULL_CREDIT_DAYS = 180

MID_TIER_END_DAYS = 1095  # ~3 years
MID_TIER_START_SCORE = 1.0
MID_TIER_END_SCORE = 0.6

OLD_TIER_END_DAYS = 2920  # ~8 years
OLD_TIER_START_SCORE = 0.6
OLD_TIER_FLOOR = 0.2

UNKNOWN_DATE_SCORE = 0.5

# Domains whose claims age faster.
FAST_MOVING_DOMAINS = frozenset({"technology"})


@dataclass(frozen=True, slots=True)
class RecencyResult:
    """A recency score with the inputs that produced it."""

    score: float
    age_days: int | None
    full_credit_days: int
    reason: str


def full_credit_window(domain: str | None) -> int:
    """Days of full recency credit for a claim domain."""
    if domain and domain.strip().lower() in FAST_MOVING_DOMAINS:
        return TECHNOLOGY_FULL_CREDIT_DAYS
    return DEFAULT_FULL_CREDIT_DAYS


def score_recency(
    published_at: date | None,
    retrieved_at: date,
    *,
    domain: str | None = None,
) -> RecencyResult:
    """Score how recent a source is, on a 0-1 scale.

    Args:
        published_at: publication date, or None when the page did not say.
        retrieved_at: when the page was fetched. This is the reference point,
            not "today", so a score is reproducible from stored data alone.
        domain: claim domain; ``technology`` uses a shorter full-credit window.

    Returns:
        RecencyResult with a score in [0, 1]. Never raises.
    """
    window = full_credit_window(domain)

    if published_at is None:
        return RecencyResult(
            score=UNKNOWN_DATE_SCORE,
            age_days=None,
            full_credit_days=window,
            reason="no publication date; neutral 0.5 applied",
        )

    age_days = (retrieved_at - published_at).days

    # A future publication date is a metadata error, not a reason to reward the
    # source. Clamp to full credit and say so.
    if age_days < 0:
        return RecencyResult(
            score=1.0,
            age_days=age_days,
            full_credit_days=window,
            reason=f"publication date is {-age_days} days in the future; clamped to 1.0",
        )

    if age_days <= window:
        return RecencyResult(
            score=1.0,
            age_days=age_days,
            full_credit_days=window,
            reason=f"{age_days} days old, within the {window}-day full-credit window",
        )

    if age_days <= MID_TIER_END_DAYS:
        span = MID_TIER_END_DAYS - window
        decayed = MID_TIER_START_SCORE - (
            (MID_TIER_START_SCORE - MID_TIER_END_SCORE) * (age_days - window) / span
        )
        return RecencyResult(
            score=round(decayed, 4),
            age_days=age_days,
            full_credit_days=window,
            reason=f"{age_days} days old; linear decay from 1.0 to 0.6",
        )

    decayed = OLD_TIER_START_SCORE - (
        (OLD_TIER_START_SCORE - OLD_TIER_FLOOR)
        * (age_days - MID_TIER_END_DAYS)
        / (OLD_TIER_END_DAYS - MID_TIER_END_DAYS)
    )
    floored = max(OLD_TIER_FLOOR, decayed)

    reason = (
        f"{age_days} days old; decayed toward the {OLD_TIER_FLOOR} floor"
        if floored > OLD_TIER_FLOOR
        else f"{age_days} days old; held at the {OLD_TIER_FLOOR} floor"
    )

    return RecencyResult(
        score=round(floored, 4),
        age_days=age_days,
        full_credit_days=window,
        reason=reason,
    )


__all__ = [
    "DEFAULT_FULL_CREDIT_DAYS",
    "FAST_MOVING_DOMAINS",
    "OLD_TIER_FLOOR",
    "TECHNOLOGY_FULL_CREDIT_DAYS",
    "UNKNOWN_DATE_SCORE",
    "RecencyResult",
    "full_credit_window",
    "score_recency",
]
