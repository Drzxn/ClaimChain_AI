"""Static domain-tier table for deterministic source authority.

Authority is a property of the *source*, not of whether its content is true. A
government site can publish something wrong; a blog can publish something right.
Stage 3 judges the passage-to-subclaim relationship. This module only answers
"what kind of publisher is this?".

The table is data, not logic: adding a domain is a one-line change, and the
whole thing is unit-testable without a network call or a model.

Tiers (from the approved technical specification):
    1.0  government / standards bodies
    0.9  peer-reviewed / academic
    0.8  established reference works
    0.7  major news / established technical organisations
    0.5  general web (default)
    0.3  low-signal (content farms, SEO listicles, unknown TLDs)

Matching is by exact host first, then by suffix, so ``docs.python.org`` inherits
the ``python.org`` tier and ``www.nist.gov`` inherits ``nist.gov``.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Tier scores -----------------------------------------------------------

TIER_GOVERNMENT = 1.0
TIER_ACADEMIC = 0.9
TIER_REFERENCE = 0.8
TIER_NEWS = 0.7
TIER_GENERAL = 0.5
TIER_LOW = 0.3

# --- TLD-based defaults ----------------------------------------------------
# Applied when no explicit domain entry matches. A TLD is a weak signal: it
# says who registered the name, not whether the content is good. So these are
# deliberately conservative and are overridden by any explicit entry.

TLD_TIERS: dict[str, float] = {
    "gov": TIER_GOVERNMENT,
    "mil": TIER_GOVERNMENT,
    "edu": TIER_ACADEMIC,
    # Two-label government TLDs. These must be listed explicitly: the matcher
    # checks the last two labels before the last one, and "gov.uk" is not a
    # single-label TLD.
    "gov.uk": TIER_GOVERNMENT,
    "gov.au": TIER_GOVERNMENT,
    "gov.ca": TIER_GOVERNMENT,
    "gov.in": TIER_GOVERNMENT,
    "govt.nz": TIER_GOVERNMENT,
    "gc.ca": TIER_GOVERNMENT,
    "ac.uk": TIER_ACADEMIC,
    "edu.au": TIER_ACADEMIC,
    "ac.jp": TIER_ACADEMIC,
    "edu.sg": TIER_ACADEMIC,
    "ac.in": TIER_ACADEMIC,
    "int": TIER_NEWS,
    # TLDs with a poor reputation for substantive content.
    "xyz": TIER_LOW,
    "top": TIER_LOW,
    "click": TIER_LOW,
    "buzz": TIER_LOW,
    "loan": TIER_LOW,
    "work": TIER_LOW,
}

# --- Explicit domain entries ----------------------------------------------
# Keyed by registrable-ish host. Subdomains inherit via suffix matching.

DOMAIN_TIERS: dict[str, float] = {
    # Government and intergovernmental
    "nist.gov": TIER_GOVERNMENT,
    "nasa.gov": TIER_GOVERNMENT,
    "nih.gov": TIER_GOVERNMENT,
    "cdc.gov": TIER_GOVERNMENT,
    "nsa.gov": TIER_GOVERNMENT,
    "cisa.gov": TIER_GOVERNMENT,
    "europa.eu": TIER_GOVERNMENT,
    "who.int": TIER_GOVERNMENT,
    "un.org": TIER_GOVERNMENT,
    # Standards bodies
    "ietf.org": TIER_GOVERNMENT,
    "rfc-editor.org": TIER_GOVERNMENT,
    "w3.org": TIER_GOVERNMENT,
    "iso.org": TIER_GOVERNMENT,
    "ieee.org": TIER_GOVERNMENT,
    "unicode.org": TIER_GOVERNMENT,
    "itu.int": TIER_GOVERNMENT,
    # Academic and research
    "arxiv.org": TIER_ACADEMIC,
    "doi.org": TIER_ACADEMIC,
    "nature.com": TIER_ACADEMIC,
    "science.org": TIER_ACADEMIC,
    "sciencedirect.com": TIER_ACADEMIC,
    "springer.com": TIER_ACADEMIC,
    "acm.org": TIER_ACADEMIC,
    "pubmed.ncbi.nlm.nih.gov": TIER_ACADEMIC,
    "semanticscholar.org": TIER_ACADEMIC,
    "jstor.org": TIER_ACADEMIC,
    "plos.org": TIER_ACADEMIC,
    # Reference works
    "britannica.com": TIER_REFERENCE,
    "wikipedia.org": TIER_REFERENCE,
    "merriam-webster.com": TIER_REFERENCE,
    "oxfordreference.com": TIER_REFERENCE,
    # Major news and established technical organisations
    "reuters.com": TIER_NEWS,
    "apnews.com": TIER_NEWS,
    "bbc.com": TIER_NEWS,
    "bbc.co.uk": TIER_NEWS,
    "npr.org": TIER_NEWS,
    "nytimes.com": TIER_NEWS,
    "washingtonpost.com": TIER_NEWS,
    "theguardian.com": TIER_NEWS,
    "economist.com": TIER_NEWS,
    "owasp.org": TIER_NEWS,
    "mozilla.org": TIER_NEWS,
    "python.org": TIER_NEWS,
    "kernel.org": TIER_NEWS,
    "apache.org": TIER_NEWS,
    "microsoft.com": TIER_NEWS,
    "google.com": TIER_NEWS,
    "apple.com": TIER_NEWS,
    "cloudflare.com": TIER_NEWS,
    "aws.amazon.com": TIER_NEWS,
    "developer.mozilla.org": TIER_NEWS,
    # Low-signal
    "medium.com": TIER_LOW,
    "blogspot.com": TIER_LOW,
    "wordpress.com": TIER_LOW,
    "tumblr.com": TIER_LOW,
    "quora.com": TIER_LOW,
    "answers.com": TIER_LOW,
    "ehow.com": TIER_LOW,
    "wikihow.com": TIER_LOW,
    "buzzfeed.com": TIER_LOW,
}

# --- Publisher-name hints --------------------------------------------------
# Used when the domain is unknown but the publisher string is informative.
# Matched case-insensitively as substrings.

PUBLISHER_HINTS: tuple[tuple[str, float], ...] = (
    ("national institute of standards", TIER_GOVERNMENT),
    ("world health organization", TIER_GOVERNMENT),
    ("european commission", TIER_GOVERNMENT),
    ("internet engineering task force", TIER_GOVERNMENT),
    ("world wide web consortium", TIER_GOVERNMENT),
    ("international organization for standardization", TIER_GOVERNMENT),
    ("university", TIER_ACADEMIC),
    ("institute of technology", TIER_ACADEMIC),
    ("peer-reviewed", TIER_ACADEMIC),
    ("journal", TIER_ACADEMIC),
    ("encyclopedia", TIER_REFERENCE),
    ("press", TIER_NEWS),
    ("news", TIER_NEWS),
    ("times", TIER_NEWS),
    ("post", TIER_NEWS),
    ("blog", TIER_LOW),
)


@dataclass(frozen=True, slots=True)
class AuthorityLookup:
    """Result of an authority lookup, with the reason it scored that way."""

    score: float
    tier_label: str
    matched: str


def _host_of(domain: str) -> str:
    """Normalize a domain string for lookup."""
    host = (domain or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip(".")


def _match_domain(host: str) -> tuple[str, float] | None:
    """Find the most specific explicit domain entry for a host."""
    if not host:
        return None

    if host in DOMAIN_TIERS:
        return host, DOMAIN_TIERS[host]

    # Suffix match, longest first, so docs.python.org beats python.org only if
    # a more specific entry exists; otherwise the parent entry applies.
    best: tuple[str, float] | None = None
    for candidate, score in DOMAIN_TIERS.items():
        if host.endswith("." + candidate):
            if best is None or len(candidate) > len(best[0]):
                best = (candidate, score)
    return best


def _match_tld(host: str) -> tuple[str, float] | None:
    """Fall back to a TLD-based tier."""
    if not host:
        return None

    parts = host.split(".")
    if len(parts) < 2:
        return None

    # Two-label TLDs first (ac.uk, edu.au), then single-label.
    if len(parts) >= 3:
        two_label = ".".join(parts[-2:])
        if two_label in TLD_TIERS:
            return two_label, TLD_TIERS[two_label]

    tld = parts[-1]
    if tld in TLD_TIERS:
        return tld, TLD_TIERS[tld]

    return None


def _match_publisher(publisher: str | None) -> tuple[str, float] | None:
    """Last-resort match on the publisher name."""
    if not publisher:
        return None
    lowered = publisher.lower()
    for hint, score in PUBLISHER_HINTS:
        if hint in lowered:
            return hint, score
    return None


def _tier_label(score: float) -> str:
    """Human-readable tier name for the rationale."""
    if score >= TIER_GOVERNMENT:
        return "government/standards"
    if score >= TIER_ACADEMIC:
        return "academic/research"
    if score >= TIER_REFERENCE:
        return "established reference"
    if score >= TIER_NEWS:
        return "established organisation/news"
    if score >= TIER_GENERAL:
        return "general web"
    return "low-signal"


def lookup_authority(domain: str, publisher: str | None = None) -> AuthorityLookup:
    """Deterministic authority score for a source.

    Order of precedence: explicit domain entry, then TLD, then publisher hint,
    then the general-web default. Every path returns a bounded score and a
    reason, so the rationale can explain itself.

    Args:
        domain: the source's host, e.g. ``nist.gov``.
        publisher: optional publisher name, used only if the domain is unknown.

    Returns:
        AuthorityLookup with a score in [0, 1].
    """
    host = _host_of(domain)

    domain_match = _match_domain(host)
    if domain_match is not None:
        matched, score = domain_match
        return AuthorityLookup(
            score=score,
            tier_label=_tier_label(score),
            matched=f"domain {matched}",
        )

    tld_match = _match_tld(host)
    if tld_match is not None:
        matched, score = tld_match
        return AuthorityLookup(
            score=score,
            tier_label=_tier_label(score),
            matched=f"TLD .{matched}",
        )

    publisher_match = _match_publisher(publisher)
    if publisher_match is not None:
        matched, score = publisher_match
        return AuthorityLookup(
            score=score,
            tier_label=_tier_label(score),
            matched=f"publisher name contains {matched!r}",
        )

    return AuthorityLookup(
        score=TIER_GENERAL,
        tier_label=_tier_label(TIER_GENERAL),
        matched="no tier match; general-web default",
    )


__all__ = [
    "DOMAIN_TIERS",
    "PUBLISHER_HINTS",
    "TLD_TIERS",
    "TIER_ACADEMIC",
    "TIER_GENERAL",
    "TIER_GOVERNMENT",
    "TIER_LOW",
    "TIER_NEWS",
    "TIER_REFERENCE",
    "AuthorityLookup",
    "lookup_authority",
]
