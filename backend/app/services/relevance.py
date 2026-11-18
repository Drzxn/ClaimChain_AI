"""Deterministic relevance prefilter.

Answers one narrow question: *is this passage plausibly about this subclaim?*
It does not decide SUPPORTS / CONTRADICTS / MIXED / INSUFFICIENT. That is
Stage 3's job, and doing it here would short-circuit the evidence model.

The score is lexical overlap with IDF weighting, computed over the passages of
a single source. No embeddings, no model, no network. It is deliberately cheap
and explainable: you can point at the shared terms and see why a passage
survived.

Design notes:
  * Stopwords are removed so "the" and "is" cannot create false matches.
  * Terms are lightly stemmed (trailing plural/verb endings) so "encrypts" and
    "encrypted" match. This is not a real stemmer and does not need to be.
  * IDF is computed within the candidate set, so a term appearing in every
    passage contributes little. This keeps generic pages from scoring highly.
  * The threshold keeps passages above a percentile, but always keeps a floor
    of passages so a subclaim is never left with nothing purely because the
    page is uniformly mediocre.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Words carrying no topical signal.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am do does did doing done have has had having
    of in on at to from by for with without about into over under again further
    once all any both each few more most other some such no nor not only own
    same so too very can will just should now i you he she it we they them his
    her its our their what which who whom when where why how
    """.split()
)

MIN_TERM_LENGTH = 3
DEFAULT_KEEP_PERCENTILE = 60
DEFAULT_MIN_KEEP = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ScoredPassage:
    """A passage with its prefilter score."""

    passage_id: int  # position within the candidate list, not a DB id
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed, lightly stemmed."""
    words = _WORD_RE.findall(text.lower())
    terms: list[str] = []

    for word in words:
        if len(word) < MIN_TERM_LENGTH or word in STOPWORDS:
            continue
        terms.append(_stem(word))

    return terms


# Terms that must never be stemmed. "https" losing its trailing "s" would
# collapse it into "http", which is a different protocol and, in this project,
# a different claim. Acronyms ending in "s" are protected wholesale.
PROTECTED_TERMS: frozenset[str] = frozenset(
    {
        "https",
        "tls",
        "dns",
        "cdn",
        "aws",
        "ios",
        "macos",
        "https",
        "css",
        "js",
        "gps",
        "sms",
        "cms",
        "api",
        "apis",
    }
)


def _stem(word: str) -> str:
    """Very light suffix stripping so inflections collide.

    Not a linguistic stemmer. It handles the cases that actually matter for
    matching technical prose: plurals and common verb endings. Protected
    acronyms are returned unchanged.
    """
    if word in PROTECTED_TERMS:
        return word
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _idf(term: str, document_frequency: dict[str, int], total: int) -> float:
    """Inverse document frequency with smoothing."""
    df = document_frequency.get(term, 0)
    return math.log((total + 1) / (df + 1)) + 1.0


def score_passages(
    subclaim_text: str, passage_texts: list[str]
) -> list[float]:
    """Score each passage against the subclaim by IDF-weighted overlap.

    Returns one score per passage, in input order. Scores are normalized to
    ``0..1`` by dividing by the best achievable score for this subclaim, so a
    threshold means the same thing regardless of subclaim length.
    """
    if not passage_texts:
        return []

    subclaim_terms = set(tokenize(subclaim_text))
    if not subclaim_terms:
        return [0.0] * len(passage_texts)

    passage_terms = [set(tokenize(text)) for text in passage_texts]

    # Document frequency across the candidate set.
    document_frequency: dict[str, int] = {}
    for terms in passage_terms:
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    total = len(passage_texts)

    # The best possible score: every subclaim term found, weighted by IDF.
    max_possible = sum(
        _idf(term, document_frequency, total) for term in subclaim_terms
    )
    if max_possible <= 0:
        return [0.0] * len(passage_texts)

    scores: list[float] = []
    for terms in passage_terms:
        matched = subclaim_terms & terms
        raw = sum(_idf(term, document_frequency, total) for term in matched)
        scores.append(min(1.0, raw / max_possible))

    return scores


def select_relevant(
    subclaim_text: str,
    passage_texts: list[str],
    *,
    keep_percentile: int = DEFAULT_KEEP_PERCENTILE,
    min_keep: int = DEFAULT_MIN_KEEP,
) -> list[int]:
    """Return the indices of passages worth keeping.

    Keeps passages scoring at or above ``keep_percentile`` of the score
    distribution, but never fewer than ``min_keep`` (or all of them, if fewer
    exist). Passages scoring exactly zero are dropped even when the floor would
    keep them, because zero means no shared content term at all.

    Returns:
        Indices into ``passage_texts``, in ascending order.
    """
    if not passage_texts:
        return []

    scores = score_passages(subclaim_text, passage_texts)

    # Drop total non-matches first: they share no content term with the claim.
    candidates = [i for i, score in enumerate(scores) if score > 0.0]
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda i: scores[i], reverse=True)
    cutoff_count = max(
        min_keep,
        math.ceil(len(ordered) * (100 - keep_percentile) / 100),
    )
    cutoff_count = min(cutoff_count, len(ordered))

    return sorted(ordered[:cutoff_count])
