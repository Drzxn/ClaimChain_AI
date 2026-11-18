"""URL normalization and hashing tests."""

from __future__ import annotations

import pytest

from app.services.urls import (
    InvalidUrlError,
    content_hash,
    extract_domain,
    normalize_and_hash,
    normalize_url,
    url_hash,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # scheme and host case
        ("HTTPS://Example.COM/Path", "https://example.com/Path"),
        # trailing slash removed
        ("https://example.com/docs/", "https://example.com/docs"),
        # root path kept
        ("https://example.com/", "https://example.com/"),
        # fragment dropped
        ("https://example.com/page#section", "https://example.com/page"),
        # default port dropped
        ("https://example.com:443/page", "https://example.com/page"),
        ("http://example.com:80/page", "http://example.com/page"),
        # non-default port kept
        ("https://example.com:8443/page", "https://example.com:8443/page"),
        # tracking params stripped
        ("https://example.com/a?utm_source=x&utm_medium=y", "https://example.com/a"),
        ("https://example.com/a?fbclid=abc123", "https://example.com/a"),
        ("https://example.com/a?gclid=zzz&utm_campaign=q", "https://example.com/a"),
        # meaningful params kept and sorted
        ("https://example.com/a?b=2&a=1", "https://example.com/a?a=1&b=2"),
        # mixed: tracking removed, real param kept
        ("https://example.com/a?utm_source=x&id=7", "https://example.com/a?id=7"),
        # path case preserved
        ("https://example.com/Docs", "https://example.com/Docs"),
    ],
)
def test_normalize_url_table(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def test_duplicate_spellings_collide() -> None:
    """The whole point: different spellings of one URL hash identically."""
    variants = [
        "https://example.com/page",
        "https://example.com/page/",
        "https://example.com/page#top",
        "HTTPS://Example.com/page",
        "https://example.com/page?utm_source=newsletter",
        "https://example.com:443/page",
    ]
    hashes = {url_hash(normalize_url(v)) for v in variants}
    assert len(hashes) == 1


def test_query_order_does_not_create_duplicates() -> None:
    a = normalize_url("https://example.com/x?a=1&b=2")
    b = normalize_url("https://example.com/x?b=2&a=1")
    assert a == b


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "example.com/page",  # no scheme
        "ftp://example.com/file",
        "file:///etc/passwd",
        "gopher://example.com/",
        "javascript:alert(1)",
        "https://",  # no host
    ],
)
def test_rejects_invalid_urls(raw: str) -> None:
    with pytest.raises(InvalidUrlError):
        normalize_url(raw)


def test_hash_is_64_hex_chars() -> None:
    _, digest = normalize_and_hash("https://example.com/page")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_is_stable() -> None:
    assert url_hash("https://example.com/a") == url_hash("https://example.com/a")


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.example.com/a", "example.com"),
        ("https://example.com/a", "example.com"),
        ("https://docs.example.co.uk/x", "docs.example.co.uk"),
    ],
)
def test_extract_domain(url: str, expected: str) -> None:
    assert extract_domain(url) == expected


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def test_content_hash_ignores_whitespace_differences() -> None:
    a = content_hash("The quick  brown\n fox")
    b = content_hash("The quick brown fox")
    assert a == b


def test_content_hash_differs_for_different_text() -> None:
    assert content_hash("alpha") != content_hash("beta")
