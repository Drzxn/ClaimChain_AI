"""URL normalization and hashing.

Deterministic, conservative normalization. The goal is to make two spellings of
the same URL collide, without ever changing what the URL *means*.

What is normalized:
  * scheme and host lowercased (they are case-insensitive)
  * default ports removed (``:80`` for http, ``:443`` for https)
  * fragment removed (never sent to the server)
  * known tracking parameters removed
  * remaining query parameters sorted, so ordering cannot create duplicates
  * trailing slash removed from the path, except at the root

What is deliberately NOT normalized:
  * the path's case (``/Docs`` and ``/docs`` can be different resources)
  * percent-encoding (re-encoding risks changing semantics)
  * the query's values (only the ordering and tracking keys change)

``url_hash`` is the SHA-256 of the normalized URL. It is the dedupe key and
backs the ``(claim_id, url_hash)`` unique constraint.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Parameters that identify a campaign, not a resource. Removing them is safe:
# two URLs differing only in these point at the same content.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "utm_reader",
        "utm_social",
        "utm_brand",
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "twclid",
        "ref_src",
        "ref_url",
        "yclid",
        "_hsenc",
        "_hsmi",
        "vero_id",
        "wickedid",
        "oly_anon_id",
        "oly_enc_id",
        "s_cid",
        "spm",
        "scm",
    }
)

DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


class InvalidUrlError(ValueError):
    """The URL cannot be normalized into something fetchable."""


def _strip_trailing_slash(path: str) -> str:
    """Remove a trailing slash, but keep a bare root path as-is."""
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/") or "/"
    return path


def normalize_url(raw_url: str) -> str:
    """Return a normalized absolute URL.

    Raises:
        InvalidUrlError: empty input, missing scheme/host, or a non-HTTP scheme.
    """
    if not raw_url or not raw_url.strip():
        raise InvalidUrlError("empty URL")

    candidate = raw_url.strip()

    # A bare hostname like "example.com/page" is not a valid absolute URL.
    # Guessing a scheme would be inventing information, so reject it.
    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()

    if not scheme:
        raise InvalidUrlError(f"URL has no scheme: {raw_url!r}")
    if scheme not in DEFAULT_PORTS:
        raise InvalidUrlError(f"unsupported scheme {scheme!r}: {raw_url!r}")
    if not parts.hostname:
        raise InvalidUrlError(f"URL has no host: {raw_url!r}")

    host = parts.hostname.lower()

    # Preserve an explicit non-default port; drop the default one.
    port = parts.port
    netloc = host
    if port is not None and port != DEFAULT_PORTS[scheme]:
        netloc = f"{host}:{port}"

    # Drop tracking parameters, sort the rest.
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    query_pairs.sort()
    query = urlencode(query_pairs, doseq=True)

    path = _strip_trailing_slash(parts.path or "/")

    # Fragment is intentionally dropped.
    return urlunsplit((scheme, netloc, path, query, ""))


def url_hash(normalized_url: str) -> str:
    """SHA-256 hex digest of a normalized URL. 64 characters."""
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()


def normalize_and_hash(raw_url: str) -> tuple[str, str]:
    """Convenience: normalize, then hash. Returns ``(url, url_hash)``."""
    normalized = normalize_url(raw_url)
    return normalized, url_hash(normalized)


def extract_domain(normalized_url: str) -> str:
    """Return the registrable-ish host for display and grouping.

    This is the hostname with a leading ``www.`` removed. It is not a true
    registrable domain (no public-suffix list); that is deliberate, since a
    suffix list is a large dependency for a display field.
    """
    host = urlsplit(normalized_url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def content_hash(text: str) -> str:
    """SHA-256 of normalized extracted text, for content-level dedupe."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
