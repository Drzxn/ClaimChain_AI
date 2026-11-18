"""Deterministic fixtures for the canonical claim.

Fixture mode exists so the pipeline can be exercised without Exa, without the
network, and without an API key. It is used by tests and by offline demos.

Everything here is clearly synthetic: the URLs use ``example.org`` /
``example.com``, which are reserved by RFC 2606 and can never resolve to a real
site. A fixture run is therefore never mistakable for a live retrieval result,
and the run is flagged ``cache_hit = true`` with
``fixture_mode`` recorded in the run error field so the provenance is visible.

The page bodies are written to actually contain the sentences the canonical
claim decomposes into, so the relevance prefilter and later stages have
something real to work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CANONICAL_CLAIM = "Using HTTPS makes a website completely secure."


@dataclass(frozen=True, slots=True)
class FixturePage:
    """One synthetic source: search metadata plus a page body."""

    url: str
    title: str
    publisher: str
    published_at: str | None
    html: str


def _html(title: str, paragraphs: list[str]) -> str:
    """Build a small, realistic HTML document around some paragraphs."""
    body = "\n".join(f"    <p>{p}</p>" for p in paragraphs)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>body {{ font-family: sans-serif; }}</style>
  <script>window.analytics = {{ track: function() {{}} }};</script>
</head>
<body>
  <header><nav><a href="/">Home</a> <a href="/docs">Docs</a></nav></header>
  <main>
    <h1>{title}</h1>
{body}
  </main>
  <footer><p>&copy; Example Organisation. All rights reserved.</p></footer>
</body>
</html>"""


FIXTURE_PAGES: tuple[FixturePage, ...] = (
    FixturePage(
        url="https://example.org/https-explained",
        title="How HTTPS Works",
        publisher="Example Organisation",
        published_at="2023-04-12",
        html=_html(
            "How HTTPS Works",
            [
                "HTTPS is HTTP carried over TLS. It encrypts data in transit "
                "between the browser and the server, so an observer on the "
                "network cannot read the request or the response.",
                "Because the traffic is encrypted, HTTPS protects against some "
                "classes of network attack, including passive eavesdropping and "
                "simple man-in-the-middle interception on untrusted networks.",
                "HTTPS does not make a website completely secure. It does not "
                "prevent cross-site scripting, SQL injection, broken access "
                "control, or vulnerabilities in the application itself. A site "
                "can serve malicious content over a perfectly valid TLS "
                "connection.",
                "Security therefore requires correct application design, input "
                "validation, patching, and access control in addition to "
                "transport encryption.",
            ],
        ),
    ),
    FixturePage(
        url="https://example.com/security/transport-encryption",
        title="Transport Encryption and Web Security",
        publisher="Example Security Institute",
        published_at="2022-11-03",
        html=_html(
            "Transport Encryption and Web Security",
            [
                "Transport layer encryption protects data while it is moving "
                "between two parties. It provides confidentiality and integrity "
                "for that channel only.",
                "Encryption in transit says nothing about what happens at either "
                "end. Data is decrypted at the server, and any compromise there "
                "is outside the protection the transport layer offers.",
                "Common web application attacks such as cross-site scripting and "
                "SQL injection operate at the application layer. Transport "
                "encryption does not detect or block them.",
                "The claim that HTTPS makes a website completely secure is "
                "therefore an overstatement. It secures the channel, not the "
                "application.",
            ],
        ),
    ),
    FixturePage(
        url="https://example.org/standards/tls-overview",
        title="TLS Overview",
        publisher="Example Standards Body",
        published_at=None,
        html=_html(
            "TLS Overview",
            [
                "TLS provides authenticated, encrypted communication between a "
                "client and a server. It is the protocol underlying HTTPS.",
                "TLS authenticates the server to the client using certificates "
                "issued by a trusted certificate authority, which prevents an "
                "attacker from silently impersonating the server.",
                "TLS does not authenticate the content the server chooses to "
                "serve, and it does not constrain what that content does once it "
                "runs in the browser.",
            ],
        ),
    ),
)


# A page whose content duplicates the first one, used to exercise content-level
# deduplication via ``sources.content_hash`` and ``sources.duplicate_of``.
DUPLICATE_PAGE = FixturePage(
    url="https://example.com/mirror/https-explained",
    title="How HTTPS Works (mirror)",
    publisher="Example Mirror",
    published_at="2023-04-12",
    html=FIXTURE_PAGES[0].html,
)


@dataclass(frozen=True, slots=True)
class FixtureSearchHit:
    """A synthetic search result."""

    url: str
    title: str
    published_at: str | None = None
    score: float = 1.0
    snippet: str = ""


def fixture_search_results(query: str) -> list[FixtureSearchHit]:
    """Return synthetic search hits for a query.

    The mapping is deterministic and ignores the query text beyond returning
    the same fixed set, which is what makes fixture runs reproducible. The
    duplicate page is included so dedupe is exercised.
    """
    hits = [
        FixtureSearchHit(
            url=page.url,
            title=page.title,
            published_at=page.published_at,
            score=1.0 - (index * 0.1),
            snippet=page.title,
        )
        for index, page in enumerate(FIXTURE_PAGES)
    ]
    hits.append(
        FixtureSearchHit(
            url=DUPLICATE_PAGE.url,
            title=DUPLICATE_PAGE.title,
            published_at=DUPLICATE_PAGE.published_at,
            score=0.5,
            snippet=DUPLICATE_PAGE.title,
        )
    )
    return hits


def fixture_page_for(url: str) -> FixturePage | None:
    """Look up a fixture page body by URL."""
    for page in (*FIXTURE_PAGES, DUPLICATE_PAGE):
        if page.url == url:
            return page
    return None


def fixture_queries(subclaim_text: str) -> list[str]:
    """Deterministic fallback queries derived from the subclaim text.

    Used both as the fixture-mode query generator and as the production
    fallback when LLM query generation fails. It never invents terms.
    """
    cleaned = " ".join(subclaim_text.split()).strip(" .?!")
    if not cleaned:
        return []
    # One query from the raw text, one narrowed to documentation-style sources.
    return [cleaned, f"{cleaned} explained"]
