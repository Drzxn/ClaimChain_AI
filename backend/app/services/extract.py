"""Readable-text extraction from HTML.

Strips everything that is not prose: scripts, styles, navigation, headers,
footers, sidebars, forms, and embedded media. What remains is normalized text
with paragraph boundaries preserved.

This is deliberately a modest, deterministic extractor rather than a
readability-scoring library. It has no model, no heuristics that vary between
runs, and no dependency beyond BeautifulSoup. For claim verification the goal
is to keep the sentences that could contain evidence and drop the chrome.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Elements whose contents are never readable prose.
STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "object",
    "embed",
    "svg",
    "canvas",
    "form",
    "input",
    "button",
    "select",
    "textarea",
    "nav",
    "header",
    "footer",
    "aside",
    "menu",
    "dialog",
    "template",
)

# Containers that usually hold navigation or boilerplate rather than content.
STRIP_ROLES = ("navigation", "banner", "contentinfo", "complementary", "search")

MIN_EXTRACTED_CHARS = 80

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


class ExtractionError(ValueError):
    """The page yielded no usable text."""


@dataclass(slots=True)
class ExtractedText:
    """Normalized readable text plus a little provenance."""

    text: str
    title: str | None
    char_count: int


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove non-content elements in place."""
    for tag in soup(list(STRIP_TAGS)):
        tag.decompose()

    for role in STRIP_ROLES:
        for tag in soup.find_all(attrs={"role": role}):
            tag.decompose()

    # Common boilerplate class/id fragments. Kept short and explicit.
    for pattern in ("cookie", "consent", "newsletter", "subscribe", "breadcrumb"):
        for tag in soup.find_all(
            attrs={"class": re.compile(pattern, re.IGNORECASE)}
        ):
            tag.decompose()
        for tag in soup.find_all(attrs={"id": re.compile(pattern, re.IGNORECASE)}):
            tag.decompose()


def _normalize(text: str) -> str:
    """Collapse horizontal whitespace, keep paragraph breaks, trim."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def extract_text(html: str, *, content_type: str = "text/html") -> ExtractedText:
    """Extract readable text from an HTML or plain-text document.

    Args:
        html: the raw response body.
        content_type: ``text/html`` (default) or ``text/plain``.

    Returns:
        ExtractedText with normalized text.

    Raises:
        ExtractionError: no usable text was found.
    """
    if content_type.startswith("text/plain"):
        normalized = _normalize(html)
        if len(normalized) < MIN_EXTRACTED_CHARS:
            raise ExtractionError(
                f"plain text too short ({len(normalized)} chars)"
            )
        return ExtractedText(text=normalized, title=None, char_count=len(normalized))

    if not html or not html.strip():
        raise ExtractionError("empty document")

    soup = BeautifulSoup(html, "lxml")

    title: str | None = None
    if soup.title and soup.title.string:
        title = _normalize(soup.title.string) or None

    _strip_noise(soup)

    # Prefer a main/article container when the page marks one; otherwise use
    # the whole body. This avoids returning a nav-heavy page's chrome.
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    # Insert paragraph breaks at block-level boundaries so chunking can snap to
    # sentence and paragraph edges rather than mid-sentence.
    for block in container.find_all(
        ["p", "div", "section", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]
    ):
        block.insert_after("\n")

    raw_text = container.get_text(separator=" ")
    normalized = _normalize(raw_text)

    if len(normalized) < MIN_EXTRACTED_CHARS:
        raise ExtractionError(
            f"extracted text too short ({len(normalized)} chars)"
        )

    return ExtractedText(text=normalized, title=title, char_count=len(normalized))
