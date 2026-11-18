"""Exa search adapter.

One provider, one adapter, one place that knows Exa's response shape. The rest
of the pipeline deals only in ``SearchResult``.

The API key is read from settings and handed to the SDK. It is never logged,
never placed in an exception message, and never returned to a caller.

Exa snippets are **not** evidence. They are used only to rank and to populate
title/date metadata. Passages always come from the fetched page, because a
snippet cannot be verified against itself and Stage 3 requires verbatim quotes
from real page text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_RESULTS_PER_QUERY = 5


class SearchError(RuntimeError):
    """The search provider failed. Carries a short, non-secret reason."""


class SearchNotConfiguredError(SearchError):
    """No search API key is configured."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A normalized search hit."""

    url: str
    title: str | None = None
    published_at: date | None = None
    author: str | None = None
    score: float | None = None
    snippet: str | None = None
    query: str = ""


def _parse_date(value: object) -> date | None:
    """Parse Exa's published date, which may be a date or an ISO datetime."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Try a full ISO timestamp first, then a bare date.
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


class ExaSearchService:
    """Wraps the Exa SDK behind a small, testable interface."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: object | None = None,
        max_results: int = MAX_RESULTS_PER_QUERY,
    ) -> None:
        settings = get_settings()
        self._api_key = (
            api_key if api_key is not None else settings.exa_api_key
        ).strip()
        self._client = client
        self.max_results = max_results

    @property
    def configured(self) -> bool:
        """Whether a key is present. Never reveals the key."""
        return bool(self._api_key)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise SearchNotConfiguredError(
                "EXA_API_KEY is not set. Add it to .env, or enable fixture mode."
            )
        try:
            from exa_py import Exa
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise SearchError("exa-py is not installed") from exc

        self._client = Exa(api_key=self._api_key)
        return self._client

    def search(self, query: str, *, num_results: int | None = None) -> list[SearchResult]:
        """Run one query and return normalized results.

        Raises:
            SearchNotConfiguredError: no key and no injected client.
            SearchError: the provider call failed.
        """
        if not query or not query.strip():
            return []

        limit = num_results or self.max_results
        client = self._ensure_client()

        try:
            response = client.search(
                query.strip(),
                num_results=limit,
                type="auto",
                contents={"text": False},  # snippets only; pages are fetched later
            )
        except Exception as exc:  # noqa: BLE001 - normalise provider failures
            # Deliberately excludes the key and the request body.
            raise SearchError(f"Exa search failed: {type(exc).__name__}") from exc

        raw_results = getattr(response, "results", None) or []
        results: list[SearchResult] = []

        for item in raw_results[:limit]:
            url = getattr(item, "url", None)
            if not url:
                continue
            results.append(
                SearchResult(
                    url=str(url),
                    title=getattr(item, "title", None),
                    published_at=_parse_date(getattr(item, "published_date", None)),
                    author=getattr(item, "author", None),
                    score=getattr(item, "score", None),
                    snippet=getattr(item, "text", None),
                    query=query,
                )
            )

        return results


def get_search_service() -> ExaSearchService:
    """FastAPI-friendly accessor. Tests override this dependency."""
    return ExaSearchService()
