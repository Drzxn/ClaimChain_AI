"""Exa adapter tests. No network access; the SDK client is faked."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from app.services.exa import (
    ExaSearchService,
    SearchError,
    SearchNotConfiguredError,
    _parse_date,
)


@dataclass
class FakeResult:
    url: str
    title: str | None = None
    published_date: str | None = None
    author: str | None = None
    score: float | None = None
    text: str | None = None


class FakeExaClient:
    """Stands in for ``exa_py.Exa``."""

    def __init__(self, results=None, *, raises: Exception | None = None):
        self._results = results or []
        self._raises = raises
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self._raises is not None:
            raise self._raises

        @dataclass
        class Response:
            results: list

        return Response(results=self._results)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_normalized_results() -> None:
    client = FakeExaClient(
        [
            FakeResult(
                url="https://example.org/a",
                title="A title",
                published_date="2023-04-12",
                author="An Author",
                score=0.9,
            )
        ]
    )
    service = ExaSearchService(api_key="k", client=client)
    results = service.search("https encryption")

    assert len(results) == 1
    assert results[0].url == "https://example.org/a"
    assert results[0].title == "A title"
    assert results[0].published_at == date(2023, 4, 12)
    assert results[0].author == "An Author"


def test_passes_query_and_limit() -> None:
    client = FakeExaClient([FakeResult(url="https://example.org/a")])
    service = ExaSearchService(api_key="k", client=client, max_results=3)
    service.search("some query")

    assert client.calls[0]["query"] == "some query"
    assert client.calls[0]["num_results"] == 3


def test_bounded_result_count() -> None:
    """More results than the limit must be truncated."""
    client = FakeExaClient(
        [FakeResult(url=f"https://example.org/{i}") for i in range(20)]
    )
    service = ExaSearchService(api_key="k", client=client, max_results=5)
    assert len(service.search("q")) == 5


def test_skips_results_without_url() -> None:
    client = FakeExaClient(
        [FakeResult(url=""), FakeResult(url="https://example.org/ok")]
    )
    service = ExaSearchService(api_key="k", client=client)
    results = service.search("q")
    assert len(results) == 1
    assert results[0].url == "https://example.org/ok"


def test_empty_query_returns_empty_without_calling() -> None:
    client = FakeExaClient([FakeResult(url="https://example.org/a")])
    service = ExaSearchService(api_key="k", client=client)
    assert service.search("   ") == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_api_failure_raises_search_error() -> None:
    client = FakeExaClient(raises=RuntimeError("boom"))
    service = ExaSearchService(api_key="k", client=client)
    with pytest.raises(SearchError, match="Exa search failed"):
        service.search("q")


def test_error_message_does_not_leak_key() -> None:
    client = FakeExaClient(raises=RuntimeError("auth failed for key SECRET123"))
    service = ExaSearchService(api_key="SECRET123", client=client)
    with pytest.raises(SearchError) as excinfo:
        service.search("q")
    assert "SECRET123" not in str(excinfo.value)


def test_missing_key_raises_not_configured() -> None:
    service = ExaSearchService(api_key="", client=None)
    with pytest.raises(SearchNotConfiguredError):
        service.search("q")


def test_configured_reflects_key_presence() -> None:
    assert ExaSearchService(api_key="k").configured is True
    assert ExaSearchService(api_key="").configured is False


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023-04-12", date(2023, 4, 12)),
        ("2023-04-12T10:30:00Z", date(2023, 4, 12)),
        ("2023-04-12T10:30:00+00:00", date(2023, 4, 12)),
        (None, None),
        ("", None),
        ("not a date", None),
    ],
)
def test_parse_date(value, expected) -> None:
    assert _parse_date(value) == expected
