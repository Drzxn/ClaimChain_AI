"""Fetcher tests using httpx MockTransport. No network access."""

from __future__ import annotations

import httpx
import pytest

from app.services.fetcher import FetchError, fetch_page

PUBLIC_IP = "93.184.216.34"


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )


def _html_response(request: httpx.Request, body: str = "<html><body><p>Hello world content here.</p></body></html>") -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text=body
    )


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------

def test_fetches_html() -> None:
    client = _client(lambda request: _html_response(request))
    result = fetch_page(
        f"https://{PUBLIC_IP}/page", client=client, resolve_dns=False
    )
    assert result.status == 200
    assert "Hello world" in result.text
    assert result.content_type == "text/html"


def test_fetches_plain_text() -> None:
    client = _client(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, text="plain body text"
        )
    )
    result = fetch_page(f"https://{PUBLIC_IP}/t.txt", client=client, resolve_dns=False)
    assert result.text == "plain body text"
    assert result.content_type == "text/plain"


def test_records_bytes_read() -> None:
    client = _client(lambda request: _html_response(request))
    result = fetch_page(f"https://{PUBLIC_IP}/", client=client, resolve_dns=False)
    assert result.bytes_read > 0


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status_code", [403, 404, 429, 500, 503])
def test_http_errors_raise(status_code: int) -> None:
    client = _client(lambda request: httpx.Response(status_code))
    with pytest.raises(FetchError) as excinfo:
        fetch_page(f"https://{PUBLIC_IP}/", client=client, resolve_dns=False)
    assert excinfo.value.status == status_code


def test_unsupported_content_type_rejected() -> None:
    client = _client(
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4"
        )
    )
    with pytest.raises(FetchError, match="unsupported content type"):
        fetch_page(f"https://{PUBLIC_IP}/doc.pdf", client=client, resolve_dns=False)


def test_oversized_declared_length_rejected() -> None:
    client = _client(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "9999999"},
            content=b"x",
        )
    )
    with pytest.raises(FetchError, match="exceeds cap"):
        fetch_page(
            f"https://{PUBLIC_IP}/", client=client, resolve_dns=False, max_bytes=1000
        )


def test_oversized_streamed_body_rejected() -> None:
    """With no declared length, the streamed cap must still fire."""
    big = ("<html><body>" + ("a" * 5000) + "</body></html>").encode()

    def handler(request):
        # A generator body means httpx cannot set content-length, so only the
        # streaming cap can catch this.
        def chunks():
            for i in range(0, len(big), 512):
                yield big[i : i + 512]

        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=chunks()
        )

    client = _client(handler)
    with pytest.raises(FetchError, match="exceeded cap"):
        fetch_page(
            f"https://{PUBLIC_IP}/", client=client, resolve_dns=False, max_bytes=1000
        )


def test_timeout_raises() -> None:
    def handler(request):
        raise httpx.TimeoutException("timed out")

    client = _client(handler)
    with pytest.raises(FetchError, match="timeout"):
        fetch_page(f"https://{PUBLIC_IP}/", client=client, resolve_dns=False)


def test_transport_error_raises() -> None:
    def handler(request):
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    with pytest.raises(FetchError, match="transport error"):
        fetch_page(f"https://{PUBLIC_IP}/", client=client, resolve_dns=False)


# ---------------------------------------------------------------------------
# SSRF integration with the fetcher
# ---------------------------------------------------------------------------

def test_fetcher_rejects_private_target_before_request() -> None:
    """The guard must fire before any request is made."""
    called = {"count": 0}

    def handler(request):
        called["count"] += 1
        return _html_response(request)

    client = _client(handler)
    with pytest.raises(FetchError, match="SSRF"):
        fetch_page("http://127.0.0.1/admin", client=client, resolve_dns=False)

    assert called["count"] == 0, "no request may be made to a blocked target"


# ---------------------------------------------------------------------------
# Redirect handling
# ---------------------------------------------------------------------------

def test_follows_safe_redirect() -> None:
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": f"https://{PUBLIC_IP}/end"})
        return _html_response(request)

    client = _client(handler)
    result = fetch_page(
        f"https://{PUBLIC_IP}/start", client=client, resolve_dns=False
    )
    assert result.url.endswith("/end")
    assert len(result.redirects) == 1


def test_redirect_to_private_address_is_blocked() -> None:
    """A public URL that redirects inward must be stopped at the hop."""

    def handler(request):
        if request.url.host == PUBLIC_IP:
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        return _html_response(request)

    client = _client(handler)
    with pytest.raises(FetchError, match="SSRF"):
        fetch_page(f"https://{PUBLIC_IP}/start", client=client, resolve_dns=False)


def test_redirect_to_metadata_endpoint_is_blocked() -> None:
    def handler(request):
        if request.url.host == PUBLIC_IP:
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )
        return _html_response(request)

    client = _client(handler)
    with pytest.raises(FetchError, match="SSRF"):
        fetch_page(f"https://{PUBLIC_IP}/", client=client, resolve_dns=False)


def test_redirect_loop_is_bounded() -> None:
    def handler(request):
        return httpx.Response(302, headers={"location": f"https://{PUBLIC_IP}/loop"})

    client = _client(handler)
    with pytest.raises(FetchError, match="redirect"):
        fetch_page(
            f"https://{PUBLIC_IP}/loop",
            client=client,
            resolve_dns=False,
            max_redirects=3,
        )


def test_redirect_without_location_raises() -> None:
    client = _client(lambda request: httpx.Response(302))
    with pytest.raises(FetchError, match="Location"):
        fetch_page(f"https://{PUBLIC_IP}/", client=client, resolve_dns=False)
