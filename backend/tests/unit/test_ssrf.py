"""SSRF protection tests.

These use ``resolve=False`` where the URL is rejected on its literal form, and
monkeypatch DNS where a hostname must be shown to resolve to a private address.
No test touches the real network.
"""

from __future__ import annotations

import socket

import pytest

from app.services.ssrf import SSRFError, is_safe_url, validate_url


# ---------------------------------------------------------------------------
# Literal rejections (no DNS needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost:8080/admin",
        "http://LOCALHOST/",
        "http://127.0.0.1/",
        "http://127.0.0.1:5433/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::]/",
        "http://10.0.0.1/",
        "http://10.255.255.255/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://172.31.255.255/",
        "http://169.254.169.254/",  # cloud metadata
        "http://100.64.0.1/",  # shared address space
        "http://[fe80::1]/",  # link-local v6
        "http://[fd00::1]/",  # unique local v6
        "http://[::ffff:127.0.0.1]/",  # v4-mapped loopback
        "http://224.0.0.1/",  # multicast
        "http://255.255.255.255/",
    ],
)
def test_rejects_non_public_addresses(url: str) -> None:
    with pytest.raises(SSRFError):
        validate_url(url, resolve=False)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
        "//example.com/no-scheme",
    ],
)
def test_rejects_non_http_schemes(url: str) -> None:
    with pytest.raises(SSRFError):
        validate_url(url, resolve=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://foo.localhost/",
        "http://printer.local/",
        "http://db.internal/",
        "http://router.lan/",
        "http://intranet/",  # single label
    ],
)
def test_rejects_internal_hostnames(url: str) -> None:
    with pytest.raises(SSRFError):
        validate_url(url, resolve=False)


def test_rejects_empty_and_malformed() -> None:
    for url in ("", "   ", "http://", "not a url"):
        with pytest.raises(SSRFError):
            validate_url(url, resolve=False)


# ---------------------------------------------------------------------------
# DNS resolution
# ---------------------------------------------------------------------------

def test_public_ip_literal_passes() -> None:
    target = validate_url("https://93.184.216.34/page", resolve=False)
    assert target.host == "93.184.216.34"
    assert target.resolved_ips == ("93.184.216.34",)


def test_hostname_resolving_to_private_ip_is_rejected(monkeypatch) -> None:
    """The classic bypass: a public-looking name that resolves inward."""

    def fake_getaddrinfo(host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(SSRFError, match="non-public"):
        validate_url("http://evil.example.com/")


def test_hostname_with_one_private_address_among_public_is_rejected(monkeypatch) -> None:
    """Any private address in the answer set is enough to reject."""

    def fake_getaddrinfo(host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(SSRFError):
        validate_url("http://mixed.example.com/")


def test_hostname_resolving_to_public_ip_passes(monkeypatch) -> None:
    def fake_getaddrinfo(host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    target = validate_url("http://example.com/")
    assert target.resolved_ips == ("93.184.216.34",)


def test_unresolvable_host_is_rejected(monkeypatch) -> None:
    def fake_getaddrinfo(host, port, **kwargs):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(SSRFError, match="could not resolve"):
        validate_url("http://does-not-exist.example/")


# ---------------------------------------------------------------------------
# Boolean wrapper
# ---------------------------------------------------------------------------

def test_is_safe_url_returns_bool_and_never_raises() -> None:
    assert is_safe_url("http://127.0.0.1/", resolve=False) is False
    assert is_safe_url("https://93.184.216.34/", resolve=False) is True
    assert is_safe_url("garbage", resolve=False) is False
