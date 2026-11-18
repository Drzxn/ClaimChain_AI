"""SSRF protection.

The system fetches URLs returned by a search engine, which means an attacker
who can influence search results can influence what this backend requests.
Every URL must therefore be checked against the network it would actually reach,
not against its spelling.

The check is two-stage and both stages are required:

1. **Literal check** — parse the URL, require http/https, and reject hostnames
   that are obviously local (``localhost``, ``*.localhost``, ``*.internal``).
2. **Resolution check** — resolve the hostname to every address it maps to and
   reject if *any* of them is loopback, private, link-local, multicast,
   reserved, or unspecified. Checking only the first address is a classic
   bypass: a hostile DNS record can return one public and one private address.

Redirects are handled by the fetcher, which re-runs this same validation on
every hop. A URL that passes here is not trusted for the redirect target.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hostnames that are local by definition and must never be resolved.
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "broadcasthost",
    }
)

# Suffixes that indicate an internal or non-public namespace.
BLOCKED_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".intranet",
    ".corp",
    ".home",
    ".lan",
    ".private",
)


class SSRFError(ValueError):
    """The URL was rejected because it could reach a non-public destination."""


@dataclass(frozen=True, slots=True)
class ValidatedTarget:
    """A URL that passed validation, plus the addresses it resolved to."""

    url: str
    host: str
    scheme: str
    resolved_ips: tuple[str, ...]


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if the address is anything other than a public unicast address."""
    # ``is_global`` is False for loopback, private, link-local, multicast,
    # reserved, unspecified, and shared address space (100.64.0.0/10).
    if not ip.is_global:
        return True
    # Belt and braces for addresses some Python versions still call global.
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True

    # IPv4-mapped IPv6 (::ffff:127.0.0.1) must be judged as its IPv4 self.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_blocked_ip(ip.ipv4_mapped)

    return False


def _check_hostname_literal(host: str) -> None:
    """Reject hostnames that are local by name, before any DNS lookup."""
    lowered = host.lower().rstrip(".")

    if lowered in BLOCKED_HOSTNAMES:
        raise SSRFError(f"blocked hostname: {host!r}")

    for suffix in BLOCKED_SUFFIXES:
        if lowered.endswith(suffix):
            raise SSRFError(f"blocked internal hostname: {host!r}")

    # A single-label hostname ("intranet", "router") is not a public FQDN.
    if "." not in lowered and ":" not in lowered:
        raise SSRFError(f"blocked single-label hostname: {host!r}")


def _resolve(host: str, port: int) -> tuple[str, ...]:
    """Resolve a hostname to all of its addresses.

    Raises:
        SSRFError: resolution failed, or produced no addresses.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError(f"could not resolve host {host!r}: {exc}") from exc

    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr and sockaddr[0] not in addresses:
            addresses.append(sockaddr[0])

    if not addresses:
        raise SSRFError(f"host {host!r} resolved to no addresses")

    return tuple(addresses)


def validate_url(url: str, *, resolve: bool = True) -> ValidatedTarget:
    """Validate a URL for outbound fetching.

    Args:
        url: the absolute URL to check.
        resolve: when False, skip DNS resolution. Used only by tests that must
            not touch the network; production callers always resolve.

    Returns:
        ValidatedTarget with the resolved addresses.

    Raises:
        SSRFError: the URL is unsafe or unresolvable.
    """
    if not url or not url.strip():
        raise SSRFError("empty URL")

    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()

    if scheme not in ALLOWED_SCHEMES:
        raise SSRFError(f"scheme {scheme!r} is not allowed; only http and https")

    host = parts.hostname
    if not host:
        raise SSRFError("URL has no host")

    _check_hostname_literal(host)

    default_port = 443 if scheme == "https" else 80
    port = parts.port or default_port

    # An IP literal needs no DNS lookup, but still needs the address check.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if _is_blocked_ip(literal_ip):
            raise SSRFError(f"blocked IP address: {host}")
        return ValidatedTarget(
            url=url, host=host, scheme=scheme, resolved_ips=(str(literal_ip),)
        )

    if not resolve:
        return ValidatedTarget(url=url, host=host, scheme=scheme, resolved_ips=())

    addresses = _resolve(host, port)
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise SSRFError(f"unparseable address {address!r} for {host!r}") from exc
        if _is_blocked_ip(ip):
            raise SSRFError(
                f"host {host!r} resolves to non-public address {address}"
            )

    return ValidatedTarget(
        url=url, host=host, scheme=scheme, resolved_ips=addresses
    )


def is_safe_url(url: str, *, resolve: bool = True) -> bool:
    """Boolean wrapper. Never raises; logs rejections at debug level."""
    try:
        validate_url(url, resolve=resolve)
        return True
    except SSRFError as exc:
        logger.debug("SSRF rejection for %s: %s", url, exc)
        return False
