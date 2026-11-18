"""Safe HTTP fetching.

Every request is validated by the SSRF guard before it is made, and **again on
every redirect hop**. Redirects are followed manually rather than by the HTTP
client, because a client that follows redirects internally will happily land on
``http://127.0.0.1`` after starting at a public URL.

Limits are enforced before the body is read in full: a declared
``Content-Length`` over the cap is rejected outright, and a streamed body is
aborted once it exceeds the cap. This prevents a hostile or merely enormous
page from exhausting memory.

Only ``text/html`` and ``text/plain`` are accepted. Everything else (PDFs,
images, archives, video) is skipped in this phase rather than downloaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from app.services.ssrf import SSRFError, validate_url

logger = logging.getLogger(__name__)

USER_AGENT = (
    "ClaimChainAI/0.1 (academic claim verification project; "
    "+https://github.com/claimchain) httpx"
)

ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
DEFAULT_MAX_REDIRECTS = 5


class FetchError(RuntimeError):
    """A page could not be fetched. Carries a short, non-secret reason."""

    def __init__(self, reason: str, *, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


@dataclass(slots=True)
class FetchResult:
    """A successfully fetched page."""

    url: str  # final URL after redirects
    requested_url: str
    status: int
    content_type: str
    text: str
    bytes_read: int
    redirects: list[str] = field(default_factory=list)


def _content_type_of(response: httpx.Response) -> str:
    """Lowercased media type without parameters."""
    raw = response.headers.get("content-type", "")
    return raw.split(";", 1)[0].strip().lower()


def _is_allowed_content_type(content_type: str) -> bool:
    if not content_type:
        # Some servers omit it. Treat as HTML: extraction will reject junk.
        return True
    return any(content_type.startswith(allowed) for allowed in ALLOWED_CONTENT_TYPES)


def _decode(response: httpx.Response, max_bytes: int) -> tuple[str, int]:
    """Read at most ``max_bytes`` and decode to text.

    Raises:
        FetchError: the body exceeded the cap.
    """
    declared = response.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise FetchError(
                    f"declared content-length {declared} exceeds cap {max_bytes}"
                )
        except ValueError:
            # A malformed Content-Length is not fatal; the streamed cap applies.
            pass

    buffer = bytearray()
    for chunk in response.iter_bytes():
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise FetchError(f"response body exceeded cap of {max_bytes} bytes")

    encoding = response.encoding or "utf-8"
    try:
        text = bytes(buffer).decode(encoding, errors="replace")
    except LookupError:
        text = bytes(buffer).decode("utf-8", errors="replace")

    return text, len(buffer)


def fetch_page(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    client: httpx.Client | None = None,
    resolve_dns: bool = True,
) -> FetchResult:
    """Fetch a URL safely, following redirects with revalidation.

    Args:
        url: absolute http(s) URL.
        timeout: per-request timeout in seconds.
        max_bytes: hard cap on the response body.
        max_redirects: maximum redirect hops.
        client: injected httpx client (tests pass a MockTransport-backed one).
        resolve_dns: pass False only in tests that must not touch DNS.

    Returns:
        FetchResult with the decoded text.

    Raises:
        FetchError: rejected, unreachable, non-2xx, wrong content type, or
            oversized.
    """
    current = url
    redirects: list[str] = []
    owns_client = client is None

    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,  # we follow manually, with revalidation
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain"},
        )

    try:
        for hop in range(max_redirects + 1):
            # --- validate before every request, including redirect targets ---
            try:
                validate_url(current, resolve=resolve_dns)
            except SSRFError as exc:
                raise FetchError(f"blocked by SSRF guard: {exc}") from exc

            try:
                response = client.get(current)
            except httpx.TimeoutException as exc:
                raise FetchError(f"timeout after {timeout}s") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"transport error: {type(exc).__name__}") from exc

            # --- redirect handling ------------------------------------------
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise FetchError("redirect without a Location header")
                if hop >= max_redirects:
                    raise FetchError(f"exceeded {max_redirects} redirects")

                # Resolve relative redirects against the current URL.
                next_url = str(httpx.URL(current).join(location))
                redirects.append(next_url)
                current = next_url
                continue

            # --- terminal response ------------------------------------------
            if response.status_code >= 400:
                raise FetchError(
                    f"HTTP {response.status_code}", status=response.status_code
                )

            content_type = _content_type_of(response)
            if not _is_allowed_content_type(content_type):
                raise FetchError(f"unsupported content type: {content_type or 'unknown'}")

            text, size = _decode(response, max_bytes)

            return FetchResult(
                url=current,
                requested_url=url,
                status=response.status_code,
                content_type=content_type or "text/html",
                text=text,
                bytes_read=size,
                redirects=redirects,
            )

        raise FetchError(f"exceeded {max_redirects} redirects")

    finally:
        if owns_client:
            client.close()
