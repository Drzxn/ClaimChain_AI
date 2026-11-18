"""Foundation smoke tests.

Covers only what the foundation build claims to do: the app constructs, the
health endpoint answers, and no secret value is ever echoed in a response.
"""

from fastapi.testclient import TestClient

from app.main import create_app


def test_app_constructs() -> None:
    """The application factory returns a usable app."""
    app = create_app()
    assert app.title == "ClaimChain AI"


def test_health_returns_ok(client: TestClient) -> None:
    """GET /api/health responds 200 with status ok."""
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    # Database may be up or down depending on whether Docker is running;
    # either is a valid, non-crashing state for the probe.
    assert body["database"] in {"ok", "unavailable"}


def test_health_never_leaks_secrets(client: TestClient) -> None:
    """The health body reports key presence as a boolean, never the value."""
    body = client.get("/api/health").json()
    assert isinstance(body["llm_configured"], bool)
    assert isinstance(body["search_configured"], bool)

    serialized = str(body)
    assert "GEMINI_API_KEY" not in serialized
    assert "EXA_API_KEY" not in serialized


def test_unknown_route_is_404(client: TestClient) -> None:
    """Unimplemented routes are absent, not stubbed."""
    assert client.get("/api/claims").status_code == 404
