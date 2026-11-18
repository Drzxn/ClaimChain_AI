"""Manual smoke test against the real Gemini API.

Excluded from the default test run. It costs quota and needs a live key.

    # PowerShell
    $env:GEMINI_API_KEY = "..."
    .\\.venv\\Scripts\\python.exe -m pytest tests/smoke -q -m smoke

Or run it directly:

    .\\.venv\\Scripts\\python.exe -m tests.smoke.test_gemini_smoke

It is marked ``smoke`` and deselected by default via ``pytest.ini``.
"""

from __future__ import annotations

import os
import sys

import pytest

from app.config import get_settings
from app.pipeline.stage1_decompose import run_stage1
from app.schemas.claim import FIXED_SUBCLAIM_WEIGHT, MAX_SUBCLAIMS
from app.services.llm import GeminiService

CANONICAL = "Using HTTPS makes a website completely secure."

pytestmark = pytest.mark.smoke


def _require_key() -> None:
    settings = get_settings()
    if not (settings.gemini_api_key or os.getenv("GEMINI_API_KEY")):
        pytest.skip("GEMINI_API_KEY is not set; skipping live smoke test")


def test_real_gemini_decomposes_canonical_claim() -> None:
    """The canonical claim must decompose into valid, weight-1.0 subclaims."""
    _require_key()

    service = GeminiService()
    result = run_stage1(CANONICAL, service=service)

    # Structured JSON parsed and validated.
    assert result.response.domain in {"technology", "science", "general"}

    subclaims = result.response.subclaims
    assert 1 <= len(subclaims) <= MAX_SUBCLAIMS, f"got {len(subclaims)} subclaims"

    valid_types = {"factual", "causal", "comparative", "definitional", "quantitative"}
    for sub in subclaims:
        assert 1 <= len(sub.text) <= 300
        assert sub.claim_type in valid_types
        # The backend pins every weight; the model's proposal is discarded.
        assert sub.weight == FIXED_SUBCLAIM_WEIGHT, (
            f"weight {sub.weight} was not pinned to 1.0"
        )

    assert [s.ordinal for s in subclaims] == list(range(1, len(subclaims) + 1))

    # Print a readable summary when run directly.
    print(f"\ndomain: {result.response.domain}")
    print(f"model:  {result.response.model}")
    print(f"attempts: {len(result.attempts)}")
    print(f"tokens: in={result.response.tokens_in} out={result.response.tokens_out}")
    for sub in subclaims:
        print(f"  {sub.ordinal}. [{sub.claim_type}] {sub.text}")


if __name__ == "__main__":
    _require_key()
    service = GeminiService()
    result = run_stage1(CANONICAL, service=service)
    print(f"domain: {result.response.domain}")
    print(f"model:  {result.response.model}")
    print(f"attempts: {len(result.attempts)}")
    for sub in result.response.subclaims:
        print(f"  {sub.ordinal}. [{sub.claim_type}] w={sub.weight} {sub.text}")
    assert all(s.weight == FIXED_SUBCLAIM_WEIGHT for s in result.response.subclaims)
    print("\nSMOKE OK")
    sys.exit(0)
