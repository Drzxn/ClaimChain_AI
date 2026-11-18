"""Verification pipeline.

Stage modules are added in later phases. Runtime execution order is:

    Stage 1 -> Stage 2 -> Stage 3 -> Stage 5 -> Stage 4 -> Stage 6

Stage numbers are conceptual/rubric labels, not execution order. Stage 5
(source assessment) must run before Stage 4 (verification) because the
deterministic aggregation in Stage 4 consumes the source weights that Stage 5
produces.
"""'
  "claimchain\backend\app\services\__init__.py" = '"""External service adapters (LLM, search, fetch, extract, chunk, cache)."""'
  "claimchain\backend\tests\__init__.py" = ''
  "claimchain\backend\tests\unit\__init__.py" = ''
  "claimchain\backend\tests\contract\__init__.py" = ''
  "claimchain\backend\tests\integration\__init__.py" = ''
  "claimchain\backend\tests\conftest.py" = @'
"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Test client bound to a freshly built application."""
    return TestClient(create_app())
