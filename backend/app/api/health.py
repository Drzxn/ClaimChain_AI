"""Health endpoint.

Liveness plus an optional database reachability probe.

The endpoint always returns HTTP 200 when the process is alive. Database
reachability is reported in the body rather than as a failing status code, so
a stopped Postgres container does not make the app look crashed — it makes it
look like the app is up and the database is down, which is the truth.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.config import get_settings
from app.db.session import check_database

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response body for the health probe."""

    status: str
    version: str
    environment: str
    database: str
    llm_configured: bool
    search_configured: bool


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """Report process liveness and dependency reachability.

    Never raises: a database failure is reported as ``database="unavailable"``.
    """
    settings = get_settings()

    try:
        db_ok = check_database()
        database_state = "ok" if db_ok else "unavailable"
    except Exception:  # noqa: BLE001 - health must never raise
        logger.warning("Database health probe failed", exc_info=True)
        database_state = "unavailable"

    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
        database=database_state,
        llm_configured=settings.llm_configured,
        search_configured=settings.search_configured,
    )
