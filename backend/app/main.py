"""FastAPI application factory.

Foundation only. The only route wired up is ``GET /api/health``. Claim,
evidence, graph, report and pipeline routers land in later phases â€” see the
implementation order in the technical specification.
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import (
    analysis,
    assessment,
    claims,
    health,
    report,
    retrieval,
    verification,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks."""
    settings = get_settings()
    logger.info(
        "ClaimChain AI starting (env=%s, llm_configured=%s, search_configured=%s)",
        settings.app_env,
        settings.llm_configured,
        settings.search_configured,
    )
    yield
    logger.info("ClaimChain AI shutting down")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    app = FastAPI(
        lifespan=lifespan,
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Evidence-grounded claim verification and evidence mapping. "
            "Foundation build: health endpoint only."
        ),
        docs_url="/docs",
        redoc_url=None,
    )

    # CORS is locked to the Vite dev origin. No wildcard: the browser talks
    # only to this backend, and this backend holds the API keys.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(claims.router, prefix=settings.api_prefix)
    app.include_router(retrieval.router, prefix=settings.api_prefix)
    app.include_router(analysis.router, prefix=settings.api_prefix)
    app.include_router(assessment.router, prefix=settings.api_prefix)
    app.include_router(verification.router, prefix=settings.api_prefix)
    app.include_router(report.router, prefix=settings.api_prefix)

    return app


app = create_app()





