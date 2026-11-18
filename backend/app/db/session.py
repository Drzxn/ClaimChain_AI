"""Database session and engine management.

Foundation only: an engine, a session factory, and a reachability probe. The
eight core tables and their ORM models arrive in the next phase; this module
deliberately contains no schema.
"""

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# ``pool_pre_ping`` avoids handing out connections that died while the Docker
# container was restarting, which is the single most common local-dev failure.
engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database() -> bool:
    """Return True if a trivial query succeeds against the configured database."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - caller decides how to report
        logger.debug("Database probe failed", exc_info=True)
        return False
