"""Pytest fixtures for database tests.

Isolation strategy
------------------
Tests never touch the development database. A dedicated ``claimchain_test``
database is created on the same PostgreSQL container, the real Alembic
migration is applied to it once per session, and every test runs inside a
transaction that is rolled back afterwards.

Using the migration (rather than ``Base.metadata.create_all``) means the tests
exercise the schema that ships, not a schema generated fresh from the models.
If the migration and the models ever drift, these tests notice.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.main import create_app

TEST_DB_NAME = "claimchain_test"


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Test client bound to a freshly built application."""
    return TestClient(create_app())


def _base_url() -> str:
    """Development URL, used to derive the test URL."""
    return get_settings().database_url


def _test_url() -> str:
    """URL of the isolated test database."""
    explicit = os.getenv("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = _base_url()
    # Replace only the trailing database name.
    head, _, _ = base.rpartition("/")
    return f"{head}/{TEST_DB_NAME}"


def _admin_engine() -> Engine:
    """Engine connected to the maintenance database, for CREATE DATABASE."""
    base = _base_url()
    head, _, _ = base.rpartition("/")
    return create_engine(f"{head}/postgres", isolation_level="AUTOCOMMIT")


@pytest.fixture(scope="session")
def test_database() -> Generator[str, None, None]:
    """Create the test database and apply all migrations to it."""
    url = _test_url()

    admin = _admin_engine()
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB_NAME}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin.dispose()

    # Apply the real migration chain to the test database.
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_alembic_ini()))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    yield url


def _alembic_ini():
    """Path to alembic.ini, resolved from this file's location."""
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture(scope="session")
def engine(test_database: str) -> Generator[Engine, None, None]:
    """Engine bound to the migrated test database."""
    eng = create_engine(test_database, pool_pre_ping=True, future=True)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine: Engine) -> Generator[Session, None, None]:
    """A session whose work is always rolled back.

    The outer transaction is never committed, so each test starts from an
    empty schema regardless of what it inserted.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
