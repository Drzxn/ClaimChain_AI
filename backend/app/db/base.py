"""Declarative base for ORM models.

The eight core tables (claims, subclaims, sources, passages, evidence_links,
subclaim_verdicts, verdicts, runs) are defined in the next implementation
phase. This module exists now so that phase has a single, stable base.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ClaimChain ORM models."""
