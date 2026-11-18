"""ORM models.

Intentionally empty in the foundation build.

Phase 2 adds the eight core tables defined in the technical specification:
claims, subclaims, sources, passages, evidence_links, subclaim_verdicts,
verdicts, runs.
"""

from app.db.base import Base

__all__ = ["Base"]
