"""Evaluation dataset loading and validation.

The dataset is a committed JSON file. Each case carries the evidence links a
retrieval stage would have produced, so the whole evaluation runs offline
through the real Stage 4 aggregation code: no Gemini, no Exa, no database.

Validation is strict. A malformed case is a bug in the dataset, not something to
skip quietly, because a silently dropped case would flatter the metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The verdict labels Stage 4 can produce. A dataset expecting anything else is
# invalid, and a prediction outside this set would indicate a broken pipeline.
VALID_VERDICTS: frozenset[str] = frozenset(
    {
        "SUPPORTED",
        "CONTRADICTED",
        "MIXED",
        "INSUFFICIENT_EVIDENCE",
        "MISLEADING",
    }
)

VALID_RELATIONSHIPS: frozenset[str] = frozenset(
    {"SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT"}
)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "dataset" / "claims.v1.json"
)


class DatasetError(ValueError):
    """The dataset is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class EvalLink:
    """One evidence link as the pipeline would have stored it."""

    relationship: str
    authority: float | None
    recency: float | None
    relevance: float
    evidence_quality: float
    stance_confidence: float

    def as_tuple(self) -> tuple[str, float | None, float | None, float, float, float]:
        """Shape expected by ``stage4_verify.build_contributions``."""
        return (
            self.relationship,
            self.authority,
            self.recency,
            self.relevance,
            self.evidence_quality,
            self.stance_confidence,
        )


@dataclass(frozen=True, slots=True)
class EvalSubclaim:
    """One subclaim with the links judged against it."""

    ordinal: int
    text: str
    claim_type: str
    links: tuple[EvalLink, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation case."""

    case_id: str
    claim_text: str
    expected_verdict: str
    category: str = "uncategorised"
    notes: str = ""
    subclaims: tuple[EvalSubclaim, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A loaded dataset with its provenance."""

    version: str
    name: str
    description: str
    aggregation_version: str
    cases: tuple[EvalCase, ...]

    @property
    def size(self) -> int:
        return len(self.cases)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def _parse_link(raw: Any, *, case_id: str, ordinal: int) -> EvalLink:
    _require(isinstance(raw, dict), f"{case_id} subclaim {ordinal}: link must be an object")

    relationship = raw.get("relationship")
    _require(
        relationship in VALID_RELATIONSHIPS,
        f"{case_id} subclaim {ordinal}: invalid relationship {relationship!r}",
    )

    def score(key: str, *, nullable: bool = False) -> float | None:
        value = raw.get(key)
        if value is None and nullable:
            return None
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{case_id} subclaim {ordinal}: {key} must be a number",
        )
        value = float(value)  # type: ignore[arg-type]
        _require(
            0.0 <= value <= 1.0,
            f"{case_id} subclaim {ordinal}: {key} must be within [0, 1], got {value}",
        )
        return value

    return EvalLink(
        relationship=str(relationship),
        authority=score("authority", nullable=True),
        recency=score("recency", nullable=True),
        relevance=score("relevance") or 0.0,  # type: ignore[arg-type]
        evidence_quality=score("evidence_quality") or 0.0,  # type: ignore[arg-type]
        stance_confidence=score("stance_confidence") or 0.0,  # type: ignore[arg-type]
    )


def _parse_subclaim(raw: Any, *, case_id: str, index: int) -> EvalSubclaim:
    _require(
        isinstance(raw, dict), f"{case_id}: subclaim {index} must be an object"
    )

    ordinal = raw.get("ordinal")
    _require(
        isinstance(ordinal, int) and ordinal >= 1,
        f"{case_id}: subclaim {index} ordinal must be an integer >= 1",
    )

    text = raw.get("text")
    _require(
        isinstance(text, str) and text.strip(),
        f"{case_id}: subclaim {index} text must be a non-empty string",
    )

    claim_type = raw.get("claim_type", "factual")
    _require(
        isinstance(claim_type, str) and claim_type.strip(),
        f"{case_id}: subclaim {index} claim_type must be a non-empty string",
    )

    raw_links = raw.get("links", [])
    _require(
        isinstance(raw_links, list),
        f"{case_id}: subclaim {index} links must be a list",
    )

    links = tuple(
        _parse_link(link, case_id=case_id, ordinal=ordinal) for link in raw_links
    )

    return EvalSubclaim(
        ordinal=ordinal,
        text=text.strip(),
        claim_type=claim_type.strip(),
        links=links,
    )


def _parse_case(raw: Any, *, index: int) -> EvalCase:
    _require(isinstance(raw, dict), f"case {index} must be an object")

    case_id = raw.get("case_id")
    _require(
        isinstance(case_id, str) and case_id.strip(),
        f"case {index}: case_id must be a non-empty string",
    )

    claim_text = raw.get("claim_text")
    _require(
        isinstance(claim_text, str) and claim_text.strip(),
        f"{case_id}: claim_text must be a non-empty string",
    )

    expected = raw.get("expected_verdict")
    _require(
        expected in VALID_VERDICTS,
        f"{case_id}: expected_verdict {expected!r} is not one of {sorted(VALID_VERDICTS)}",
    )

    raw_subclaims = raw.get("subclaims")
    _require(
        isinstance(raw_subclaims, list) and raw_subclaims,
        f"{case_id}: subclaims must be a non-empty list",
    )

    subclaims = tuple(
        _parse_subclaim(sub, case_id=case_id, index=i)
        for i, sub in enumerate(raw_subclaims)
    )

    ordinals = [s.ordinal for s in subclaims]
    _require(
        len(ordinals) == len(set(ordinals)),
        f"{case_id}: duplicate subclaim ordinals {ordinals}",
    )

    return EvalCase(
        case_id=case_id.strip(),
        claim_text=claim_text.strip(),
        expected_verdict=str(expected),
        category=str(raw.get("category", "uncategorised")),
        notes=str(raw.get("notes", "")),
        subclaims=subclaims,
    )


def load_dataset(path: str | Path | None = None) -> EvalDataset:
    """Load and validate an evaluation dataset.

    Args:
        path: dataset file; defaults to the committed demonstration set.

    Returns:
        A validated EvalDataset.

    Raises:
        DatasetError: the file is missing, not JSON, or fails validation.
    """
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        raise DatasetError(f"dataset not found: {dataset_path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"dataset is not valid JSON: {exc}") from exc

    _require(isinstance(raw, dict), "dataset root must be an object")

    raw_cases = raw.get("cases")
    _require(
        isinstance(raw_cases, list) and raw_cases,
        "dataset must contain a non-empty 'cases' list",
    )

    cases = tuple(_parse_case(case, index=i) for i, case in enumerate(raw_cases))

    ids = [c.case_id for c in cases]
    duplicates = {cid for cid in ids if ids.count(cid) > 1}
    _require(not duplicates, f"duplicate case_id values: {sorted(duplicates)}")

    return EvalDataset(
        version=str(raw.get("version", "unversioned")),
        name=str(raw.get("name", "unnamed dataset")),
        description=str(raw.get("description", "")),
        aggregation_version=str(raw.get("aggregation_version", "unknown")),
        cases=cases,
    )


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DatasetError",
    "EvalCase",
    "EvalDataset",
    "EvalLink",
    "EvalSubclaim",
    "VALID_RELATIONSHIPS",
    "VALID_VERDICTS",
    "load_dataset",
]
