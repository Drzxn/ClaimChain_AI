"""Deterministic evaluation runner.

Runs each dataset case through the **real** Stage 4 aggregation functions, then
compares the predicted verdict against the expected one.

Why this is a genuine evaluation and not a reimplementation
-----------------------------------------------------------
``stage4_verify.aggregate_subclaim`` and ``aggregate_overall`` are pure: they take
link tuples and subclaim aggregates and return verdicts, with no database, no
model and no network. The runner feeds each case's links straight into those
functions. So the verdicts under test are produced by the same code that serves
the API, not by a parallel copy that could drift.

What is NOT exercised here
--------------------------
Stages 1, 2, 3, 5 and 6 are not run. Retrieval, decomposition and source
assessment need live Gemini and Exa, which would make the evaluation
non-deterministic and quota-dependent. The dataset therefore supplies the
evidence links those stages would have produced, and the evaluation measures the
deterministic decision layer: given this evidence, does the system reach the
right verdict?

That is a real limitation and it is stated plainly in the results and README.

Failure handling
----------------
A case that raises is recorded as a failure with its error message and excluded
from the metrics, rather than crashing the run or being silently dropped. The
failure count is reported separately so a run cannot look clean while cases
vanished.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.eval.dataset import EvalCase, EvalDataset
from app.eval.metrics import MetricsReport, compute_metrics
from app.pipeline.stage4_verify import (
    AGGREGATION_VERSION,
    Stage4Error,
    aggregate_overall,
    aggregate_subclaim,
    build_contributions,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CaseResult:
    """Outcome for one dataset case."""

    case_id: str
    claim_text: str
    category: str
    expected_verdict: str
    predicted_verdict: str | None
    correct: bool
    score: float | None = None
    coverage: float | None = None
    contradicted_weight_ratio: float | None = None
    subclaim_verdicts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(slots=True)
class EvaluationReport:
    """The full result of one evaluation run."""

    dataset_name: str
    dataset_version: str
    dataset_path: str
    aggregation_version: str
    generated_at: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    metrics: MetricsReport
    cases: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view. Stable key order for diffable output."""
        return {
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version,
                "path": self.dataset_path,
                "aggregation_version": self.aggregation_version,
            },
            "generated_at": self.generated_at,
            "aggregation_version": self.aggregation_version,
            "summary": {
                "total_cases": self.total_cases,
                "successful_cases": self.successful_cases,
                "failed_cases": self.failed_cases,
            },
            "metrics": self.metrics.as_dict(),
            "cases": [
                {
                    "case_id": c.case_id,
                    "claim_text": c.claim_text,
                    "category": c.category,
                    "expected_verdict": c.expected_verdict,
                    "predicted_verdict": c.predicted_verdict,
                    "correct": c.correct,
                    "score": c.score,
                    "coverage": c.coverage,
                    "contradicted_weight_ratio": c.contradicted_weight_ratio,
                    "subclaim_verdicts": c.subclaim_verdicts,
                    "error": c.error,
                }
                for c in self.cases
            ],
        }


def evaluate_case(case: EvalCase) -> CaseResult:
    """Run one case through the real Stage 4 aggregation.

    Args:
        case: the dataset case.

    Returns:
        CaseResult. A case that raises is returned with ``error`` set rather
        than propagating, so one bad case cannot abort the whole run.
    """
    try:
        aggregates = []
        for subclaim in case.subclaims:
            contributions = build_contributions(
                [link.as_tuple() for link in subclaim.links]
            )
            aggregates.append(
                aggregate_subclaim(
                    # A stable synthetic id: the aggregation only needs identity,
                    # not a database row.
                    subclaim_id=uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{case.case_id}#{subclaim.ordinal}"
                    ),
                    ordinal=subclaim.ordinal,
                    text=subclaim.text,
                    weight=1.0,
                    contributions=contributions,
                )
            )

        overall = aggregate_overall(aggregates)

        return CaseResult(
            case_id=case.case_id,
            claim_text=case.claim_text,
            category=case.category,
            expected_verdict=case.expected_verdict,
            predicted_verdict=overall.verdict,
            correct=overall.verdict == case.expected_verdict,
            score=overall.score,
            coverage=overall.coverage,
            contradicted_weight_ratio=overall.contradicted_weight_ratio,
            subclaim_verdicts=[a.verdict for a in aggregates],
        )

    except (Stage4Error, ValueError, TypeError, KeyError) as exc:
        logger.warning("Case %s failed: %s", case.case_id, exc)
        return CaseResult(
            case_id=case.case_id,
            claim_text=case.claim_text,
            category=case.category,
            expected_verdict=case.expected_verdict,
            predicted_verdict=None,
            correct=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_dataset(
    dataset: EvalDataset,
    *,
    dataset_path: str = "",
    generated_at: datetime | None = None,
) -> EvaluationReport:
    """Evaluate every case in a dataset.

    Args:
        dataset: a loaded dataset.
        dataset_path: recorded in the report for provenance.
        generated_at: override for reproducible output; defaults to now.

    Returns:
        EvaluationReport with per-case results and aggregate metrics.
    """
    results = [evaluate_case(case) for case in dataset.cases]

    # Metrics are computed over successful cases only. A failed case has no
    # prediction, and counting it as wrong would conflate a bug with a miss.
    pairs = [
        (r.expected_verdict, r.predicted_verdict)
        for r in results
        if r.predicted_verdict is not None
    ]

    metrics = compute_metrics(pairs)

    timestamp = (generated_at or datetime.now(timezone.utc)).isoformat()

    return EvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_path=dataset_path,
        aggregation_version=AGGREGATION_VERSION,
        generated_at=timestamp,
        total_cases=len(results),
        successful_cases=len(pairs),
        failed_cases=len(results) - len(pairs),
        metrics=metrics,
        cases=results,
    )


def format_summary(report: EvaluationReport) -> str:
    """Render a human-readable summary of a run.

    Plain text, fixed width, no colour: it is meant to be readable in a terminal
    and pasteable into a report.
    """
    m = report.metrics
    lines: list[str] = []

    lines.append("=" * 68)
    lines.append("ClaimChain AI — evaluation summary")
    lines.append("=" * 68)
    lines.append(f"Dataset          : {report.dataset_name} (v{report.dataset_version})")
    lines.append(f"Aggregation      : {report.aggregation_version}")
    lines.append(f"Generated        : {report.generated_at}")
    lines.append(f"Cases            : {report.total_cases} total, "
                 f"{report.successful_cases} evaluated, {report.failed_cases} failed")
    lines.append("")
    lines.append("Overall")
    lines.append(f"  Accuracy        : {m.accuracy:.4f}  ({m.correct}/{m.total})")
    lines.append(f"  Macro precision : {m.macro_precision:.4f}")
    lines.append(f"  Macro recall    : {m.macro_recall:.4f}")
    lines.append(f"  Macro F1        : {m.macro_f1:.4f}")
    lines.append("")
    lines.append("Per class")
    lines.append(f"  {'label':<24} {'support':>8} {'pred':>6} {'prec':>8} {'rec':>8} {'f1':>8}")
    for label in m.labels:
        c = m.per_class[label]
        lines.append(
            f"  {label:<24} {c.support:>8} {c.predicted:>6} "
            f"{c.precision:>8.4f} {c.recall:>8.4f} {c.f1:>8.4f}"
        )

    lines.append("")
    lines.append("Confusion matrix (rows = expected, columns = predicted)")
    header = "  " + " " * 24 + "".join(f"{label[:12]:>14}" for label in m.labels)
    lines.append(header)
    for expected in m.labels:
        row = f"  {expected:<24}"
        for predicted in m.labels:
            row += f"{m.confusion_matrix[expected][predicted]:>14}"
        lines.append(row)

    if report.failed_cases:
        lines.append("")
        lines.append("Failures")
        for case in report.cases:
            if case.failed:
                lines.append(f"  {case.case_id}: {case.error}")

    lines.append("")
    lines.append("Note: this dataset is a small, manually curated demonstration set,")
    lines.append("not a statistically representative benchmark. Stages 1, 2, 3 and 5")
    lines.append("are not executed; the dataset supplies the evidence links they would")
    lines.append("have produced, so the evaluation measures the deterministic decision")
    lines.append("layer only.")
    lines.append("=" * 68)

    return "\n".join(lines)


__all__ = [
    "CaseResult",
    "EvaluationReport",
    "evaluate_case",
    "evaluate_dataset",
    "format_summary",
]
