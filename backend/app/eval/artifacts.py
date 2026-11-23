"""Result artifact generation.

Writes the two artifacts the harness produces:

* ``eval_<timestamp>.json``: full structured report, including every case.
* ``eval_<timestamp>.csv``: one row per case, for spreadsheet inspection.

Both are generated artifacts. They live in ``app/eval/results/`` which is
git-ignored apart from its README and ``.gitkeep``, so a run never dirties the
working tree.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.eval.evaluator import EvaluationReport

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

CSV_COLUMNS = [
    "case_id",
    "category",
    "expected_verdict",
    "predicted_verdict",
    "correct",
    "score",
    "coverage",
    "contradicted_weight_ratio",
    "subclaim_verdicts",
    "error",
]


@dataclass(frozen=True, slots=True)
class WrittenArtifacts:
    """Paths of the files a run produced."""

    json_path: Path
    csv_path: Path


def _stamp(report: EvaluationReport) -> str:
    """A filename-safe timestamp derived from the report, not the clock.

    Using the report's own timestamp keeps the filenames consistent with the
    contents when ``generated_at`` is overridden for a reproducible run.
    """
    try:
        parsed = datetime.fromisoformat(report.generated_at)
    except ValueError:
        parsed = datetime.now(timezone.utc)
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def render_csv(report: EvaluationReport) -> str:
    """Render the per-case CSV as a string.

    Separated from writing so it can be tested without touching the filesystem.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for case in report.cases:
        writer.writerow(
            {
                "case_id": case.case_id,
                "category": case.category,
                "expected_verdict": case.expected_verdict,
                "predicted_verdict": case.predicted_verdict or "",
                "correct": "true" if case.correct else "false",
                "score": "" if case.score is None else f"{case.score:.4f}",
                "coverage": "" if case.coverage is None else f"{case.coverage:.4f}",
                "contradicted_weight_ratio": (
                    ""
                    if case.contradicted_weight_ratio is None
                    else f"{case.contradicted_weight_ratio:.4f}"
                ),
                "subclaim_verdicts": "|".join(case.subclaim_verdicts),
                "error": case.error or "",
            }
        )

    return buffer.getvalue()


def render_json(report: EvaluationReport) -> str:
    """Render the full report as indented JSON."""
    return json.dumps(report.as_dict(), indent=2, sort_keys=False) + "\n"


def write_artifacts(
    report: EvaluationReport, output_dir: str | Path | None = None
) -> WrittenArtifacts:
    """Write the JSON and CSV artifacts.

    Args:
        report: the completed evaluation report.
        output_dir: destination; defaults to ``app/eval/results/``.

    Returns:
        The paths written.
    """
    directory = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    stamp = _stamp(report)
    json_path = directory / f"eval_{stamp}.json"
    csv_path = directory / f"eval_{stamp}.csv"

    json_path.write_text(render_json(report), encoding="utf-8")
    csv_path.write_text(render_csv(report), encoding="utf-8")

    return WrittenArtifacts(json_path=json_path, csv_path=csv_path)


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_RESULTS_DIR",
    "WrittenArtifacts",
    "render_csv",
    "render_json",
    "write_artifacts",
]
