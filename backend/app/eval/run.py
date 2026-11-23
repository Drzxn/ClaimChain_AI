"""CLI entry point for the evaluation harness.

    cd backend
    .\\.venv\\Scripts\\python.exe -m app.eval.run

Runs the committed demonstration dataset through the deterministic Stage 4
aggregation, prints a summary, and writes JSON and CSV artifacts.

No API keys are required and no network call is made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.eval.artifacts import DEFAULT_RESULTS_DIR, write_artifacts
from app.eval.dataset import DEFAULT_DATASET_PATH, DatasetError, load_dataset
from app.eval.evaluator import evaluate_dataset, format_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.run",
        description=(
            "Run the ClaimChain deterministic evaluation harness. "
            "Offline: no Gemini, Exa, or database access."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"dataset JSON file (default: {DEFAULT_DATASET_PATH.name})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory for JSON and CSV artifacts (default: app/eval/results)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the summary without writing artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the harness. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = evaluate_dataset(dataset, dataset_path=str(args.dataset))

    print(format_summary(report))

    if not args.no_write:
        written = write_artifacts(report, args.output_dir)
        print()
        print(f"JSON written to: {written.json_path}")
        print(f"CSV  written to: {written.csv_path}")

    # A failed case is a harness or dataset problem, not a wrong prediction.
    # Surface it in the exit code so CI can catch it.
    if report.failed_cases:
        print(
            f"\nwarning: {report.failed_cases} case(s) failed to evaluate",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
