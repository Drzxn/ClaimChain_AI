"""Evaluation harness tests.

No network, no database, no API keys. The harness is pure computation over a
committed dataset, so these run anywhere.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.eval.artifacts import CSV_COLUMNS, render_csv, render_json, write_artifacts
from app.eval.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    EvalCase,
    EvalDataset,
    EvalLink,
    EvalSubclaim,
    load_dataset,
)
from app.eval.evaluator import (
    evaluate_case,
    evaluate_dataset,
    format_summary,
)
from app.eval.metrics import build_confusion_matrix, compute_metrics
from app.eval.run import main
from app.pipeline.stage4_verify import AGGREGATION_VERSION

FIXED_TIME = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_committed_dataset_loads() -> None:
    dataset = load_dataset()
    assert dataset.size >= 8
    assert dataset.version
    assert dataset.aggregation_version == AGGREGATION_VERSION


def test_committed_dataset_covers_every_verdict_class() -> None:
    """The dataset must exercise all five outcomes, or the metrics are hollow."""
    dataset = load_dataset()
    expected = {case.expected_verdict for case in dataset.cases}
    assert expected == {
        "SUPPORTED",
        "CONTRADICTED",
        "MIXED",
        "INSUFFICIENT_EVIDENCE",
        "MISLEADING",
    }


def test_committed_dataset_ids_are_unique() -> None:
    dataset = load_dataset()
    ids = [case.case_id for case in dataset.cases]
    assert len(ids) == len(set(ids))


def test_committed_dataset_has_an_insufficient_case_with_no_links() -> None:
    """Abstention must be represented by genuinely absent evidence."""
    dataset = load_dataset()
    no_evidence = [
        case
        for case in dataset.cases
        if case.expected_verdict == "INSUFFICIENT_EVIDENCE"
        and all(not sub.links for sub in case.subclaims)
    ]
    assert no_evidence


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(path)


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "d.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _minimal_case(**overrides) -> dict:
    case = {
        "case_id": "c1",
        "claim_text": "A claim.",
        "expected_verdict": "SUPPORTED",
        "subclaims": [
            {
                "ordinal": 1,
                "text": "A subclaim.",
                "claim_type": "factual",
                "links": [
                    {
                        "relationship": "SUPPORTS",
                        "authority": 1.0,
                        "recency": 1.0,
                        "relevance": 1.0,
                        "evidence_quality": 1.0,
                        "stance_confidence": 1.0,
                    }
                ],
            }
        ],
    }
    case.update(overrides)
    return case


def test_rejects_unknown_expected_verdict(tmp_path: Path) -> None:
    path = _write(tmp_path, {"cases": [_minimal_case(expected_verdict="TRUE")]})
    with pytest.raises(DatasetError, match="expected_verdict"):
        load_dataset(path)


def test_rejects_empty_cases(tmp_path: Path) -> None:
    path = _write(tmp_path, {"cases": []})
    with pytest.raises(DatasetError, match="non-empty"):
        load_dataset(path)


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = _write(tmp_path, {"cases": [_minimal_case(), _minimal_case()]})
    with pytest.raises(DatasetError, match="duplicate case_id"):
        load_dataset(path)


def test_rejects_invalid_relationship(tmp_path: Path) -> None:
    case = _minimal_case()
    case["subclaims"][0]["links"][0]["relationship"] = "REFUTES"
    path = _write(tmp_path, {"cases": [case]})
    with pytest.raises(DatasetError, match="invalid relationship"):
        load_dataset(path)


def test_rejects_out_of_range_score(tmp_path: Path) -> None:
    case = _minimal_case()
    case["subclaims"][0]["links"][0]["relevance"] = 1.5
    path = _write(tmp_path, {"cases": [case]})
    with pytest.raises(DatasetError, match=r"within \[0, 1\]"):
        load_dataset(path)


def test_rejects_missing_subclaims(tmp_path: Path) -> None:
    case = _minimal_case()
    del case["subclaims"]
    path = _write(tmp_path, {"cases": [case]})
    with pytest.raises(DatasetError, match="subclaims"):
        load_dataset(path)


def test_rejects_duplicate_ordinals(tmp_path: Path) -> None:
    case = _minimal_case()
    case["subclaims"].append(dict(case["subclaims"][0]))
    path = _write(tmp_path, {"cases": [case]})
    with pytest.raises(DatasetError, match="duplicate subclaim ordinals"):
        load_dataset(path)


def test_accepts_null_authority(tmp_path: Path) -> None:
    """An unassessed source is valid dataset input; Stage 4 treats it as 0 weight."""
    case = _minimal_case()
    case["subclaims"][0]["links"][0]["authority"] = None
    case["subclaims"][0]["links"][0]["recency"] = None
    path = _write(tmp_path, {"cases": [case]})
    dataset = load_dataset(path)
    assert dataset.cases[0].subclaims[0].links[0].authority is None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_accuracy_on_perfect_predictions() -> None:
    m = compute_metrics([("A", "A"), ("B", "B")])
    assert m.accuracy == 1.0
    assert m.correct == 2


def test_accuracy_on_all_wrong() -> None:
    m = compute_metrics([("A", "B"), ("B", "A")])
    assert m.accuracy == 0.0


def test_per_class_precision_recall_f1() -> None:
    # A: 1 TP, 1 FN, 0 FP -> precision 0, recall 0.5
    # B: 1 TP, 0 FN, 1 FP -> precision 0.5, recall 1.0
    m = compute_metrics([("A", "A"), ("A", "B"), ("B", "B")])

    a = m.per_class["A"]
    assert a.support == 2
    assert a.predicted == 1
    assert a.true_positives == 1
    assert a.precision == 1.0
    assert a.recall == 0.5
    assert a.f1 == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    b = m.per_class["B"]
    assert b.precision == 0.5
    assert b.recall == 1.0


def test_macro_average_excludes_absent_classes() -> None:
    """A class with zero support must not drag the macro average down."""
    m = compute_metrics([("A", "A"), ("A", "A")], labels=["A", "B"])
    assert m.per_class["B"].support == 0
    # Only A is averaged, and A is perfect.
    assert m.macro_precision == 1.0
    assert m.macro_recall == 1.0
    assert m.macro_f1 == 1.0


def test_zero_denominators_yield_zero_not_error() -> None:
    m = compute_metrics([("A", "A")], labels=["A", "B"])
    b = m.per_class["B"]
    assert b.precision == 0.0
    assert b.recall == 0.0
    assert b.f1 == 0.0


def test_empty_input_yields_zero_metrics() -> None:
    m = compute_metrics([])
    assert m.total == 0
    assert m.accuracy == 0.0
    assert m.macro_f1 == 0.0


def test_metrics_are_deterministic() -> None:
    pairs = [("A", "A"), ("B", "C"), ("C", "C")]
    first = compute_metrics(pairs)
    second = compute_metrics(pairs)
    assert first.as_dict() == second.as_dict()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_counts_pairs() -> None:
    matrix = build_confusion_matrix([("A", "A"), ("A", "B"), ("B", "B")])
    assert matrix["A"]["A"] == 1
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["B"] == 1
    assert matrix["B"]["A"] == 0


def test_confusion_matrix_has_every_cell() -> None:
    """Zeros must be explicit, not inferred from a missing key."""
    matrix = build_confusion_matrix([("A", "A")], labels=["A", "B"])
    assert set(matrix) == {"A", "B"}
    for row in matrix.values():
        assert set(row) == {"A", "B"}


def test_confusion_matrix_row_sums_equal_support() -> None:
    pairs = [("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")]
    matrix = build_confusion_matrix(pairs)
    assert sum(matrix["A"].values()) == 2
    assert sum(matrix["B"].values()) == 2


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------

def _case(
    *,
    case_id: str = "c1",
    expected: str = "SUPPORTED",
    links: tuple[EvalLink, ...] = (),
    ordinal: int = 1,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        claim_text="A claim.",
        expected_verdict=expected,
        subclaims=(
            EvalSubclaim(
                ordinal=ordinal,
                text="A subclaim.",
                claim_type="factual",
                links=links,
            ),
        ),
    )


def _supporting() -> EvalLink:
    return EvalLink("SUPPORTS", 1.0, 1.0, 1.0, 1.0, 1.0)


def _contradicting() -> EvalLink:
    return EvalLink("CONTRADICTS", 1.0, 1.0, 1.0, 1.0, 1.0)


def test_evaluate_case_marks_correct() -> None:
    result = evaluate_case(_case(expected="SUPPORTED", links=(_supporting(),)))
    assert result.predicted_verdict == "SUPPORTED"
    assert result.correct is True
    assert result.error is None


def test_evaluate_case_marks_incorrect() -> None:
    result = evaluate_case(_case(expected="CONTRADICTED", links=(_supporting(),)))
    assert result.predicted_verdict == "SUPPORTED"
    assert result.correct is False


def test_evaluate_case_abstains_without_evidence() -> None:
    result = evaluate_case(_case(expected="INSUFFICIENT_EVIDENCE", links=()))
    assert result.predicted_verdict == "INSUFFICIENT_EVIDENCE"


def test_evaluate_case_uses_real_aggregation() -> None:
    """The predicted verdict must come from Stage 4, not a local reimplementation."""
    result = evaluate_case(_case(expected="CONTRADICTED", links=(_contradicting(),)))
    assert result.predicted_verdict == "CONTRADICTED"
    assert result.score == -1.0


def test_evaluate_case_records_failure_without_raising() -> None:
    """A malformed case must be recorded, not crash the run."""
    bad = EvalCase(
        case_id="bad",
        claim_text="A claim.",
        expected_verdict="SUPPORTED",
        subclaims=(
            EvalSubclaim(
                ordinal=1,
                text="A subclaim.",
                claim_type="factual",
                links=(EvalLink("NOT_A_RELATIONSHIP", 1.0, 1.0, 1.0, 1.0, 1.0),),
            ),
        ),
    )
    result = evaluate_case(bad)
    assert result.failed is True
    assert result.predicted_verdict is None
    assert result.correct is False
    assert "invalid stored relationship" in (result.error or "")


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

def _dataset(cases: tuple[EvalCase, ...]) -> EvalDataset:
    return EvalDataset(
        version="test",
        name="test dataset",
        description="",
        aggregation_version=AGGREGATION_VERSION,
        cases=cases,
    )


def test_evaluate_dataset_counts_successes_and_failures() -> None:
    good = _case(case_id="good", expected="SUPPORTED", links=(_supporting(),))
    bad = EvalCase(
        case_id="bad",
        claim_text="A claim.",
        expected_verdict="SUPPORTED",
        subclaims=(
            EvalSubclaim(
                ordinal=1,
                text="A subclaim.",
                claim_type="factual",
                links=(EvalLink("NOPE", 1.0, 1.0, 1.0, 1.0, 1.0),),
            ),
        ),
    )

    report = evaluate_dataset(_dataset((good, bad)), generated_at=FIXED_TIME)

    assert report.total_cases == 2
    assert report.successful_cases == 1
    assert report.failed_cases == 1


def test_failed_cases_are_excluded_from_metrics() -> None:
    """A failure is a bug, not a wrong answer; it must not count as a miss."""
    good = _case(case_id="good", expected="SUPPORTED", links=(_supporting(),))
    bad = EvalCase(
        case_id="bad",
        claim_text="A claim.",
        expected_verdict="SUPPORTED",
        subclaims=(
            EvalSubclaim(
                ordinal=1,
                text="A subclaim.",
                claim_type="factual",
                links=(EvalLink("NOPE", 1.0, 1.0, 1.0, 1.0, 1.0),),
            ),
        ),
    )

    report = evaluate_dataset(_dataset((good, bad)), generated_at=FIXED_TIME)

    assert report.metrics.total == 1
    assert report.metrics.accuracy == 1.0


def test_report_records_aggregation_version() -> None:
    report = evaluate_dataset(_dataset((_case(links=(_supporting(),)),)))
    assert report.aggregation_version == AGGREGATION_VERSION


def test_report_is_deterministic_for_a_fixed_timestamp() -> None:
    dataset = _dataset((_case(links=(_supporting(),)),))
    first = evaluate_dataset(dataset, generated_at=FIXED_TIME).as_dict()
    second = evaluate_dataset(dataset, generated_at=FIXED_TIME).as_dict()
    assert first == second


def test_committed_dataset_evaluates_without_failures() -> None:
    report = evaluate_dataset(load_dataset(), generated_at=FIXED_TIME)
    assert report.failed_cases == 0
    assert report.successful_cases == report.total_cases


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def _report():
    return evaluate_dataset(load_dataset(), generated_at=FIXED_TIME)


def test_json_artifact_is_valid_json() -> None:
    payload = json.loads(render_json(_report()))
    assert payload["aggregation_version"] == AGGREGATION_VERSION
    assert payload["summary"]["total_cases"] > 0
    assert "metrics" in payload
    assert "cases" in payload


def test_json_artifact_contains_every_case() -> None:
    report = _report()
    payload = json.loads(render_json(report))
    assert len(payload["cases"]) == report.total_cases
    assert {c["case_id"] for c in payload["cases"]} == {
        c.case_id for c in report.cases
    }


def test_json_artifact_includes_confusion_matrix() -> None:
    payload = json.loads(render_json(_report()))
    matrix = payload["metrics"]["confusion_matrix"]
    assert matrix
    for row in matrix.values():
        assert isinstance(row, dict)


def test_csv_artifact_parses_with_expected_columns() -> None:
    text = render_csv(_report())
    reader = csv.DictReader(io.StringIO(text))
    assert reader.fieldnames == CSV_COLUMNS
    rows = list(reader)
    assert rows
    assert set(rows[0]) == set(CSV_COLUMNS)


def test_csv_artifact_has_one_row_per_case() -> None:
    report = _report()
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert len(rows) == report.total_cases


def test_csv_records_correctness_as_text() -> None:
    rows = list(csv.DictReader(io.StringIO(render_csv(_report()))))
    for row in rows:
        assert row["correct"] in {"true", "false"}


def test_csv_records_a_failure_clearly() -> None:
    bad = EvalCase(
        case_id="bad",
        claim_text="A claim.",
        expected_verdict="SUPPORTED",
        subclaims=(
            EvalSubclaim(
                ordinal=1,
                text="A subclaim.",
                claim_type="factual",
                links=(EvalLink("NOPE", 1.0, 1.0, 1.0, 1.0, 1.0),),
            ),
        ),
    )
    report = evaluate_dataset(_dataset((bad,)), generated_at=FIXED_TIME)
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert rows[0]["error"]
    assert rows[0]["predicted_verdict"] == ""


def test_write_artifacts_creates_both_files(tmp_path: Path) -> None:
    written = write_artifacts(_report(), tmp_path)
    assert written.json_path.exists()
    assert written.csv_path.exists()
    assert written.json_path.suffix == ".json"
    assert written.csv_path.suffix == ".csv"


def test_written_json_round_trips(tmp_path: Path) -> None:
    written = write_artifacts(_report(), tmp_path)
    payload = json.loads(written.json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_cases"] > 0


def test_written_csv_round_trips(tmp_path: Path) -> None:
    written = write_artifacts(_report(), tmp_path)
    with written.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_mentions_the_key_numbers() -> None:
    report = _report()
    text = format_summary(report)
    assert "Accuracy" in text
    assert "Macro F1" in text
    assert "Confusion matrix" in text
    assert str(report.total_cases) in text


def test_summary_states_the_dataset_limitation() -> None:
    """The summary must not let a reader mistake this for a benchmark."""
    text = format_summary(_report()).lower()
    assert "not a statistically representative benchmark" in text
    assert "demonstration" in text


def test_summary_lists_failures() -> None:
    bad = EvalCase(
        case_id="bad",
        claim_text="A claim.",
        expected_verdict="SUPPORTED",
        subclaims=(
            EvalSubclaim(
                ordinal=1,
                text="A subclaim.",
                claim_type="factual",
                links=(EvalLink("NOPE", 1.0, 1.0, 1.0, 1.0, 1.0),),
            ),
        ),
    )
    text = format_summary(evaluate_dataset(_dataset((bad,)), generated_at=FIXED_TIME))
    assert "Failures" in text
    assert "bad" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_runs_and_writes_artifacts(tmp_path: Path) -> None:
    exit_code = main(["--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert list(tmp_path.glob("eval_*.json"))
    assert list(tmp_path.glob("eval_*.csv"))


def test_cli_no_write_produces_no_files(tmp_path: Path) -> None:
    exit_code = main(["--output-dir", str(tmp_path), "--no-write"])
    assert exit_code == 0
    assert not list(tmp_path.glob("eval_*"))


def test_cli_returns_error_for_missing_dataset(tmp_path: Path) -> None:
    exit_code = main(["--dataset", str(tmp_path / "missing.json"), "--no-write"])
    assert exit_code == 2


def test_cli_accepts_an_explicit_dataset(tmp_path: Path) -> None:
    path = _write(tmp_path, {"cases": [_minimal_case()]})
    exit_code = main(["--dataset", str(path), "--no-write"])
    assert exit_code == 0


def test_default_dataset_path_exists() -> None:
    assert DEFAULT_DATASET_PATH.exists()
