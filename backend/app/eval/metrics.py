"""Deterministic multiclass classification metrics.

Standard library only. No sklearn, no numpy: the arithmetic is simple enough to
read and audit, and a dependency would obscure it for no benefit.

Conventions that matter for correctness:

* **Macro averaging** is over classes with non-zero support only. A class that
  never appears in the expected labels has an undefined recall, and averaging in
  a 0.0 for it would understate the score for no reason.
* **Zero denominators yield 0.0**, not an exception and not NaN. A class with no
  predictions has precision 0.0; a class with no support is excluded from macro
  averages entirely.
* **Labels are sorted** in the confusion matrix so the output is stable across
  runs and diffable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision, recall and F1 for one label."""

    label: str
    support: int
    predicted: int
    true_positives: int
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class MetricsReport:
    """The full metrics block for one evaluation run."""

    total: int
    correct: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view."""
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "labels": self.labels,
            "per_class": {
                label: {
                    "support": m.support,
                    "predicted": m.predicted,
                    "true_positives": m.true_positives,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                }
                for label, m in self.per_class.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that yields 0.0 rather than raising or producing NaN."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean, 0.0 when either side is zero."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_confusion_matrix(
    pairs: list[tuple[str, str]], labels: list[str] | None = None
) -> dict[str, dict[str, int]]:
    """Count expected -> predicted pairs.

    Args:
        pairs: ``(expected, predicted)`` for each case.
        labels: label order. Defaults to the sorted union of both sides, so a
            predicted label that never appears as expected still gets a row.

    Returns:
        ``{expected_label: {predicted_label: count}}`` with every cell present,
        so a reader can see zeros rather than infer them from absence.
    """
    if labels is None:
        labels = sorted({label for pair in pairs for label in pair})

    matrix: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }

    for expected, predicted in pairs:
        # A pair outside the label set would silently vanish; make it visible.
        if expected not in matrix:
            matrix[expected] = {label: 0 for label in labels}
        if predicted not in matrix[expected]:
            matrix[expected][predicted] = 0
        matrix[expected][predicted] += 1

    return matrix


def compute_metrics(
    pairs: list[tuple[str, str]], labels: list[str] | None = None
) -> MetricsReport:
    """Compute accuracy, macro precision/recall/F1, per-class metrics and the
    confusion matrix for a set of ``(expected, predicted)`` pairs.

    Args:
        pairs: one ``(expected, predicted)`` tuple per evaluated case.
        labels: label universe. Defaults to the sorted union of both sides.

    Returns:
        MetricsReport. An empty input yields all-zero metrics rather than an
        error, so a run with no successful cases still produces a valid report.
    """
    if labels is None:
        labels = sorted({label for pair in pairs for label in pair})

    total = len(pairs)
    correct = sum(1 for expected, predicted in pairs if expected == predicted)

    per_class: dict[str, ClassMetrics] = {}

    for label in labels:
        support = sum(1 for expected, _ in pairs if expected == label)
        predicted_count = sum(1 for _, predicted in pairs if predicted == label)
        true_positives = sum(
            1 for expected, predicted in pairs if expected == label and predicted == label
        )

        precision = _safe_div(true_positives, predicted_count)
        recall = _safe_div(true_positives, support)

        per_class[label] = ClassMetrics(
            label=label,
            support=support,
            predicted=predicted_count,
            true_positives=true_positives,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
        )

    # Macro averages over classes that actually occur in the expected labels.
    evaluated = [m for m in per_class.values() if m.support > 0]

    if evaluated:
        macro_precision = sum(m.precision for m in evaluated) / len(evaluated)
        macro_recall = sum(m.recall for m in evaluated) / len(evaluated)
        macro_f1 = sum(m.f1 for m in evaluated) / len(evaluated)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    return MetricsReport(
        total=total,
        correct=correct,
        accuracy=_safe_div(correct, total),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=build_confusion_matrix(pairs, labels),
        labels=list(labels),
    )


__all__ = [
    "ClassMetrics",
    "MetricsReport",
    "build_confusion_matrix",
    "compute_metrics",
]
