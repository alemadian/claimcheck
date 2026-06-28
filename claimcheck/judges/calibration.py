"""
Judge calibration tracker.

The judge's verdicts are trusted only while it agrees with the operator's own
hand labels. This loads a hand-labeled subset, re-runs the judge on those same
cases, and reports judge-vs-human agreement per axis. If agreement drops below a
band, the judge prompt needs re-tuning before its scores are trusted again.

Calibration file format (one object per line)
---------------------------------------------
{"case_id": "pricing-supported-0001", "axis": "citation_correctness", "human_score": 2}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..corpus import Corpus
from ..schema import ClaimCase, ReviewOutput
from .client import JudgeClient
from .rubric_judge import judge_case


@dataclass
class AxisAgreement:
    axis: str
    n: int
    exact_agreement: float
    within_one: float
    mean_abs_error: float


@dataclass
class CalibrationReport:
    per_axis: dict[str, AxisAgreement]
    overall_exact: float
    n_labels: int

    def passes(self, min_exact: float) -> bool:
        return self.overall_exact >= min_exact


def load_human_labels(path: str | Path) -> dict[tuple[str, str], int]:
    labels: dict[tuple[str, str], int] = {}
    p = Path(path)
    if not p.exists():
        return labels
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels[(obj["case_id"], obj["axis"])] = int(obj["human_score"])
    return labels


def calibrate(
    cases_by_id: dict[str, ClaimCase],
    outputs_by_id: dict[str, ReviewOutput],
    human_labels: dict[tuple[str, str], int],
    corpus: Corpus,
    client: JudgeClient,
    min_exact: float = 0.7,
) -> CalibrationReport:
    buckets: dict[str, list[tuple[int, int]]] = {}
    for (case_id, axis), human in human_labels.items():
        case = cases_by_id.get(case_id)
        out = outputs_by_id.get(case_id)
        if case is None or out is None:
            continue
        verdict = judge_case(case, out, corpus, client)
        axis_score = verdict.axes.get(axis)
        if axis_score is None:
            continue
        buckets.setdefault(axis, []).append((axis_score.score, human))

    per_axis: dict[str, AxisAgreement] = {}
    all_pairs: list[tuple[int, int]] = []
    for axis, pairs in buckets.items():
        all_pairs.extend(pairs)
        n = len(pairs)
        exact = sum(1 for j, h in pairs if j == h) / n if n else 0.0
        w1 = sum(1 for j, h in pairs if abs(j - h) <= 1) / n if n else 0.0
        mae = sum(abs(j - h) for j, h in pairs) / n if n else 0.0
        per_axis[axis] = AxisAgreement(axis, n, exact, w1, mae)

    overall = (sum(1 for j, h in all_pairs if j == h) / len(all_pairs)) if all_pairs else 0.0
    return CalibrationReport(per_axis=per_axis, overall_exact=overall, n_labels=len(all_pairs))
