"""
Metrics aggregation.

Turns a list of CaseResult into a single serializable scoreboard: headline
metrics, per-slice breakdowns, and per-case detail.

The headline metric is **published_falsehood_rate**: of every claim the reviewer
marked SUPPORTED, the fraction the source does NOT actually support. That is the
number a marketing org cares about, because it is the rate at which the agent
would wave a false claim through to publication. A trustworthy reviewer keeps it
at zero, even at the cost of flagging more claims for a human (which is the
``over_flag_rate``, the productivity tax, reported right next to it so both
failure directions are visible).

Verdict-based metrics are computed deterministically from the gold labels (no
judge needed). The judge adds the two semantic axes the labels cannot:
faithfulness and citation correctness.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Optional

from .dataset import slices_for
from .runner import CaseResult
from .schema import Verdict

PASS_THRESHOLD = 1.0  # normalized judge score (i.e. a 2/2) required to count as "good"


def _safe_mean(xs: list[float]) -> Optional[float]:
    return mean(xs) if xs else None


@dataclass
class SliceMetrics:
    n: int
    deterministic_pass_rate: Optional[float]
    verdict_accuracy: Optional[float]
    published_falsehood_rate: Optional[float]
    citation_correctness: Optional[float]
    retrieval_recall_at_k: Optional[float]


@dataclass
class Scoreboard:
    n_cases: int
    headline: dict[str, Optional[float]]
    slices: dict[str, dict]
    per_case: list[dict]
    judge_used: bool
    judge_config: Optional[dict] = None   # model id + prompt fingerprint, for reproducibility

    def to_dict(self) -> dict:
        return asdict(self)


def _judge_good(result: CaseResult, axis: str) -> Optional[bool]:
    v = result.verdict
    if v is None or v.skipped:
        return None
    norm = v.axis_norm(axis)
    if norm is None:
        return None
    return norm >= PASS_THRESHOLD


def _collect_judge(results: Iterable[CaseResult], axis: str) -> list[float]:
    vals: list[float] = []
    for r in results:
        good = _judge_good(r, axis)
        if good is not None:
            vals.append(1.0 if good else 0.0)
    return vals


def _is_published_falsehood(r: CaseResult) -> bool:
    """A claim the reviewer cleared to publish (verdict SUPPORTED) that should
    not have been cleared. It counts if the source does not actually support it
    (gold is not SUPPORTED), or if it was cleared on a broken trust contract: a
    failed deterministic check, or a missing or unresolvable citation. Folding
    the citation and structural failures in here is deliberate, so a SUPPORTED
    verdict with a bogus citation cannot keep the headline at zero."""
    if r.output.verdict is not Verdict.SUPPORTED:
        return False
    if r.case.gold_verdict is not Verdict.SUPPORTED:
        return True
    if not r.deterministic_passed:
        return True
    retrieved = set(r.output.retrieved_ids)
    if not r.output.citations or any(c.doc_id not in retrieved for c in r.output.citations):
        return True
    return False


def _published_falsehood_rate(results: list[CaseResult]) -> Optional[float]:
    supported_preds = [r for r in results if r.output.verdict is Verdict.SUPPORTED]
    if not supported_preds:
        # no claims were cleared to publish, so none were cleared falsely.
        return 0.0
    bad = [r for r in supported_preds if _is_published_falsehood(r)]
    return len(bad) / len(supported_preds)


def aggregate(results: list[CaseResult]) -> Scoreboard:
    judge_used = any(r.verdict is not None for r in results)

    det_pass = [1.0 if r.deterministic_passed else 0.0 for r in results]
    verdict_acc = [1.0 if r.verdict_correct else 0.0 for r in results]

    gold_contradicted = [r for r in results if r.case.gold_verdict is Verdict.CONTRADICTED]
    contradiction_recall = [
        1.0 if r.output.verdict is Verdict.CONTRADICTED else 0.0 for r in gold_contradicted
    ]
    gold_supported = [r for r in results if r.case.gold_verdict is Verdict.SUPPORTED]
    over_flag = [1.0 if r.output.verdict is not Verdict.SUPPORTED else 0.0 for r in gold_supported]
    must_abstain = [r for r in results if r.case.must_abstain]
    abstention_recall = [1.0 if r.output.abstained else 0.0 for r in must_abstain]

    faith = _collect_judge(results, "faithfulness")
    cit = _collect_judge(results, "citation_correctness")

    groundable = [r for r in results if r.case.gold_doc_ids]
    recall = [r.retrieval.recall_at_k for r in groundable]
    mrr = [r.retrieval.mrr for r in groundable]

    headline = {
        "deterministic_pass_rate": _safe_mean(det_pass),
        "published_falsehood_rate": _published_falsehood_rate(results),
        "verdict_accuracy": _safe_mean(verdict_acc),
        "contradiction_recall": _safe_mean(contradiction_recall),
        "abstention_recall": _safe_mean(abstention_recall),
        "over_flag_rate": _safe_mean(over_flag),
        "faithfulness_rate": _safe_mean(faith),
        "citation_correctness": _safe_mean(cit),
        "retrieval_recall_at_k": _safe_mean(recall),
        "rerank_mrr": _safe_mean(mrr),
    }

    # ---- per-slice -------------------------------------------------------- #
    by_slice: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        for label in slices_for(r.case):
            by_slice[label].append(r)

    slices: dict[str, dict] = {}
    for label, rs in sorted(by_slice.items()):
        sm = SliceMetrics(
            n=len(rs),
            deterministic_pass_rate=_safe_mean([1.0 if r.deterministic_passed else 0.0 for r in rs]),
            verdict_accuracy=_safe_mean([1.0 if r.verdict_correct else 0.0 for r in rs]),
            published_falsehood_rate=_published_falsehood_rate(rs),
            citation_correctness=_safe_mean(_collect_judge(rs, "citation_correctness")),
            retrieval_recall_at_k=_safe_mean(
                [r.retrieval.recall_at_k for r in rs if r.case.gold_doc_ids]),
        )
        slices[label] = asdict(sm)

    # ---- per-case detail -------------------------------------------------- #
    per_case = []
    for r in results:
        entry = {
            "id": r.case.id,
            "split": r.case.split,
            "gold_verdict": r.case.gold_verdict.value,
            "predicted_verdict": r.output.verdict.value,
            "verdict_correct": r.verdict_correct,
            "published_falsehood": _is_published_falsehood(r),
            "deterministic_passed": r.deterministic_passed,
            "failed_checks": [c.name for c in r.checks if not c.passed],
            "abstained": r.output.abstained,
            "retrieval_recall_at_k": r.retrieval.recall_at_k,
            "error": r.error,
        }
        if r.verdict and not r.verdict.skipped:
            entry["judge"] = {
                ax: {"score": a.score, "justification": a.justification}
                for ax, a in r.verdict.axes.items()
            }
        per_case.append(entry)

    # record which judge produced these verdicts, so a baseline is comparable
    # only against runs from the same pinned judge.
    judge_config = None
    for r in results:
        if r.verdict and not r.verdict.skipped:
            judge_config = {"judge_model_id": r.verdict.judge_model_id,
                            "prompt_fingerprint": r.verdict.prompt_fingerprint}
            break

    return Scoreboard(
        n_cases=len(results),
        headline=headline,
        slices=slices,
        per_case=per_case,
        judge_used=judge_used,
        judge_config=judge_config,
    )
