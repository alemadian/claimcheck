"""
The runner: drives the reviewer (via an adapter) over a set of golden cases and
collects, per case, the result of every layer.

Order is cheapest-first. Deterministic checks always run. The model judge runs
only when a judge is supplied (so per-commit runs stay free and fast, and
nightly / pre-merge runs pay for the judge). An adapter crash is recorded as a
case failure, not a harness crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agent import ReviewerAdapter
from .corpus import Corpus
from .deterministic import CheckResult, run_deterministic
from .judges.client import JudgeClient
from .judges.rubric_judge import JudgeVerdict, judge_case
from .retrieval import RetrievalScore, score_retrieval
from .schema import ClaimCase, ReviewOutput, Verdict


@dataclass
class CaseResult:
    case: ClaimCase
    output: ReviewOutput
    checks: list[CheckResult]
    retrieval: RetrievalScore
    verdict: Optional[JudgeVerdict] = None
    error: Optional[str] = None

    @property
    def deterministic_passed(self) -> bool:
        # an adapter error is a hard deterministic failure; an empty checks list
        # (only produced on an adapter crash) must NOT count as a pass.
        if self.error:
            return False
        if not self.checks:
            return False
        return all(c.passed for c in self.checks)

    @property
    def verdict_correct(self) -> bool:
        return self.output.verdict is self.case.gold_verdict


def run_case(
    adapter: ReviewerAdapter,
    case: ClaimCase,
    corpus: Corpus,
    *,
    judge: Optional[JudgeClient] = None,
    recall_k: int = 4,
) -> CaseResult:
    try:
        out = adapter.review(case)
    except Exception as exc:  # an adapter crash is a case failure, not a harness crash
        empty = ReviewOutput(verdict=Verdict.UNSUPPORTED, citations=[], retrieved_ids=[],
                             abstained=True, model_was_called=False)
        return CaseResult(case=case, output=empty, checks=[],
                          retrieval=score_retrieval(case, empty, recall_k),
                          error=f"adapter raised: {exc!r}")

    checks = run_deterministic(case, out, corpus)
    retrieval = score_retrieval(case, out, recall_k)
    verdict = judge_case(case, out, corpus, judge) if judge is not None else None
    return CaseResult(case=case, output=out, checks=checks, retrieval=retrieval, verdict=verdict)


def run_suite(
    adapter: ReviewerAdapter,
    cases: list[ClaimCase],
    corpus: Corpus,
    *,
    judge: Optional[JudgeClient] = None,
    recall_k: int = 4,
) -> list[CaseResult]:
    return [run_case(adapter, c, corpus, judge=judge, recall_k=recall_k) for c in cases]
