"""
CI regression gate.

Layered:

  Tier 1 (hard, absolute): every deterministic trust check must be 100% green.
          Any adapter error or failed check blocks the merge, full stop.

  Tier 2 (delta vs baseline): metrics are gated on movement against the last
          green baseline, not an absolute floor.
            - published_falsehood_rate and over_flag_rate may not RISE beyond
              tolerance (a higher published-falsehood rate is the one failure a
              marketing-review tool can never ship).
            - verdict_accuracy, contradiction_recall, abstention_recall,
              citation_correctness, faithfulness_rate and retrieval_recall_at_k
              may not DROP beyond tolerance.
            - no single previously-passing case may regress, and any new case
              added in the PR must pass.

The judge runs at temperature 0 / fixed seed so the comparison is
apples-to-apples (enforced by JudgeConfig, recorded in the baseline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .metrics import Scoreboard

# higher-is-better metrics: block when they DROP by more than the tolerance.
HIGHER_BETTER_TOL = {
    "verdict_accuracy": 0.01,
    "contradiction_recall": 0.0,        # catching a false price is non-negotiable
    "abstention_recall": 0.0,           # never vouch for the ungroundable
    "citation_correctness": 0.02,
    "faithfulness_rate": 0.02,
    "retrieval_recall_at_k": 0.02,
}
# lower-is-better metrics: block when they RISE by more than the tolerance.
LOWER_BETTER_TOL = {
    "published_falsehood_rate": 0.0,    # never allowed to rise
    "over_flag_rate": 0.02,
}


@dataclass
class GateFinding:
    severity: str   # "block" | "warn"
    message: str


@dataclass
class GateResult:
    passed: bool
    findings: list[GateFinding] = field(default_factory=list)

    def add(self, severity: str, message: str) -> None:
        self.findings.append(GateFinding(severity, message))
        if severity == "block":
            self.passed = False


def _case_passed(entry: dict) -> bool:
    """A case 'passes' if it has no error, no failed deterministic checks, the
    verdict matches gold, and (when judged) every judge axis is a full 2."""
    if entry.get("error"):
        return False
    if entry.get("failed_checks"):
        return False
    if not entry.get("verdict_correct", False):
        return False
    judge = entry.get("judge")
    if judge:
        for axis in ("faithfulness", "citation_correctness"):
            a = judge.get(axis)
            if a is not None and a["score"] < 2:
                return False
    return True


def evaluate_gate(
    scoreboard: Scoreboard,
    baseline: Optional[dict],
    *,
    higher_tol: dict[str, float] | None = None,
    lower_tol: dict[str, float] | None = None,
) -> GateResult:
    hi = {**HIGHER_BETTER_TOL, **(higher_tol or {})}
    lo = {**LOWER_BETTER_TOL, **(lower_tol or {})}
    result = GateResult(passed=True)

    # ---------- Tier 1: deterministic checks must be 100% green ----------- #
    det = scoreboard.headline.get("deterministic_pass_rate")
    if det is None or det < 1.0:
        failing = [c["id"] for c in scoreboard.per_case if c["failed_checks"] or c["error"]]
        result.add("block",
                   f"Tier 1 FAILED: deterministic trust checks not 100% green "
                   f"({det if det is not None else 0:.3f}). Offending cases: {failing}")
    else:
        result.add("warn", "Tier 1 OK: all deterministic trust checks green.")

    if baseline is None:
        result.add("warn", "No baseline supplied; Tier 2 delta checks skipped "
                           "(treating this run as the new baseline candidate).")
        return result

    # If the baseline was scored with the judge, a run that drops the judge is
    # not comparable and would silently remove the semantic layer.
    if baseline.get("judge_used") and not scoreboard.judge_used:
        result.add("block",
                   "Tier 2 FAILED: baseline was judged but this run has the judge "
                   "disabled; the semantic layer (faithfulness / citation correctness) "
                   "would be silently dropped.")

    base_head = baseline.get("headline", {})
    base_cases = {c["id"]: c for c in baseline.get("per_case", [])}

    # ---------- Tier 2a: per-metric delta vs baseline --------------------- #
    for metric, allowed_drop in hi.items():
        cur, old = scoreboard.headline.get(metric), base_head.get(metric)
        if cur is None or old is None:
            continue
        delta = cur - old
        if delta < -allowed_drop:
            result.add("block",
                       f"Tier 2 FAILED: {metric} dropped {delta:+.3f} "
                       f"(baseline {old:.3f} -> {cur:.3f}, tolerance {allowed_drop:.3f}).")
        elif delta < 0:
            result.add("warn", f"{metric} dipped {delta:+.3f} (within tolerance).")

    for metric, allowed_rise in lo.items():
        cur, old = scoreboard.headline.get(metric), base_head.get(metric)
        if cur is None or old is None:
            continue
        delta = cur - old
        if delta > allowed_rise:
            result.add("block",
                       f"Tier 2 FAILED: {metric} rose {delta:+.3f} "
                       f"(baseline {old:.3f} -> {cur:.3f}, tolerance {allowed_rise:.3f}).")
        elif delta > 0:
            result.add("warn", f"{metric} rose {delta:+.3f} (within tolerance).")

    # ---------- Tier 2b: no previously-passing case may regress ----------- #
    regressions, new_failures = [], []
    for entry in scoreboard.per_case:
        cid = entry["id"]
        now_ok = _case_passed(entry)
        if cid in base_cases:
            if _case_passed(base_cases[cid]) and not now_ok:
                regressions.append(cid)
        elif not now_ok:
            new_failures.append(cid)

    # a case that was in the baseline but is gone now is a silent coverage drop
    current_ids = {e["id"] for e in scoreboard.per_case}
    removed = sorted(cid for cid in base_cases if cid not in current_ids)

    if regressions:
        result.add("block", f"Tier 2 FAILED: previously-passing case(s) now failing: {regressions}")
    if new_failures:
        result.add("block", f"Tier 2 FAILED: new golden case(s) do not pass: {new_failures}")
    if removed:
        result.add("block", f"Tier 2 FAILED: case(s) in the baseline are missing from this run "
                            f"(coverage dropped): {removed}")
    if not regressions and not new_failures and not removed:
        result.add("warn", "Tier 2 OK: no per-case regressions, no dropped cases, all new cases pass.")

    return result


def load_baseline(path: str | Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_baseline(scoreboard: Scoreboard, path: str | Path) -> None:
    Path(path).write_text(json.dumps(scoreboard.to_dict(), indent=2), encoding="utf-8")
