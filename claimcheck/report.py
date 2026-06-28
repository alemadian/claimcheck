"""
Report / scoreboard artifact.

Renders a Scoreboard (and optionally a GateResult) as a compact console summary
and a Markdown report suitable for a CI build artifact and a PR comment. The
output leads with the headline numbers and the gate verdict, then the failing
cases (published falsehoods first, because that is the one that matters), then
per-slice, so a reviewer sees exactly what the harness concluded and why.
"""

from __future__ import annotations

from typing import Optional

from .gating import GateResult
from .metrics import Scoreboard

_HEADLINE_ORDER = [
    "deterministic_pass_rate",
    "published_falsehood_rate",
    "verdict_accuracy",
    "contradiction_recall",
    "abstention_recall",
    "over_flag_rate",
    "faithfulness_rate",
    "citation_correctness",
    "retrieval_recall_at_k",
    "rerank_mrr",
]


def _fmt(v: Optional[float]) -> str:
    return "  n/a" if v is None else f"{v:6.3f}"


def console_summary(sb: Scoreboard, gate: Optional[GateResult] = None) -> str:
    lines = [f"Cases: {sb.n_cases}   judge: {'on' if sb.judge_used else 'off'}", ""]
    for k in _HEADLINE_ORDER:
        lines.append(f"  {k:<26} {_fmt(sb.headline.get(k))}")
    falsehoods = [c["id"] for c in sb.per_case if c.get("published_falsehood")]
    if falsehoods:
        lines.append("")
        lines.append(f"  PUBLISHED FALSEHOODS ({len(falsehoods)}): {falsehoods}")
    if gate is not None:
        lines.append("")
        lines.append(f"GATE: {'PASS' if gate.passed else 'BLOCK'}")
        for f in gate.findings:
            mark = "x" if f.severity == "block" else "-"
            lines.append(f"  [{mark}] {f.message}")
    return "\n".join(lines)


def markdown_report(sb: Scoreboard, gate: Optional[GateResult] = None,
                    baseline: Optional[dict] = None) -> str:
    md: list[str] = []
    verdict = "n/a" if gate is None else ("PASS" if gate.passed else "BLOCK")
    md.append("## Claims-review eval report")
    md.append(f"**Gate: {verdict}**  -  {sb.n_cases} cases  -  "
              f"judge {'on' if sb.judge_used else 'off'}")
    md.append("")

    base_head = (baseline or {}).get("headline", {})
    md.append("| metric | value | vs baseline |")
    md.append("|---|---|---|")
    for k in _HEADLINE_ORDER:
        cur = sb.headline.get(k)
        old = base_head.get(k)
        if cur is None:
            val, delta = "n/a", ""
        else:
            val = f"{cur:.3f}"
            delta = "" if old is None else f"{cur - old:+.3f}"
        md.append(f"| {k} | {val} | {delta} |")
    md.append("")

    if gate is not None:
        md.append("### Gate findings")
        for f in gate.findings:
            icon = "BLOCK" if f.severity == "block" else "ok"
            md.append(f"- **{icon}** {f.message}")
        md.append("")

    falsehoods = [c for c in sb.per_case if c.get("published_falsehood")]
    if falsehoods:
        md.append("### Published falsehoods (claims wrongly cleared to publish)")
        for c in falsehoods:
            md.append(f"- `{c['id']}` predicted **{c['predicted_verdict']}** "
                      f"but gold is **{c['gold_verdict']}**")
        md.append("")

    failing = [c for c in sb.per_case
               if c["failed_checks"] or c.get("error") or not c["verdict_correct"]
               or _has_low_judge(c)]
    if failing:
        md.append("### Failing / low-scoring cases")
        for c in failing[:50]:
            bits = [f"`{c['id']}` (gold {c['gold_verdict']} / pred {c['predicted_verdict']})"]
            if c.get("error"):
                bits.append(f"error: {c['error']}")
            if c["failed_checks"]:
                bits.append(f"failed checks: {', '.join(c['failed_checks'])}")
            for ax, a in c.get("judge", {}).items():
                if a["score"] < 2:
                    bits.append(f"{ax}={a['score']} ({a['justification']})")
            md.append("- " + " - ".join(bits))
        md.append("")

    md.append("### Per-slice (deterministic / verdict acc / published-falsehood / recall@k)")
    md.append("| slice | n | det | verdict | falsehood | recall@k |")
    md.append("|---|---|---|---|---|---|")
    for label, s in sb.slices.items():
        md.append(f"| {label} | {s['n']} | "
                  f"{_md_fmt(s['deterministic_pass_rate'])} | "
                  f"{_md_fmt(s['verdict_accuracy'])} | "
                  f"{_md_fmt(s['published_falsehood_rate'])} | "
                  f"{_md_fmt(s['retrieval_recall_at_k'])} |")
    return "\n".join(md)


def _md_fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _has_low_judge(case_entry: dict) -> bool:
    return any(a["score"] < 2 for a in case_entry.get("judge", {}).values())
