"""
CLI for the claims-review harness.

Commands
--------
  review      review a piece of marketing copy against the corpus and print a
              publish-readiness report (the live demo)
  run         run the eval suite, print a summary, optionally write a report
  gate        run the suite and exit non-zero if the CI gate blocks
  baseline    run the suite and save the scoreboard as the new green baseline
  drift       run the frozen split, append to the time series, alarm on drift
  calibrate   run the judge against the operator's hand labels, report agreement

The reviewer and judge default to the in-process fakes so every command runs out
of the box. Point --adapter at your own ReviewerAdapter factory (module:function)
to evaluate a real reviewer; point --judge at a real JudgeClient factory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .agent import ReferenceReviewer, ReviewerAdapter, split_into_claims
from .corpus import Corpus
from .dataset import filter_cases, load_dir, validate_against_corpus
from .drift import run_drift_check
from .gating import evaluate_gate, load_baseline, save_baseline
from .judges.calibration import calibrate, load_human_labels
from .judges.client import FakeJudgeClient, JudgeClient, JudgeConfig
from .judges.rubric_judge import judge_case
from .metrics import aggregate
from .report import console_summary, markdown_report
from .runner import run_suite
from .schema import Verdict

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = str(_ROOT / "corpus" / "stripe_docs.jsonl")
DEFAULT_DATA = str(_ROOT / "data" / "golden")
DEFAULT_CALIB = str(_ROOT / "data" / "calibration" / "human_labels.jsonl")


# --------------------------------------------------------------------------- #
# Factory resolution                                                           #
# --------------------------------------------------------------------------- #
def _load_corpus(args) -> Corpus:
    return Corpus.load(args.corpus)


def _load_adapter(spec: str | None, corpus: Corpus) -> ReviewerAdapter:
    if not spec:
        return ReferenceReviewer(corpus)
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise SystemExit(f"--adapter must be 'module:factory', got {spec!r}")
    factory = getattr(importlib.import_module(mod_name), fn_name)
    obj = factory()
    if not isinstance(obj, ReviewerAdapter):
        raise SystemExit(f"{spec} did not return a ReviewerAdapter")
    return obj


def _load_judge(spec: str | None, enabled: bool) -> JudgeClient | None:
    if not enabled:
        return None
    if not spec:
        return FakeJudgeClient(JudgeConfig())
    mod_name, _, fn_name = spec.partition(":")
    factory = getattr(importlib.import_module(mod_name), fn_name)
    obj = factory()
    if not isinstance(obj, JudgeClient):
        raise SystemExit(f"{spec} did not return a JudgeClient")
    return obj


def _load_cases(args, corpus: Corpus):
    cases = load_dir(args.data)
    validate_against_corpus(cases, corpus)
    if getattr(args, "split", None):
        cases = filter_cases(cases, split=args.split)
    if not cases:
        raise SystemExit(f"no cases loaded from {args.data}")
    return cases


# --------------------------------------------------------------------------- #
# Commands                                                                      #
# --------------------------------------------------------------------------- #
def cmd_review(args) -> int:
    corpus = _load_corpus(args)
    reviewer = ReferenceReviewer(corpus)

    if args.text:
        copy = args.text
    elif args.file:
        copy = Path(args.file).read_text(encoding="utf-8")
    else:
        copy = sys.stdin.read()
    claims = split_into_claims(copy)
    if not claims:
        raise SystemExit("no reviewable claims found in the copy")

    icon = {Verdict.SUPPORTED: "[OK ]", Verdict.CONTRADICTED: "[!! ]", Verdict.UNSUPPORTED: "[ ? ]"}
    blocking = []
    print(f"\nReviewing {len(claims)} claim(s) against {len(corpus)} source passages "
          f"(corpus: {Path(args.corpus).name})\n")
    for i, claim in enumerate(claims, 1):
        out = reviewer.review_claim(claim)
        if out.verdict is not Verdict.SUPPORTED:
            blocking.append((claim, out.verdict))
        print(f"{icon[out.verdict]} claim {i}: {claim}")
        print(f"        verdict: {out.verdict.value.upper()} - {out.rationale}")
        for c in out.citations:
            doc = corpus.get(c.doc_id)
            if doc:
                print(f"        source : [{doc.id}] {doc.source_url} (captured {doc.captured_on})")
                print(f"                 \"{doc.text}\"")
        print()

    if blocking:
        print(f"VERDICT: HOLD - {len(blocking)} of {len(claims)} claim(s) are not cleared to publish:")
        for claim, v in blocking:
            print(f"  - [{v.value}] {claim}")
        return 1
    print(f"VERDICT: PUBLISH - all {len(claims)} claim(s) are supported by a cited source.")
    return 0


def cmd_run(args) -> int:
    corpus = _load_corpus(args)
    cases = _load_cases(args, corpus)
    adapter = _load_adapter(args.adapter, corpus)
    judge = _load_judge(args.judge, enabled=not args.no_judge)
    results = run_suite(adapter, cases, corpus, judge=judge, recall_k=args.k)
    sb = aggregate(results)
    baseline = load_baseline(args.baseline) if args.baseline else None
    gate = evaluate_gate(sb, baseline) if args.baseline else None
    print(console_summary(sb, gate))
    if args.report:
        Path(args.report).write_text(markdown_report(sb, gate, baseline), encoding="utf-8")
        print(f"\nwrote report -> {args.report}")
    if args.out:
        Path(args.out).write_text(json.dumps(sb.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote scoreboard -> {args.out}")
    return 0


def cmd_gate(args) -> int:
    corpus = _load_corpus(args)
    cases = _load_cases(args, corpus)
    adapter = _load_adapter(args.adapter, corpus)
    judge = _load_judge(args.judge, enabled=not args.no_judge)
    results = run_suite(adapter, cases, corpus, judge=judge, recall_k=args.k)
    sb = aggregate(results)
    if not Path(args.baseline).exists():
        raise SystemExit(
            f"gate: baseline {args.baseline!r} not found. Create one with "
            f"`claimcheck baseline --out {args.baseline}` on a known-green commit."
        )
    baseline = load_baseline(args.baseline)
    gate = evaluate_gate(sb, baseline)
    print(console_summary(sb, gate))
    if args.report:
        Path(args.report).write_text(markdown_report(sb, gate, baseline), encoding="utf-8")
        print(f"\nwrote report -> {args.report}")
    return 0 if gate.passed else 1


def _baseline_not_green(sb) -> list[str]:
    """A baseline is the definition of correct, so it must clear a structural bar
    before it can be saved: deterministic checks all green, perfect verdict
    accuracy, zero published falsehoods, and full abstention recall. This stops a
    degenerate reviewer (for example, one that abstains on everything) from
    saving itself as the bar the gate then compares against. The judge axes are
    gated on delta versus this baseline, not on an absolute floor, because the
    bundled mechanical judge is deliberately imperfect."""
    h = sb.headline
    problems = []
    if (h.get("deterministic_pass_rate") or 0.0) < 1.0:
        problems.append(f"deterministic_pass_rate {h.get('deterministic_pass_rate')} != 1.0")
    if (h.get("verdict_accuracy") or 0.0) < 1.0:
        problems.append(f"verdict_accuracy {h.get('verdict_accuracy')} != 1.0")
    if (h.get("published_falsehood_rate") or 0.0) > 0.0:
        problems.append(f"published_falsehood_rate {h.get('published_falsehood_rate')} != 0.0")
    ar = h.get("abstention_recall")
    if ar is not None and ar < 1.0:
        problems.append(f"abstention_recall {ar} != 1.0")
    return problems


def cmd_baseline(args) -> int:
    corpus = _load_corpus(args)
    cases = _load_cases(args, corpus)
    adapter = _load_adapter(args.adapter, corpus)
    judge = _load_judge(args.judge, enabled=not args.no_judge)
    results = run_suite(adapter, cases, corpus, judge=judge, recall_k=args.k)
    sb = aggregate(results)
    print(console_summary(sb))
    problems = _baseline_not_green(sb)
    if problems and not args.force:
        raise SystemExit(
            "\nrefusing to save a baseline that does not clear the structural bar "
            "(deterministic, verdict accuracy, zero published falsehoods, abstention "
            "recall):\n  - "
            + "\n  - ".join(problems)
            + "\nJudge axes are gated on delta vs this baseline, not absolutely. "
            "Pass --force only if you know why."
        )
    save_baseline(sb, args.out)
    print(f"\nsaved baseline -> {args.out}")
    return 0


def cmd_drift(args) -> int:
    corpus = _load_corpus(args)
    cases = filter_cases(load_dir(args.data), split="frozen")
    validate_against_corpus(cases, corpus)
    if not cases:
        raise SystemExit("no frozen-split cases found for drift check")
    adapter = _load_adapter(args.adapter, corpus)
    judge = _load_judge(args.judge, enabled=not args.no_judge)
    results = run_suite(adapter, cases, corpus, judge=judge, recall_k=args.k)
    sb = aggregate(results)
    alerts: list[str] = []
    notify = (lambda m: (alerts.append(m), print("ALERT:", m))) if not args.quiet else alerts.append
    run_drift_check(sb, args.timeseries, window=args.window, notify=notify)
    print(console_summary(sb))
    print(f"\ndrift alerts: {len(alerts)}")
    return 0


def cmd_calibrate(args) -> int:
    corpus = _load_corpus(args)
    cases = _load_cases(args, corpus)
    adapter = _load_adapter(args.adapter, corpus)
    results = run_suite(adapter, cases, corpus, judge=None)
    cases_by_id = {r.case.id: r.case for r in results}
    outputs_by_id = {r.case.id: r.output for r in results}
    judge = _load_judge(args.judge, enabled=True)
    labels = load_human_labels(args.labels)
    if not labels:
        raise SystemExit(f"no human labels found at {args.labels}")
    rep = calibrate(cases_by_id, outputs_by_id, labels, corpus, judge, min_exact=args.min_exact)
    print(f"judge-vs-human calibration  (n={rep.n_labels})")
    print(f"  overall exact agreement: {rep.overall_exact:.3f} "
          f"(min required {args.min_exact:.3f}) -> "
          f"{'TRUSTED' if rep.passes(args.min_exact) else 'RETUNE JUDGE'}")
    for axis, a in rep.per_axis.items():
        print(f"  {axis:<22} exact={a.exact_agreement:.3f} "
              f"within1={a.within_one:.3f} mae={a.mean_abs_error:.3f} (n={a.n})")
    return 0 if rep.passes(args.min_exact) else 2


# --------------------------------------------------------------------------- #
# Arg parsing                                                                   #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="claimcheck", description="marketing claims-review eval harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    def corpus_arg(sp):
        sp.add_argument("--corpus", default=DEFAULT_CORPUS, help="source-of-truth corpus jsonl")

    def common(sp):
        corpus_arg(sp)
        sp.add_argument("--data", default=DEFAULT_DATA, help="golden dataset dir or file")
        sp.add_argument("--adapter", default=None, help="module:factory -> ReviewerAdapter")
        sp.add_argument("--judge", default=None, help="module:factory -> JudgeClient")
        sp.add_argument("--no-judge", action="store_true", help="skip the model judge")
        sp.add_argument("-k", type=int, default=4, help="recall@k cutoff")

    sp = sub.add_parser("review", help="review a piece of copy against the corpus")
    corpus_arg(sp)
    sp.add_argument("--text", help="the copy to review (inline)")
    sp.add_argument("--file", help="a file of copy to review")
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("run", help="run the suite and print a summary")
    common(sp); sp.add_argument("--split"); sp.add_argument("--baseline")
    sp.add_argument("--report"); sp.add_argument("--out")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("gate", help="run the suite, exit non-zero if gate blocks")
    common(sp); sp.add_argument("--split")
    sp.add_argument("--baseline", required=True); sp.add_argument("--report")
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("baseline", help="run the suite and save a green baseline")
    common(sp); sp.add_argument("--split")
    sp.add_argument("--out", required=True)
    sp.add_argument("--force", action="store_true", help="save even if the run is not green")
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("drift", help="run frozen split, log time series, alarm on drift")
    common(sp)
    sp.add_argument("--timeseries", required=True)
    sp.add_argument("--window", type=int, default=7)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_drift)

    sp = sub.add_parser("calibrate", help="judge-vs-human agreement on labeled subset")
    common(sp); sp.add_argument("--split")
    sp.add_argument("--labels", default=DEFAULT_CALIB)
    sp.add_argument("--min-exact", type=float, default=0.7)
    sp.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
