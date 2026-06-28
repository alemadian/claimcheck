"""
Layer 1: deterministic trust checks. No model in the loop. Sub-second.

These gate the build on every commit and must be 100% green to merge. Each
check returns a CheckResult with a hard pass/fail and a human-readable reason.
There is zero nondeterminism: same input, same verdict, always.

These checks encode the invariants a *trustworthy* reviewer must never break,
independent of whether its verdict happens to be correct:

- grounding_isolation     : the prompt the model saw contained the claim and the
                            retrieved passages and did NOT have the answer leaked
                            into it.
- citation_resolvability  : every citation resolves to a passage that was
                            actually retrieved (no invented pointers).
- cite_or_abstain         : a SUPPORTED/CONTRADICTED verdict carries a citation;
                            an UNSUPPORTED verdict carries none. This is the
                            core of the trust layer: never vouch without a
                            source, never "flag-but-cite-support".
- fail_closed             : on a claim nothing can ground, the agent abstains,
                            and if there was no context at all it did not spend a
                            model call (no spend before grounding).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .schema import ClaimCase, ReviewOutput, Verdict

# the marker a leak would introduce; the prompt legitimately contains the words
# "support"/"contradict" in its instruction, so we detect an injected ANSWER,
# not the mere presence of a verdict word.
_LEAK_MARKERS = ("correct verdict is", "reviewer hint", "the answer is")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str

    def __bool__(self) -> bool:  # so `all(results)` works
        return self.passed


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def check_grounding_isolation(case: ClaimCase, out: ReviewOutput, corpus: Corpus) -> CheckResult:
    name = "grounding_isolation"
    # Check whenever a model was actually called, even if the agent then
    # abstained: an adapter must not be able to skip isolation by abstaining.
    if not out.model_was_called:
        return CheckResult(name, True, "no model call; nothing to leak")
    prompt = out.prompt_to_model
    if prompt is None:
        return CheckResult(name, False, "model was called but prompt was not recorded")

    np = _normalize(prompt)
    if _normalize(case.claim) not in np:
        return CheckResult(name, False, "prompt does not contain the claim")

    # every retrieved passage's actual TEXT must be present (proves the verdict
    # was formed from retrieved context, not from the label).
    for rid in out.retrieved_ids:
        doc = corpus.get(rid)
        if doc is None:
            return CheckResult(name, False, f"retrieved id {rid!r} is not in the corpus")
        body = _normalize(doc.text)
        probe = body[:120] if len(body) > 120 else body
        if probe and probe not in np:
            return CheckResult(
                name, False,
                f"retrieved passage {rid!r} text not present in prompt "
                f"(id present is not sufficient)",
            )

    for marker in _LEAK_MARKERS:
        if marker in np:
            return CheckResult(name, False, f"gold verdict leaked into the prompt (marker {marker!r})")
    return CheckResult(name, True, "prompt = claim + retrieved passages only")


def check_citation_resolvability(case: ClaimCase, out: ReviewOutput, corpus: Corpus) -> CheckResult:
    name = "citation_resolvability"
    if out.abstained:
        if out.citations:
            return CheckResult(name, False, "abstained but still emitted citations")
        return CheckResult(name, True, "abstained, no citations")
    retrieved = set(out.retrieved_ids)
    bad = [c.doc_id for c in out.citations if c.doc_id not in retrieved]
    if bad:
        return CheckResult(
            name, False, f"citation(s) to non-retrieved passage(s): {sorted(set(bad))}"
        )
    # A quoted span must actually appear in the cited passage. This is what makes
    # "cite the exact passage" real and not a fabricated quote.
    for c in out.citations:
        if not c.span:
            continue
        doc = corpus.get(c.doc_id)
        if doc and _normalize(c.span) not in _normalize(doc.text):
            return CheckResult(name, False, f"citation span not found in passage {c.doc_id!r}")
    # a grounded verdict must quote a real, non-empty span (no bare doc pointer).
    if out.verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED):
        empty = sorted({c.doc_id for c in out.citations if not c.span.strip()})
        if empty:
            return CheckResult(name, False, f"grounded verdict with an empty citation span for: {empty}")
    return CheckResult(name, True, "all citations resolve to retrieved passages")


def check_cite_or_abstain(case: ClaimCase, out: ReviewOutput, corpus: Corpus) -> CheckResult:
    name = "cite_or_abstain"
    if out.verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED):
        if not out.citations:
            return CheckResult(
                name, False,
                f"{out.verdict.value} verdict with no citation (vouched without a source)",
            )
        return CheckResult(name, True, f"{out.verdict.value} verdict carries a citation")
    # UNSUPPORTED: the agent declined to vouch, so it must not present support.
    if out.citations:
        return CheckResult(
            name, False,
            "unsupported verdict but still cited a source (incoherent: declined yet cited)",
        )
    return CheckResult(name, True, "unsupported verdict, correctly carries no citation")


def check_fail_closed(case: ClaimCase, out: ReviewOutput, corpus: Corpus) -> CheckResult:
    name = "fail_closed"
    if not case.must_abstain:
        return CheckResult(name, True, "claim is groundable; check not applicable")
    if not out.abstained:
        return CheckResult(
            name, False,
            f"ungroundable claim but agent returned {out.verdict.value} (vouched on no support)",
        )
    if not out.retrieved_ids and out.model_was_called:
        return CheckResult(
            name, False, "abstained but called the model on empty context (spend before grounding)"
        )
    return CheckResult(name, True, "correctly failed closed")


ALL_CHECKS = (
    check_grounding_isolation,
    check_citation_resolvability,
    check_cite_or_abstain,
    check_fail_closed,
)


def run_deterministic(case: ClaimCase, out: ReviewOutput, corpus: Corpus) -> list[CheckResult]:
    return [check(case, out, corpus) for check in ALL_CHECKS]
