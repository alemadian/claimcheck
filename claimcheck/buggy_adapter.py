"""
Deliberately buggy adapter used only to prove the CI gate has teeth.

The production/reference reviewer stays conservative. This adapter wraps it and
injects four documented failure modes when a golden case id carries a marker:

  - "leakverdict"  : leaks the gold verdict into the prompt.
  - "phantomcite"  : cites a passage it never retrieved.
  - "rubberstamp"  : marks an ungrounded claim SUPPORTED.
  - "greenlight"   : vouches for an out-of-corpus claim instead of abstaining.

Keeping these failures in a separate adapter avoids test-specific behavior in
the reviewer a hiring manager will inspect first, while preserving the public
"gate has teeth" demo.
"""

from __future__ import annotations

from pathlib import Path

from .agent import ReferenceReviewer, overlap_score
from .corpus import Corpus
from .schema import Citation, ClaimCase, ReviewOutput, Verdict

BUG_MARKERS = ("leakverdict", "phantomcite", "rubberstamp", "greenlight")


class BuggyReviewerAdapter(ReferenceReviewer):
    name = "deliberately-buggy-reviewer"

    def review(self, case: ClaimCase) -> ReviewOutput:
        bugs = {marker for marker in BUG_MARKERS if marker in case.id}
        if not bugs:
            return super().review(case)

        if "greenlight" in bugs:
            retrieved = self.retriever.retrieve(case.claim, self.corpus, self.k)
            if not retrieved:
                best = max(self.corpus.docs, key=lambda d: (overlap_score(case.claim, d.text), d.id))
                prompt = (
                    "Decide whether the SOURCES support, contradict, or fail to ground "
                    "the CLAIM. Cite only the passages you used.\n\n"
                    f"SOURCES:\n\nCLAIM: {case.claim}\n"
                )
                return ReviewOutput(
                    verdict=Verdict.SUPPORTED,
                    citations=[Citation(doc_id=best.id, span=best.text)],
                    retrieved_ids=[],
                    abstained=False,
                    model_was_called=True,
                    prompt_to_model=prompt,
                    rationale="(buggy demo) vouched without grounding.",
                )

        out = super().review(case)
        verdict = out.verdict
        citations = list(out.citations)
        retrieved_ids = list(out.retrieved_ids)
        prompt = out.prompt_to_model
        rationale = out.rationale
        model_was_called = out.model_was_called

        if "leakverdict" in bugs and prompt is not None:
            prompt += f"\n(reviewer hint: the correct verdict is {case.gold_verdict.value})"

        if "rubberstamp" in bugs:
            retrieved = self.retriever.retrieve(case.claim, self.corpus, self.k)
            if retrieved:
                top = retrieved[0]
                verdict = Verdict.SUPPORTED
                citations = [Citation(doc_id=top.id, span=top.text)]
                retrieved_ids = [d.id for d in retrieved]
                model_was_called = True
                rationale = "(buggy demo) rubber-stamped against a loosely related passage."

        if "phantomcite" in bugs:
            citations.append(Citation(doc_id="__phantom__", span="not retrieved"))

        return ReviewOutput(
            verdict=verdict,
            citations=citations,
            retrieved_ids=retrieved_ids,
            abstained=verdict is Verdict.UNSUPPORTED,
            model_was_called=model_was_called,
            prompt_to_model=prompt,
            rationale=rationale,
        )


def make_adapter() -> BuggyReviewerAdapter:
    root = Path(__file__).resolve().parent.parent
    return BuggyReviewerAdapter(Corpus.load(root / "corpus" / "stripe_docs.jsonl"))
