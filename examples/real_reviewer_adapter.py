"""
Skeleton: wrapping a real (LLM-backed) content-review agent behind the
ReviewerAdapter contract, so the harness scores it without any other change.

The only hard rules your adapter must honor:
  1. Retrieve over the SAME pinned corpus the harness scores against, so the
     citations resolve and the run is reproducible.
  2. Populate the trust-accounting fields on ReviewOutput:
       - retrieved_ids   : every passage you actually pulled
       - model_was_called : False when you abstained without grounding (no spend)
       - prompt_to_model  : the exact string you sent (for grounding isolation)
  3. Enforce cite-or-abstain in your own prompt and output handling: never
     return SUPPORTED/CONTRADICTED without a citation to a retrieved passage,
     and never attach a support citation to an UNSUPPORTED verdict.

Point the CLI at it:  claimcheck gate --adapter examples.real_reviewer_adapter:make_adapter ...
"""

from __future__ import annotations

from claimcheck.agent import ReviewerAdapter
from claimcheck.corpus import Corpus, LexicalRetriever
from claimcheck.schema import Citation, ClaimCase, ReviewOutput, Verdict

# Reuse the bundled, captured corpus by default so the example is runnable.
from claimcheck.cli import DEFAULT_CORPUS


class LLMReviewer(ReviewerAdapter):
    name = "llm-reviewer"

    def __init__(self, corpus: Corpus, k: int = 4) -> None:
        self.corpus = corpus
        self.k = k
        self.retriever = LexicalRetriever()  # swap in your embedding retriever

    def review(self, case: ClaimCase) -> ReviewOutput:
        retrieved = self.retriever.retrieve(case.claim, self.corpus, self.k)
        retrieved_ids = [d.id for d in retrieved]

        # Fail closed before spending a token when nothing is groundable.
        if not retrieved:
            return ReviewOutput(
                verdict=Verdict.UNSUPPORTED, citations=[], retrieved_ids=[],
                abstained=True, model_was_called=False, prompt_to_model=None,
                rationale="Nothing in the source grounds this claim; flagged for a human.",
            )

        context = "\n".join(f"[{d.id}] {d.text}" for d in retrieved)
        prompt = (
            "Decide whether the SOURCES support, contradict, or fail to ground "
            "the CLAIM. Return JSON {verdict, doc_id, span, rationale}. Mark "
            "SUPPORTED only if a passage entails the claim; never invent a "
            "citation; if nothing grounds it, return UNSUPPORTED with no doc_id.\n\n"
            f"SOURCES:\n{context}\n\nCLAIM: {case.claim}\n"
        )

        # ---- call your model here (temperature 0, fixed seed) -------------- #
        # raw = your_client.complete(prompt)
        # parsed = json.loads(raw)
        raise NotImplementedError(
            "Wire your model call here, then build the ReviewOutput below from "
            "its parsed JSON. The harness scores the rest."
        )

        # Example of building the result once you have `parsed`:
        # verdict = Verdict(parsed["verdict"])
        # citations = []
        # if verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED) and parsed.get("doc_id") in retrieved_ids:
        #     citations = [Citation(doc_id=parsed["doc_id"], span=parsed.get("span", ""))]
        # return ReviewOutput(
        #     verdict=verdict, citations=citations, retrieved_ids=retrieved_ids,
        #     abstained=(verdict is Verdict.UNSUPPORTED), model_was_called=True,
        #     prompt_to_model=prompt, rationale=parsed.get("rationale", ""),
        # )


def make_adapter() -> ReviewerAdapter:
    return LLMReviewer(Corpus.load(DEFAULT_CORPUS))
