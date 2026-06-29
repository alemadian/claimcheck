"""
The content-review agent (the system under test) and the adapter contract.

The harness never talks to a reviewer directly; it talks to a ``ReviewerAdapter``.
To evaluate a real reviewer you write one subclass that calls your service and
returns a ``ReviewOutput``. Everything else in the harness stays the same.

This file also ships ``ReferenceReviewer``, a fully in-process reviewer that
makes the whole harness runnable out of the box with zero dependencies. It is a
genuine (if simple) reviewer:

  retrieve over the pinned corpus
    -> nothing clears the relevance floor?  ABSTAIN, before any model spend
    -> a passage is on-subject and the claim's facts match it?  SUPPORTED (cite)
    -> a passage is on-subject but a stated number conflicts?    CONTRADICTED (cite)
    -> on-subject passage that does not actually back the claim?  UNSUPPORTED (no cite)

The discipline is *cite-or-abstain*: it never returns SUPPORTED without a
citation to a retrieved passage, and it never invents a citation for a claim it
could not ground.

"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from .corpus import Corpus, Doc, LexicalRetriever, overlap_score
from .schema import Citation, ClaimCase, ReviewOutput, Verdict

# A claim/passage are treated as "about the same fact" once their content-word
# overlap clears this. Below it (but above the retriever's floor) a passage is a
# near-miss: on topic, but not grounds to vouch.
SAME_SUBJECT = 0.5


# --------------------------------------------------------------------------- #
# Quantity extraction (percentages, money, bare numbers)                       #
#                                                                              #
# This is what lets the reviewer catch the failure a marketing org fears most: #
# a confidently-worded but WRONG number (a price, a fee, a compliance level).  #
# Percentages and money are parsed off the raw text, not the word tokens, so   #
# "2.9%" and "$0.30" survive as single quantities; "30 cents" normalizes to    #
# 0.30 so a paraphrase still matches the source.                               #
# --------------------------------------------------------------------------- #
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_DOLLAR = re.compile(r"(CA\$|US\$|USD|CAD|\$)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_CENTS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cents?|¢)", re.IGNORECASE)
_NUM = re.compile(r"\d+(?:\.\d+)?")

# Map a money prefix to an explicit currency, or None when it is ambiguous: a
# bare "$" or "30 cents" could be either, so we do not assume one.
_CURRENCY = {"ca$": "CAD", "cad": "CAD", "us$": "USD", "usd": "USD", "$": None}


def extract_quantities(text: str) -> dict[str, set]:
    """Return {"pct": {...}, "money": {...}, "num": {...}} found in ``text``.

    Percentages and bare numbers are sets of floats. Money is a set of
    ``(currency, amount)`` pairs: the amount is normalized to dollars (cents fold
    in as fractional dollars) and the currency is "USD", "CAD", or None when it
    is not stated (a bare "$" or "30 cents"). Carrying the currency lets the
    reviewer catch a US-dollar claim cited to a Canadian-dollar source, while an
    unstated currency still matches either so an honest paraphrase is not
    punished. Percent and money spans are removed before bare numbers are read,
    so a value is never double-counted across kinds.
    """
    pcts: set[float] = set()
    moneys: set[tuple[str | None, float]] = set()
    nums: set[float] = set()

    work = text or ""
    for m in _PCT.finditer(work):
        pcts.add(round(float(m.group(1)), 4))
    work = _PCT.sub(" ", work)

    for m in _DOLLAR.finditer(work):
        currency = _CURRENCY.get(m.group(1).lower())
        moneys.add((currency, round(float(m.group(2)), 4)))
    work = _DOLLAR.sub(" ", work)

    for m in _CENTS.finditer(work):
        moneys.add((None, round(float(m.group(1)) / 100.0, 4)))
    work = _CENTS.sub(" ", work)

    for m in _NUM.finditer(work):
        nums.add(round(float(m.group(0)), 4))

    return {"pct": pcts, "money": moneys, "num": nums}


def _has_any(q: dict[str, set[float]]) -> bool:
    return any(q[k] for k in q)


def _matched(value: float, candidates: set[float]) -> bool:
    return any(math.isclose(value, c, rel_tol=1e-9, abs_tol=0.005) for c in candidates)


def _currency_compatible(a: str | None, b: str | None) -> bool:
    """Two currencies are compatible if equal or at least one is unstated."""
    return a is None or b is None or a == b


def _money_matched(value: tuple[str | None, float], candidates: set) -> bool:
    """A money value matches only when the amount is close AND the currency is
    compatible, so US$0.30 does not match a source stating CA$0.30."""
    cur, amount = value
    return any(
        math.isclose(amount, c_amount, rel_tol=1e-9, abs_tol=0.005)
        and _currency_compatible(cur, c_cur)
        for c_cur, c_amount in candidates
    )


# --------------------------------------------------------------------------- #
# Adapter contract                                                             #
# --------------------------------------------------------------------------- #
class ReviewerAdapter(ABC):
    """Wrap a content-review agent so the harness can drive it on the corpus.

    Implementations MUST treat the pinned ``Corpus`` as the only source of
    truth (retrieval over the frozen snapshot) so runs are reproducible, and
    MUST populate the trust-accounting fields on ``ReviewOutput`` (see its
    docstring).
    """

    name: str = "abstract"

    @abstractmethod
    def review(self, case: ClaimCase) -> ReviewOutput:  # pragma: no cover - interface
        ...


# --------------------------------------------------------------------------- #
# Reference reviewer                                                           #
# --------------------------------------------------------------------------- #
class ReferenceReviewer(ReviewerAdapter):
    name = "reference-reviewer"

    def __init__(self, corpus: Corpus, k: int = 4, retriever: LexicalRetriever | None = None) -> None:
        self.corpus = corpus
        self.k = k
        self.retriever = retriever or LexicalRetriever()

    # ----- public entry points -------------------------------------------- #
    def review(self, case: ClaimCase) -> ReviewOutput:
        """Review one golden case (the harness path)."""
        return self._review(case.claim)

    def review_claim(self, claim: str) -> ReviewOutput:
        """Review a single free-text claim (the live `review` command path)."""
        return self._review(claim)

    # ----- the actual reviewer -------------------------------------------- #
    def _review(self, claim: str) -> ReviewOutput:
        retrieved = self.retriever.retrieve(claim, self.corpus, self.k)
        retrieved_ids = [d.id for d in retrieved]

        # ---- fail closed before any spend when nothing is groundable ------ #
        if not retrieved:
            return ReviewOutput(
                verdict=Verdict.UNSUPPORTED,
                citations=[],
                retrieved_ids=[],
                abstained=True,
                model_was_called=False,   # the load-bearing assertion
                prompt_to_model=None,
                rationale="No passage in the source cleared the relevance floor; flagged for human review.",
            )

        # ---- build a grounded prompt from ONLY the claim + retrieved text -- #
        context_block = "\n".join(f"[{d.id}] {d.text}" for d in retrieved)
        prompt = (
            "Decide whether the SOURCES support, contradict, or fail to ground "
            "the CLAIM. Cite only the passages you used.\n\n"
            f"SOURCES:\n{context_block}\n\nCLAIM: {claim}\n"
        )

        # ---- decide the verdict from the on-subject passages -------------- #
        subject_docs = [d for d in retrieved if overlap_score(claim, d.text) >= SAME_SUBJECT]
        verdict, cited, rationale = self._decide(claim, subject_docs, retrieved)

        return ReviewOutput(
            verdict=verdict,
            citations=cited,
            retrieved_ids=retrieved_ids,
            abstained=verdict is Verdict.UNSUPPORTED,
            model_was_called=True,
            prompt_to_model=prompt,
            rationale=rationale,
        )

    def _decide(
        self, claim: str, subject_docs: list[Doc], retrieved: list[Doc]
    ) -> tuple[Verdict, list[Citation], str]:
        if not subject_docs:
            return (
                Verdict.UNSUPPORTED,
                [],
                "Retrieved passages are on the same topic but none actually states this claim.",
            )

        claim_q = extract_quantities(claim)
        # union of quantities across the on-subject passages
        doc_q: dict[str, set[float]] = {"pct": set(), "money": set(), "num": set()}
        for d in subject_docs:
            dq = extract_quantities(d.text)
            for kind in doc_q:
                doc_q[kind] |= dq[kind]

        if _has_any(claim_q):
            unmatched: list = []
            conflict = None
            for kind, values in claim_q.items():
                for value in values:
                    hit = (
                        _money_matched(value, doc_q[kind])
                        if kind == "money"
                        else _matched(value, doc_q[kind])
                    )
                    if hit:
                        continue
                    unmatched.append((kind, value))
                    # a conflict requires the source to state a DIFFERENT value
                    # of the same kind (e.g. claim 1.9%, source 2.9%; or a US$
                    # amount where the source states the same number in CA$).
                    if doc_q[kind]:
                        conflict = (kind, value)
            if not unmatched:
                best = self._best(claim, subject_docs)
                return (
                    Verdict.SUPPORTED,
                    [Citation(doc_id=best.id, span=best.text)],
                    "Every figure in the claim matches the cited source.",
                )
            if conflict is not None:
                culprit = self._best_with_kind(claim, subject_docs, conflict[0])
                return (
                    Verdict.CONTRADICTED,
                    [Citation(doc_id=culprit.id, span=culprit.text)],
                    f"The source states a different {conflict[0]} value than the claim.",
                )
            return (
                Verdict.UNSUPPORTED,
                [],
                "The claim asserts a figure the source does not state.",
            )

        # qualitative claim, strongly on-subject: treat as supported
        best = self._best(claim, subject_docs)
        return (
            Verdict.SUPPORTED,
            [Citation(doc_id=best.id, span=best.text)],
            "The cited source states this claim.",
        )

    @staticmethod
    def _best(claim: str, docs: list[Doc]) -> Doc:
        return max(docs, key=lambda d: (overlap_score(claim, d.text), d.id))

    @staticmethod
    def _best_with_kind(claim: str, docs: list[Doc], kind: str) -> Doc:
        """The on-subject passage that actually states a value of ``kind`` (so a
        contradiction is cited to the passage carrying the conflicting figure)."""
        with_kind = [d for d in docs if extract_quantities(d.text)[kind]]
        pool = with_kind or docs
        return max(pool, key=lambda d: (overlap_score(claim, d.text), d.id))


# --------------------------------------------------------------------------- #
# Splitting a piece of copy into claims, one per sentence (the live review path) #
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*[;]\s+")
_ALPHA = re.compile(r"[A-Za-z]+")


def split_into_claims(copy: str) -> list[str]:
    """Split a piece of marketing copy into claim sentences, one per sentence.

    Deliberately simple and inspectable: split on sentence enders and
    semicolons, keep fragments that carry at least a few words. This is
    sentence-level, not true atomic-claim decomposition, so a compound sentence
    is reviewed as one unit. A production splitter uses a model; the point here
    is that review happens one sentence-level claim at a time, which is what
    makes per-claim citation honest.
    """
    claims = []
    for raw in _SENT_SPLIT.split(copy or ""):
        sent = raw.strip()
        if len(_ALPHA.findall(sent)) >= 3:
            claims.append(sent)
    return claims
