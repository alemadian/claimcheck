"""
Schema for the claims-review harness.

Everything is a plain dataclass with a strict ``from_dict`` so the JSONL golden
files are validated on load (a malformed case fails the build, it is never
silently skipped). Keeping the schema in one file means the golden dataset, the
reviewer adapter, and the judges all agree on field names.

Golden case layout (one JSON object per line in a .jsonl file)
--------------------------------------------------------------
{
  "id": "pricing-contradicted-0001",   # stable, unique, used in the gate
  "split": "frozen" | "living",         # frozen never changes (drift baseline)
  "claim": "Stripe's standard rate is just 1.9% + 10c per transaction.",
  "gold_verdict": "supported" | "contradicted" | "unsupported",
  "gold_doc_ids": ["d_card_rate"],      # passage(s) that support/contradict it;
                                        # MUST be empty for an unsupported claim
  "claim_type": "pricing",              # free-text slice label
  "copy_context": "...",                # optional: the larger copy it came from
  "adversarial": null | "out_of_corpus" | "near_miss" | "stale" | "trap",
  "notes": "..."                        # free text, e.g. the prod miss it came from
}

The ``gold_doc_ids`` reference the SHARED corpus snapshot (see corpus.py), not a
per-case fixture. Membership is validated against the loaded corpus in
dataset.py, so a dangling reference blocks the build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Closed vocabularies (an unknown value is a load error, by design)            #
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    SUPPORTED = "supported"          # the source backs the claim
    CONTRADICTED = "contradicted"    # the source conflicts with the claim
    UNSUPPORTED = "unsupported"      # nothing grounds it -> abstain / flag


class Adversarial(str, Enum):
    OUT_OF_CORPUS = "out_of_corpus"  # the topic is simply not in the source
    NEAR_MISS = "near_miss"          # an on-topic passage that does NOT back it
    STALE = "stale"                  # a time-sensitive / future claim
    TRAP = "trap"                    # looks supported but isn't


def _require(d: dict, key: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required field {key!r}")
    return d[key]


# --------------------------------------------------------------------------- #
# Golden case                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class ClaimCase:
    id: str
    claim: str
    gold_verdict: Verdict
    gold_doc_ids: list[str]
    claim_type: str = "general"
    copy_context: str = ""
    split: str = "frozen"
    adversarial: Optional[Adversarial] = None
    notes: str = ""

    @property
    def must_abstain(self) -> bool:
        """True when the only correct behavior is to NOT vouch (flag for human).

        An unsupported claim has nothing in the source to ground it, so a
        trustworthy reviewer must abstain rather than assert support.
        """
        return self.gold_verdict is Verdict.UNSUPPORTED

    @property
    def is_groundable(self) -> bool:
        """True when a real passage supports or contradicts the claim."""
        return self.gold_verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED)

    @property
    def safe_to_publish(self) -> bool:
        """The publish gate: only a SUPPORTED claim should clear copy to ship."""
        return self.gold_verdict is Verdict.SUPPORTED

    @staticmethod
    def from_dict(d: dict) -> "ClaimCase":
        verdict = Verdict(_require(d, "gold_verdict"))
        gold_ids = [str(x) for x in d.get("gold_doc_ids", [])]

        # An unsupported claim cannot carry supporting/contradicting passages:
        # if a passage grounded it, the verdict would not be "unsupported".
        if verdict is Verdict.UNSUPPORTED and gold_ids:
            raise ValueError(
                f"case {d.get('id')!r}: an unsupported claim must have empty "
                f"gold_doc_ids (got {gold_ids})"
            )
        # A supported/contradicted claim MUST name the passage that grounds the
        # verdict, otherwise citation correctness cannot be graded and the case
        # would earn free credit.
        if verdict is not Verdict.UNSUPPORTED and not gold_ids:
            raise ValueError(
                f"case {d.get('id')!r}: a {verdict.value} claim must name at "
                f"least one gold_doc_id"
            )

        claim = str(_require(d, "claim")).strip()
        if not claim:
            raise ValueError(f"case {d.get('id')!r}: empty claim")

        adv = d.get("adversarial")
        return ClaimCase(
            id=str(_require(d, "id")),
            claim=claim,
            gold_verdict=verdict,
            gold_doc_ids=gold_ids,
            claim_type=str(d.get("claim_type", "general")),
            copy_context=str(d.get("copy_context", "")),
            split=str(d.get("split", "frozen")),
            adversarial=Adversarial(adv) if adv else None,
            notes=str(d.get("notes", "")),
        )


# --------------------------------------------------------------------------- #
# What the reviewer (system under test) must return - the adapter contract      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Citation:
    """A pointer from a verdict to the corpus passage it relied on.

    ``doc_id`` must resolve to a passage that was actually retrieved. ``span``
    is the quoted text the reviewer says justifies its verdict (used by the
    judge's citation-correctness axis and shown in the human report).
    """
    doc_id: str
    span: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Citation":
        return Citation(
            doc_id=str(_require(d, "doc_id")),
            span=str(d.get("span", "")),
        )


@dataclass
class ReviewOutput:
    """The structured result the reviewer returns for one claim.

    Fields that make the trust accounting checkable:
      - verdict          : supported / contradicted / unsupported.
      - abstained        : the agent declined to vouch (verdict is unsupported).
      - model_was_called : did the verdict step actually invoke a model? We
                           assert this is False when there was nothing to ground
                           against (no spend before grounding).
      - prompt_to_model  : the exact string handed to the verdict model, so the
                           grounding-isolation check can prove no label leaked.
      - retrieved_ids    : ordered passage ids the retriever surfaced.
    """
    verdict: Verdict
    citations: list[Citation]
    retrieved_ids: list[str]
    abstained: bool
    model_was_called: bool
    prompt_to_model: Optional[str] = None
    rationale: str = ""
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "ReviewOutput":
        return ReviewOutput(
            verdict=Verdict(_require(d, "verdict")),
            citations=[Citation.from_dict(c) for c in d.get("citations", [])],
            retrieved_ids=[str(x) for x in d.get("retrieved_ids", [])],
            abstained=bool(d.get("abstained", False)),
            model_was_called=bool(d.get("model_was_called", False)),
            prompt_to_model=d.get("prompt_to_model"),
            rationale=str(d.get("rationale", "")),
            raw=dict(d.get("raw", {})),
        )
