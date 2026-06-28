"""
claimcheck - a marketing claims-review agent with a cite-or-abstain trust
layer, and the evaluation harness that scores it.

The system under test (SUT) is a *content-review agent*. Given a piece of
marketing copy, it:
  - splits the copy into claim sentences (one per sentence; true atomic-claim
    decomposition is a documented next step, not what the bundled splitter does),
  - checks each claim against a pinned corpus of real, public source-of-truth
    documents (here: public Stripe pricing and security docs),
  - returns one of three verdicts per claim with a citation to the exact
    passage it relied on:
        SUPPORTED     - the source backs the claim (cite the passage),
        CONTRADICTED  - the source conflicts with the claim (cite the passage),
        UNSUPPORTED   - nothing in the source grounds the claim, so the agent
                        abstains and flags it for a human instead of vouching.

The discipline that makes it trustworthy is *cite-or-abstain*: the agent never
marks a claim SUPPORTED without a resolvable citation, and a piece of copy is
only cleared to publish when every claim is SUPPORTED.

The harness scores that system on layers, cheapest and most deterministic
first, so a build fails fast before any model-judge spend:

  1. Deterministic trust checks (no model, sub-second, every commit)
     - grounding isolation, citation resolvability, cite-or-abstain integrity,
       fail-closed on ungroundable claims.
  2. Retrieval evaluation (recall@k / MRR / nDCG against the gold passages).
  3. Model-as-judge (faithfulness + citation correctness, versioned rubric,
     run in CI and on a nightly schedule).

Plus the metric that actually matters to a marketing org: the
*published-falsehood rate* (how often the agent vouches for a claim the source
does not support), a layered CI regression gate that never lets that rate rise,
drift detection over a frozen split, and judge-vs-human calibration.

Nothing here imports a network client at import time. The demo runs on the
Python standard library alone.
"""

from .schema import (
    Citation,
    ClaimCase,
    ReviewOutput,
    Verdict,
    Adversarial,
)
from .corpus import Doc, Corpus
from .agent import ReviewerAdapter

__all__ = [
    "Citation",
    "ClaimCase",
    "ReviewOutput",
    "Verdict",
    "Adversarial",
    "Doc",
    "Corpus",
    "ReviewerAdapter",
]

__version__ = "0.1.0"
