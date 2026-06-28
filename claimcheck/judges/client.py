"""
Judge client abstraction.

The judge model is itself a pinned, version-controlled dependency that can
drift, so we (a) record the exact model id, prompt version, temperature and
seed for every run, and (b) hide the transport behind an interface so the
harness logic never depends on a specific provider.

``JudgeClient.complete(system, user)`` returns a JSON string. The judge prompt
asks for a strict JSON object; the parser in rubric_judge.py validates it.

``FakeJudgeClient`` is a deterministic, offline judge used for tests and the
out-of-the-box demo. It applies the rubric *mechanically* (content-word overlap
between the claim and the cited passage) so the demo exercises the full judge
pipeline without a network call. Because it is mechanical it does NOT pretend to
distinguish entailment from contradiction by meaning; that is exactly why a
production judge subclasses ``JudgeClient`` and calls a real model at
temperature 0. See examples/real_judge_adapter.py.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..corpus import content_tokens


@dataclass(frozen=True)
class JudgeConfig:
    """Everything that must be pinned for judge reproducibility."""
    model_id: str = "fake-judge-v1"
    temperature: float = 0.0
    seed: int = 7
    prompt_version: str = "v1"

    def fingerprint(self, prompt_template: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_id.encode())
        h.update(str(self.temperature).encode())
        h.update(str(self.seed).encode())
        h.update(self.prompt_version.encode())
        h.update(prompt_template.encode())
        return h.hexdigest()[:16]


class JudgeClient(ABC):
    config: JudgeConfig

    @abstractmethod
    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        """Return the judge's raw response (expected to be a JSON object)."""
        ...


def _jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class FakeJudgeClient(JudgeClient):
    """Deterministic offline judge.

    It reads a structured payload (embedded after a sentinel in the user
    message) and scores the requested axis mechanically. This keeps the demo
    offline while still flowing through the exact prompt-build / parse /
    aggregate path a real judge uses.
    """

    SENTINEL = "###PAYLOAD###"

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()

    def complete(self, system: str, user: str) -> str:
        payload = self._extract_payload(user)
        axis = payload["axis"]
        claim = payload.get("claim") or ""
        passage = payload.get("passage_text") or ""
        ov = _jaccard(claim, passage)

        if axis == "faithfulness":
            score = 2 if ov >= 0.5 else (1 if ov >= 0.25 else 0)
        elif axis == "citation_correctness":
            score = 2 if ov >= 0.4 else (1 if ov >= 0.2 else 0)
        else:  # pragma: no cover - guarded upstream
            raise ValueError(f"unknown axis {axis!r}")

        return json.dumps({
            "score": score,
            "supporting_span": passage[:120],
            "justification": f"claim/passage content overlap {ov:.2f}",
        })

    def _extract_payload(self, user: str) -> dict:
        # rfind: the harness appends the real payload last, so untrusted claim or
        # passage text that happens to contain the sentinel cannot hijack it.
        idx = user.rfind(self.SENTINEL)
        if idx < 0:
            raise ValueError("FakeJudgeClient expected an embedded payload")
        return json.loads(user[idx + len(self.SENTINEL):])
