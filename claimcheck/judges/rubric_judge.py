"""
Layer 3: model-as-judge with explicit, version-controlled rubrics.

Two axes, scored 0/1/2 on an anchored scale, judged only on grounded verdicts
(SUPPORTED / CONTRADICTED). The verdict's *correctness* against gold is graded
deterministically in metrics.py; the judge's job is the harder, semantic
question the deterministic layer cannot answer: does the cited passage actually
justify the verdict, or is it a plausible-but-wrong pointer?

Axes
----
- FAITHFULNESS         : (SUPPORTED only) is the claim entailed by the cited
                         passage, with no facts added beyond it.
- CITATION_CORRECTNESS : does the cited passage actually pertain to THIS claim,
                         beating the hardest decoy passage? A snippet that is
                         on-topic but does not bear on the specific claim scores
                         low. Evaluated per citation; the weakest citation sets
                         the axis, so one bad pointer cannot hide behind good
                         ones.

Position-swap: citation correctness scores the cited passage AND the most
lexically-similar non-cited passage (the hardest decoy), and requires the cited
passage to BEAT the decoy. A tie does not earn full credit, which is what catches
a judge that rubber-stamps every pairing with a high score.

The judge model id, temperature, seed and prompt version are pinned in
JudgeConfig and recorded in the verdict, because the judge is a dependency that
drifts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..corpus import Corpus, content_tokens
from ..schema import ClaimCase, ReviewOutput, Verdict
from .client import JudgeClient

_SCALE = (
    "SCORE SCALE (choose exactly one):\n"
    "  2 = fully supports / fully on-point, no added or unrelated facts.\n"
    "  1 = partially on-point, some unsupported detail.\n"
    "  0 = does not support / off-point / wrong pointer.\n"
)

FAITHFULNESS_RUBRIC = (
    "You are grading FAITHFULNESS. Is the CLAIM entailed by the CITED PASSAGE, "
    "with no facts added beyond it?\n"
    "First QUOTE the exact span of the passage that supports (or fails to "
    "support) the claim. Then score.\n" + _SCALE
)

CITATION_CORRECTNESS_RUBRIC = (
    "You are grading CITATION CORRECTNESS. Does the CITED PASSAGE actually bear "
    "on the SPECIFIC CLAIM it is attached to? A passage that is on-topic but "
    "does not bear on THIS claim scores low.\n"
    "First QUOTE the span that bears on the claim (or state none exists). Then "
    "score.\n" + _SCALE
)

_SYSTEM = (
    "You are a strict, literal grader for a marketing claims-review system. "
    "You never reward fluent text that the cited source does not bear out. "
    "You always quote the supporting span before you score. "
    "Respond ONLY with a JSON object: "
    '{"score": <0|1|2>, "supporting_span": "<quoted text>", '
    '"justification": "<one line>"}.'
)


@dataclass(frozen=True)
class AxisScore:
    axis: str
    score: int          # 0..2
    supporting_span: str
    justification: str

    @property
    def normalized(self) -> float:
        return self.score / 2.0


@dataclass
class JudgeVerdict:
    case_id: str
    axes: dict[str, AxisScore]
    judge_model_id: str
    prompt_fingerprint: str
    skipped: bool = False
    skip_reason: str = ""

    def axis_norm(self, axis: str) -> float | None:
        a = self.axes.get(axis)
        return a.normalized if a else None


def _jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_user(axis: str, rubric: str, *, claim: str, passage_text: str) -> str:
    from .client import FakeJudgeClient
    human_visible = (
        f"{rubric}\n\nCLAIM: {claim}\nCITED PASSAGE: {passage_text}\n"
    )
    payload = json.dumps({"axis": axis, "claim": claim, "passage_text": passage_text})
    return human_visible + "\n" + FakeJudgeClient.SENTINEL + payload


def _ask(client: JudgeClient, axis: str, rubric: str, *, claim: str, passage_text: str) -> AxisScore:
    user = _build_user(axis, rubric, claim=claim, passage_text=passage_text)
    raw = client.complete(_SYSTEM, user)
    try:
        obj = json.loads(raw)
        score = int(obj["score"])
        if score not in (0, 1, 2):
            raise ValueError(f"score out of range: {score}")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"judge returned malformed verdict for {axis}: {raw!r}") from exc
    return AxisScore(
        axis=axis,
        score=score,
        supporting_span=str(obj.get("supporting_span", "")),
        justification=str(obj.get("justification", "")),
    )


def _pick_decoy(claim: str, cited_id: str, corpus: Corpus) -> str | None:
    """Hardest decoy: the most lexically-similar passage that is NOT the cited
    one, so the position-swap tests resistance to a plausible distractor."""
    others = [d for d in corpus.docs if d.id != cited_id]
    if not others:
        return None
    best = max(others, key=lambda d: (_jaccard(claim, d.text), d.id))
    return best.id


def judge_case(case: ClaimCase, out: ReviewOutput, corpus: Corpus, client: JudgeClient) -> JudgeVerdict:
    fingerprint = client.config.fingerprint(FAITHFULNESS_RUBRIC + CITATION_CORRECTNESS_RUBRIC)
    base = dict(case_id=case.id, judge_model_id=client.config.model_id, prompt_fingerprint=fingerprint)

    # Only grounded verdicts are judged. An abstention has no citation to grade;
    # its correctness is measured by the deterministic fail-closed check and the
    # metrics layer.
    if out.verdict is Verdict.UNSUPPORTED or out.abstained or not out.citations:
        return JudgeVerdict(axes={}, skipped=True,
                            skip_reason="abstained / unsupported / no citation to grade", **base)

    axes: dict[str, AxisScore] = {}

    # ---- citation correctness (both supported and contradicted) ----------- #
    pair_scores: list[int] = []
    pair_spans: list[str] = []
    for c in out.citations:
        doc = corpus.get(c.doc_id)
        if doc is None:
            pair_scores.append(0)
            pair_spans.append(f"{c.doc_id}: unresolved")
            continue
        cited_passage = c.span if c.span else doc.text
        real = _ask(client, "citation_correctness", CITATION_CORRECTNESS_RUBRIC,
                    claim=case.claim, passage_text=cited_passage)
        decoy_id = _pick_decoy(case.claim, c.doc_id, corpus)
        decoy_score = 0
        if decoy_id is not None:
            decoy = _ask(client, "citation_correctness", CITATION_CORRECTNESS_RUBRIC,
                         claim=case.claim, passage_text=corpus.get(decoy_id).text)
            decoy_score = decoy.score
        # The cited passage must BEAT the hardest decoy. A tie is not distinctive
        # enough for full credit (this is what catches a rubber-stamp judge that
        # scores everything high); a decoy that wins means a wrong pointer.
        if real.score > decoy_score:
            effective = real.score
        elif real.score == decoy_score:
            effective = min(real.score, 1)
        else:
            effective = 0
        pair_scores.append(effective)
        pair_spans.append(f"{c.doc_id}:{real.supporting_span[:50]}")
    axes["citation_correctness"] = AxisScore(
        "citation_correctness",
        min(pair_scores) if pair_scores else 0,
        " | ".join(pair_spans)[:240],
        f"weakest of {len(pair_scores)} citation pair(s)",
    )

    # ---- faithfulness (supported verdicts only) --------------------------- #
    if out.verdict is Verdict.SUPPORTED:
        cited_text = "\n".join(
            corpus.get(c.doc_id).text for c in out.citations if corpus.get(c.doc_id)
        )
        axes["faithfulness"] = _ask(client, "faithfulness", FAITHFULNESS_RUBRIC,
                                    claim=case.claim, passage_text=cited_text)

    return JudgeVerdict(axes=axes, **base)
