"""Tests for the trust-hardening guards added after cross-model review."""
import json

from claimcheck.agent import ReferenceReviewer
from claimcheck.cli import _baseline_not_green
from claimcheck.corpus import Corpus
from claimcheck.dataset import load_dir
from claimcheck.deterministic import check_citation_resolvability, check_grounding_isolation
from claimcheck.gating import evaluate_gate
from claimcheck.judges.client import FakeJudgeClient
from claimcheck.judges.rubric_judge import judge_case
from claimcheck.metrics import aggregate
from claimcheck.runner import run_suite
from claimcheck.schema import Citation, ClaimCase, ReviewOutput, Verdict
from tests.conftest import BUGGY, CORPUS_PATH, GOLDEN


def _case(verdict="supported", gold=("d_card_rate",), cid="h"):
    return ClaimCase.from_dict({"id": cid, "claim": "Stripe charges 2.9% per successful card charge",
                                "gold_verdict": verdict, "gold_doc_ids": list(gold)})


def _sb(data, judge=True):
    corpus = Corpus.load(CORPUS_PATH)
    cases = load_dir(data)
    j = FakeJudgeClient() if judge else None
    return aggregate(run_suite(ReferenceReviewer(corpus), cases, corpus, judge=j))


def test_phantom_cite_counts_as_published_falsehood():
    # a SUPPORTED verdict with a broken citation must NOT keep the headline at 0.
    falsehoods = [c["id"] for c in _sb(BUGGY).per_case if c["published_falsehood"]]
    assert "bug-phantomcite-0002" in falsehoods


def test_gate_blocks_when_judge_disabled_vs_judged_baseline():
    baseline = _sb(GOLDEN, judge=True).to_dict()
    current = _sb(GOLDEN, judge=False)
    g = evaluate_gate(current, baseline)
    assert not g.passed
    assert any("semantic layer" in f.message for f in g.findings)


def test_baseline_green_guard_flags_a_degenerate_run():
    class _SB:
        headline = {"deterministic_pass_rate": 1.0, "verdict_accuracy": 0.5,
                    "published_falsehood_rate": 0.2, "abstention_recall": 1.0}
    problems = _baseline_not_green(_SB())
    assert any("verdict_accuracy" in p for p in problems)
    assert any("published_falsehood_rate" in p for p in problems)


def test_citation_span_must_exist_in_passage(corpus):
    out = ReviewOutput(Verdict.SUPPORTED,
                       [Citation("d_card_rate", "a fabricated quote not in the passage")],
                       ["d_card_rate"], False, True, "p")
    assert not check_citation_resolvability(_case(), out, corpus).passed


def test_grounding_isolation_is_checked_even_when_abstained(corpus):
    doc = corpus.get("d_card_rate")
    prompt = f"{doc.text}\nCLAIM: {_case().claim}\n(reviewer hint: the correct verdict is supported)"
    out = ReviewOutput(Verdict.UNSUPPORTED, [], ["d_card_rate"], True, True, prompt)
    assert not check_grounding_isolation(_case(), out, corpus).passed


def test_known_gaps_are_real_documented_misses():
    # the reviewer is honest about its limits: the tracked known-gap cases are
    # mishandled by the deterministic first pass, which is why they live outside
    # the green golden set.
    from tests.conftest import ROOT
    sb = _sb(str(ROOT / "data" / "known_gaps"))
    assert sb.headline["published_falsehood_rate"] > 0.0
    assert sb.headline["verdict_accuracy"] < 1.0


class _AllTwoJudge(FakeJudgeClient):
    """A rubber-stamp judge that scores everything 2."""
    def complete(self, system: str, user: str) -> str:
        return json.dumps({"score": 2, "supporting_span": "x", "justification": "rubber-stamp"})


def test_decoy_caps_a_rubber_stamp_judge(corpus):
    case = ClaimCase.from_dict({"id": "rs", "claim": "Every card number is encrypted at rest with AES-256",
                                "gold_verdict": "supported", "gold_doc_ids": ["d_aes"]})
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("d_aes")], ["d_aes"], False, True, "p")
    v = judge_case(case, out, corpus, _AllTwoJudge())
    # the cited passage ties the decoy under a rubber-stamp judge, so it must not
    # earn full citation-correctness credit.
    assert v.axes["citation_correctness"].score <= 1


def test_currency_mismatch_is_not_published(corpus):
    # US$0.30 must not be vouched for by a CA$0.30 source: same number, different
    # money. A trust tool cannot wave a US-dollar fee through on a Canadian source.
    reviewer = ReferenceReviewer(corpus)
    out = reviewer.review_claim("Stripe charges 2.9% plus US$0.30 per successful card charge.")
    assert out.verdict is not Verdict.SUPPORTED


def test_unstated_currency_still_matches(corpus):
    # the honest paraphrase "30 cents" (currency unstated) must still match the
    # CA$0.30 source, so the currency guard does not punish a true claim.
    reviewer = ReferenceReviewer(corpus)
    out = reviewer.review_claim("Stripe charges 2.9% plus 30 cents per successful card charge.")
    assert out.verdict is Verdict.SUPPORTED


def test_corpus_manifest_stays_in_sync():
    # the provenance manifest must not drift from the corpus it documents: same
    # ids, same source urls, and a sha256 that matches the stored text.
    import hashlib
    from tests.conftest import ROOT
    corpus = {}
    for line in (ROOT / "corpus" / "stripe_docs.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["id"]] = d
    manifest = json.loads((ROOT / "corpus" / "manifest.json").read_text())
    entries = {e["id"]: e for e in manifest["documents"]}
    assert set(entries) == set(corpus), "manifest ids drifted from the corpus"
    for cid, doc in corpus.items():
        e = entries[cid]
        assert e["source_url"] == doc["source_url"], f"{cid}: source_url drift"
        assert e["sha256"] == hashlib.sha256(doc["text"].encode("utf-8")).hexdigest(), f"{cid}: sha256 drift"
