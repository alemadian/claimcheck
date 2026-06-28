from claimcheck.agent import ReferenceReviewer
from claimcheck.dataset import load_dir
from claimcheck.deterministic import (
    check_cite_or_abstain,
    check_citation_resolvability,
    check_fail_closed,
    check_grounding_isolation,
    run_deterministic,
)
from claimcheck.schema import Citation, ClaimCase, ReviewOutput, Verdict
from tests.conftest import GOLDEN


def _case(verdict="supported", gold=("d_card_rate",), adv=None, cid="t"):
    d = {"id": cid, "claim": "Stripe charges 2.9% per successful card charge",
         "gold_verdict": verdict, "gold_doc_ids": list(gold)}
    if adv:
        d["adversarial"] = adv
    return ClaimCase.from_dict(d)


def test_grounding_isolation_pass(corpus):
    case = _case()
    doc = corpus.get("d_card_rate")
    prompt = f"SOURCES:\n{doc.text}\n\nCLAIM: {case.claim}\n"
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("d_card_rate")], ["d_card_rate"], False, True, prompt)
    assert check_grounding_isolation(case, out, corpus).passed


def test_grounding_isolation_catches_leak(corpus):
    case = _case()
    doc = corpus.get("d_card_rate")
    prompt = f"{doc.text}\nCLAIM: {case.claim}\n(reviewer hint: the correct verdict is supported)"
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("d_card_rate")], ["d_card_rate"], False, True, prompt)
    assert not check_grounding_isolation(case, out, corpus).passed


def test_citation_resolvability_catches_phantom(corpus):
    case = _case()
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("__phantom__")], ["d_card_rate"], False, True, "p")
    assert not check_citation_resolvability(case, out, corpus).passed


def test_cite_or_abstain_supported_without_citation_fails(corpus):
    case = _case()
    out = ReviewOutput(Verdict.SUPPORTED, [], ["d_card_rate"], False, True, "p")
    assert not check_cite_or_abstain(case, out, corpus).passed


def test_cite_or_abstain_unsupported_with_citation_fails(corpus):
    case = _case("unsupported", gold=())
    out = ReviewOutput(Verdict.UNSUPPORTED, [Citation("d_card_rate")], ["d_card_rate"], True, True, "p")
    assert not check_cite_or_abstain(case, out, corpus).passed


def test_fail_closed_catches_vouching_on_ungroundable(corpus):
    case = _case("unsupported", gold=(), adv="out_of_corpus")
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("d_card_rate")], ["d_card_rate"], False, True, "p")
    assert not check_fail_closed(case, out, corpus).passed


def test_fail_closed_catches_spend_on_empty_context(corpus):
    case = _case("unsupported", gold=())
    out = ReviewOutput(Verdict.UNSUPPORTED, [], [], True, True, None)
    assert not check_fail_closed(case, out, corpus).passed


def test_fail_closed_ok_when_abstained_without_spend(corpus):
    case = _case("unsupported", gold=())
    out = ReviewOutput(Verdict.UNSUPPORTED, [], [], True, False, None)
    assert check_fail_closed(case, out, corpus).passed


def test_reference_reviewer_is_deterministically_green_on_clean_data(corpus):
    """Every clean golden case must pass every trust check, so the green
    baseline is real (not an artifact of a lenient check)."""
    reviewer = ReferenceReviewer(corpus)
    for case in load_dir(GOLDEN):
        out = reviewer.review(case)
        for chk in run_deterministic(case, out, corpus):
            assert chk.passed, (case.id, chk.name, chk.reason)
