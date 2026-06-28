from claimcheck.agent import ReferenceReviewer, extract_quantities
from claimcheck.judges.client import FakeJudgeClient
from claimcheck.judges.rubric_judge import judge_case
from claimcheck.retrieval import score_retrieval
from claimcheck.schema import Citation, ClaimCase, ReviewOutput, Verdict


def _case(claim, verdict="supported", gold=("d_aes",), cid="c"):
    return ClaimCase.from_dict({
        "id": cid, "claim": claim, "gold_verdict": verdict, "gold_doc_ids": list(gold),
    })


def test_extract_quantities_normalizes_money_and_percent():
    q = extract_quantities("2.9% + 30 cents, capped at CA$5.00, AES-256")
    assert 2.9 in q["pct"]
    assert (None, 0.30) in q["money"]    # "30 cents" -> 0.30, currency unstated
    assert ("CAD", 5.0) in q["money"]    # "CA$5.00" -> 5.00 Canadian dollars
    assert 256 in q["num"]


def test_retrieval_recall_hits_gold(corpus):
    reviewer = ReferenceReviewer(corpus)
    case = _case("Stripe charges 2.9% + 30 cents per successful card charge",
                 gold=("d_card_rate",))
    out = reviewer.review(case)
    sc = score_retrieval(case, out, 4)
    assert sc.recall_at_k == 1.0
    assert sc.mrr > 0.0


def test_judge_rewards_the_right_citation(corpus):
    case = _case("Every card number is encrypted at rest with AES-256")
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("d_aes")], ["d_aes"], False, True, "p")
    v = judge_case(case, out, corpus, FakeJudgeClient())
    assert v.axes["citation_correctness"].score == 2


def test_judge_punishes_a_wrong_citation_via_decoy(corpus):
    # cite an unrelated passage; the decoy (the real AES passage) beats it -> 0
    case = _case("Every card number is encrypted at rest with AES-256")
    out = ReviewOutput(Verdict.SUPPORTED, [Citation("d_dispute")], ["d_dispute"], False, True, "p")
    v = judge_case(case, out, corpus, FakeJudgeClient())
    assert v.axes["citation_correctness"].score == 0


def test_judge_skips_abstention(corpus):
    case = _case("Stripe guarantees 99.999% uptime", verdict="unsupported", gold=())
    out = ReviewOutput(Verdict.UNSUPPORTED, [], [], True, False, None)
    v = judge_case(case, out, corpus, FakeJudgeClient())
    assert v.skipped
