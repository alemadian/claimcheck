import pytest

from claimcheck.corpus import Corpus
from claimcheck.dataset import load_dir, validate_against_corpus
from claimcheck.schema import ClaimCase, ReviewOutput, Verdict
from tests.conftest import CORPUS_PATH, GOLDEN


def _case(**kw):
    base = {
        "id": "x", "claim": "a claim with words",
        "gold_verdict": "supported", "gold_doc_ids": ["d_card_rate"],
    }
    base.update(kw)
    return ClaimCase.from_dict(base)


def test_unsupported_cannot_carry_gold_docs():
    with pytest.raises(ValueError):
        _case(gold_verdict="unsupported", gold_doc_ids=["d_card_rate"])


def test_supported_must_carry_gold_docs():
    with pytest.raises(ValueError):
        _case(gold_verdict="supported", gold_doc_ids=[])


def test_contradicted_must_carry_gold_docs():
    with pytest.raises(ValueError):
        _case(gold_verdict="contradicted", gold_doc_ids=[])


def test_unknown_verdict_is_a_load_error():
    with pytest.raises(ValueError):
        _case(gold_verdict="probably")


def test_empty_claim_rejected():
    with pytest.raises(ValueError):
        _case(claim="   ")


def test_properties():
    c = _case(gold_verdict="unsupported", gold_doc_ids=[])
    assert c.must_abstain and not c.is_groundable and not c.safe_to_publish
    c2 = _case(gold_verdict="supported")
    assert c2.safe_to_publish and c2.is_groundable and not c2.must_abstain


def test_review_output_roundtrip():
    out = ReviewOutput.from_dict({
        "verdict": "contradicted",
        "citations": [{"doc_id": "d_card_rate", "span": "2.9%"}],
        "retrieved_ids": ["d_card_rate"],
        "abstained": False, "model_was_called": True,
    })
    assert out.verdict is Verdict.CONTRADICTED
    assert out.citations[0].doc_id == "d_card_rate"


def test_golden_dataset_loads_and_validates():
    corpus = Corpus.load(CORPUS_PATH)
    cases = load_dir(GOLDEN)
    assert len(cases) >= 15
    validate_against_corpus(cases, corpus)  # must not raise


def test_dangling_gold_doc_ref_is_caught():
    corpus = Corpus.load(CORPUS_PATH)
    bad = _case(gold_doc_ids=["d_does_not_exist"])
    with pytest.raises(ValueError):
        validate_against_corpus([bad], corpus)
