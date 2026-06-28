from claimcheck.agent import ReferenceReviewer
from claimcheck.corpus import Corpus
from claimcheck.dataset import load_dir
from claimcheck.drift import detect_drift
from claimcheck.gating import evaluate_gate
from claimcheck.judges.client import FakeJudgeClient
from claimcheck.metrics import aggregate
from claimcheck.runner import run_suite
from tests.conftest import BUGGY, CORPUS_PATH, GOLDEN


def _scoreboard(data):
    corpus = Corpus.load(CORPUS_PATH)
    cases = load_dir(data)
    results = run_suite(ReferenceReviewer(corpus), cases, corpus, judge=FakeJudgeClient())
    return aggregate(results)


def test_clean_run_has_zero_published_falsehoods():
    sb = _scoreboard(GOLDEN)
    assert sb.headline["published_falsehood_rate"] == 0.0
    assert sb.headline["verdict_accuracy"] == 1.0
    assert sb.headline["deterministic_pass_rate"] == 1.0


def test_gate_passes_against_itself():
    sb = _scoreboard(GOLDEN)
    assert evaluate_gate(sb, sb.to_dict()).passed


def test_gate_blocks_on_buggy_set():
    baseline = _scoreboard(GOLDEN).to_dict()
    gate = evaluate_gate(_scoreboard(BUGGY), baseline)
    assert not gate.passed
    assert any("published_falsehood_rate" in f.message for f in gate.findings)
    assert any("Tier 1" in f.message and f.severity == "block" for f in gate.findings)


def test_drift_flags_rising_published_falsehood():
    history = [{"published_falsehood_rate": 0.0} for _ in range(5)]
    alerts = detect_drift(history, {"published_falsehood_rate": 0.2}, window=5)
    assert any(a.metric == "published_falsehood_rate" for a in alerts)


def test_drift_ignores_an_improving_metric():
    history = [{"verdict_accuracy": 0.80} for _ in range(5)]
    alerts = detect_drift(history, {"verdict_accuracy": 0.95}, window=5)
    assert alerts == []
