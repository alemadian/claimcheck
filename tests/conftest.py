from pathlib import Path

import pytest

from claimcheck.corpus import Corpus

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "corpus" / "stripe_docs.jsonl"
GOLDEN = str(ROOT / "data" / "golden")
BUGGY = str(ROOT / "data" / "golden_buggy")
CALIB = str(ROOT / "data" / "calibration" / "human_labels.jsonl")


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.load(CORPUS_PATH)
