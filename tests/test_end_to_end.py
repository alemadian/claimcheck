"""End-to-end smoke tests that mirror the demo flow in the README."""
from claimcheck.cli import main
from tests.conftest import BUGGY, CALIB, GOLDEN


def test_cli_run_clean(capsys):
    assert main(["run", "--data", GOLDEN]) == 0
    out = capsys.readouterr().out
    assert "published_falsehood_rate" in out
    assert "verdict_accuracy" in out


def test_cli_gate_blocks_on_buggy(tmp_path, capsys):
    base = tmp_path / "baseline.json"
    assert main(["baseline", "--data", GOLDEN, "--out", str(base)]) == 0
    assert main(["gate", "--data", BUGGY, "--baseline", str(base)]) == 1
    assert "BLOCK" in capsys.readouterr().out


def test_cli_calibrate(capsys):
    code = main(["calibrate", "--data", GOLDEN, "--labels", CALIB, "--min-exact", "0.7"])
    assert code in (0, 2)  # trusted or flags retune; both are valid exits
    assert "calibration" in capsys.readouterr().out


def test_cli_review_holds_bad_copy(capsys):
    code = main(["review", "--text",
                 "The rate for a successful card charge is just 1.9%. "
                 "Stripe guarantees your funds are insured up to one million dollars."])
    assert code == 1
    assert "HOLD" in capsys.readouterr().out


def test_cli_review_publishes_supported_copy(capsys):
    code = main(["review", "--text",
                 "Stripe charges 2.9% plus 30 cents per successful card charge. "
                 "Every card number is encrypted at rest with AES-256."])
    assert code == 0
    assert "PUBLISH" in capsys.readouterr().out
