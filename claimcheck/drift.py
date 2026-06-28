"""
Drift monitor + time-series log.

The frozen golden split is re-run on a schedule (nightly). Every headline
metric is appended to a JSONL time series. When a metric moves outside its band
versus the trailing window, an alert fires. The most important one to watch is
published_falsehood_rate creeping up with no code change, which means a model or
corpus change has started letting false claims through.

Direction matters: a lower-is-better metric (published_falsehood_rate,
over_flag_rate) pages when it RISES; everything else pages when it DROPS. The
alert sink is pluggable; in production you would wire ``notify`` to your
iMessage / Slack / Telegram path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

from .metrics import Scoreboard

DEFAULT_BANDS = {
    "published_falsehood_rate": 0.02,
    "verdict_accuracy": 0.03,
    "contradiction_recall": 0.03,
    "abstention_recall": 0.03,
    "over_flag_rate": 0.05,
    "citation_correctness": 0.03,
    "faithfulness_rate": 0.03,
    "retrieval_recall_at_k": 0.03,
}

# metrics where a RISE is the bad direction (everything else pages on a drop).
_LOWER_BETTER = frozenset({"published_falsehood_rate", "over_flag_rate"})


@dataclass
class DriftAlert:
    metric: str
    current: float
    trailing_mean: float
    delta: float
    band: float

    @property
    def message(self) -> str:
        direction = "up" if self.delta > 0 else "down"
        return (f"DRIFT {self.metric}: {direction} {self.delta:+.3f} "
                f"(now {self.current:.3f} vs trailing {self.trailing_mean:.3f}, "
                f"band +/-{self.band:.3f})")


def append_timeseries(scoreboard: Scoreboard, path: str | Path, ts: Optional[float] = None) -> None:
    row = {"ts": ts if ts is not None else time.time(), **scoreboard.headline}
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def load_timeseries(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def detect_drift(
    history: list[dict],
    current: dict,
    *,
    window: int = 7,
    bands: dict[str, float] | None = None,
) -> list[DriftAlert]:
    bands = {**DEFAULT_BANDS, **(bands or {})}
    trailing = history[-window:]
    alerts: list[DriftAlert] = []
    for metric, band in bands.items():
        cur = current.get(metric)
        if cur is None:
            continue
        past = [row[metric] for row in trailing if row.get(metric) is not None]
        if len(past) < 2:
            continue
        tm = mean(past)
        delta = cur - tm
        if abs(delta) <= band:
            continue
        bad_direction = (delta > 0) if metric in _LOWER_BETTER else (delta < 0)
        if bad_direction:
            alerts.append(DriftAlert(metric, cur, tm, delta, band))
    return alerts


def run_drift_check(
    scoreboard: Scoreboard,
    timeseries_path: str | Path,
    *,
    window: int = 7,
    bands: dict[str, float] | None = None,
    notify: Optional[Callable[[str], None]] = None,
    record: bool = True,
) -> list[DriftAlert]:
    history = load_timeseries(timeseries_path)
    alerts = detect_drift(history, scoreboard.headline, window=window, bands=bands)
    if record:
        append_timeseries(scoreboard, timeseries_path)
    if notify:
        for a in alerts:
            notify(a.message)
    return alerts
