"""
Loading, validating and slicing the golden dataset.

A golden dataset is one or more .jsonl files; each non-blank line is one
ClaimCase. Loading is strict: a malformed line raises with file and line
number, so a bad case blocks the build instead of being skipped. Gold passage
references are cross-checked against the loaded corpus, so a dangling reference
also blocks the build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .corpus import Corpus
from .schema import Adversarial, ClaimCase, Verdict


def load_jsonl(path: str | Path) -> list[ClaimCase]:
    path = Path(path)
    cases: list[ClaimCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                case = ClaimCase.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases


def load_dir(path: str | Path) -> list[ClaimCase]:
    """Load and merge every *.jsonl under a directory (sorted, dedup across files)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    cases: list[ClaimCase] = []
    seen: set[str] = set()
    for f in files:
        for case in load_jsonl(f):
            if case.id in seen:
                raise ValueError(f"duplicate case id across files: {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases


def validate_against_corpus(cases: Iterable[ClaimCase], corpus: Corpus) -> None:
    """Every gold passage id must exist in the corpus, else the build fails."""
    for case in cases:
        unknown = [d for d in case.gold_doc_ids if d not in corpus]
        if unknown:
            raise ValueError(
                f"case {case.id!r}: gold_doc_ids reference passages not in the "
                f"corpus: {unknown}"
            )


# --------------------------------------------------------------------------- #
# Slicing                                                                       #
# --------------------------------------------------------------------------- #
def filter_cases(
    cases: Iterable[ClaimCase],
    *,
    split: str | None = None,
    verdict: Verdict | None = None,
    adversarial: Adversarial | None = None,
    predicate: Callable[[ClaimCase], bool] | None = None,
) -> list[ClaimCase]:
    out = []
    for c in cases:
        if split is not None and c.split != split:
            continue
        if verdict is not None and c.gold_verdict is not verdict:
            continue
        if adversarial is not None and c.adversarial is not adversarial:
            continue
        if predicate is not None and not predicate(c):
            continue
        out.append(c)
    return out


def slices_for(case: ClaimCase) -> list[str]:
    """Return the slice labels a case belongs to (for per-slice aggregation)."""
    labels = [
        f"split:{case.split}",
        f"gold_verdict:{case.gold_verdict.value}",
        f"claim_type:{case.claim_type}",
    ]
    if case.adversarial:
        labels.append(f"adversarial:{case.adversarial.value}")
    return labels
