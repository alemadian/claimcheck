"""
The source-of-truth corpus and the retriever over it.

A content-review agent is only as trustworthy as the corpus it checks against,
so the corpus is treated as a pinned, versioned artifact: every passage records
the URL it came from and the date it was captured. Swapping the corpus is a
visible, reviewable change, exactly as it should be for a thing that decides
whether marketing copy is true.

The bundled corpus (``corpus/stripe_docs.jsonl``) is real, public Stripe
content captured on a fixed date. Nothing here is invented; each passage cites
its source URL so a reviewer can re-verify it.

The retriever is deliberately simple and deterministic (lexical overlap with a
relevance floor). The point of this repo is the *trust layer and the
evaluation*, not a novel retriever; a production system swaps in embeddings
behind the same interface without touching the harness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Doc:
    """One retrievable passage of source-of-truth, with provenance."""
    id: str
    text: str
    source_url: str
    captured_on: str
    doc_type: str = "unknown"

    @staticmethod
    def from_dict(d: dict) -> "Doc":
        for key in ("id", "text", "source_url", "captured_on"):
            if key not in d:
                raise ValueError(f"corpus doc missing required field {key!r}")
        return Doc(
            id=str(d["id"]),
            text=str(d["text"]),
            source_url=str(d["source_url"]),
            captured_on=str(d["captured_on"]),
            doc_type=str(d.get("doc_type", "unknown")),
        )


class Corpus:
    """An immutable, id-addressable set of source-of-truth passages."""

    def __init__(self, docs: Iterable[Doc]) -> None:
        self.docs: list[Doc] = list(docs)
        self._by_id: dict[str, Doc] = {}
        for doc in self.docs:
            if doc.id in self._by_id:
                raise ValueError(f"duplicate corpus doc id {doc.id!r}")
            self._by_id[doc.id] = doc

    def __len__(self) -> int:
        return len(self.docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._by_id

    def get(self, doc_id: str) -> Optional[Doc]:
        return self._by_id.get(doc_id)

    def ids(self) -> set[str]:
        return set(self._by_id)

    @staticmethod
    def load(path: str | Path) -> "Corpus":
        path = Path(path)
        docs: list[Doc] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    docs.append(Doc.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if not docs:
            raise ValueError(f"no corpus documents loaded from {path}")
        return Corpus(docs)


# --------------------------------------------------------------------------- #
# Tokenization (shared by the retriever and the reference reviewer)            #
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")

# Stopwords are ignored so a query only matches on content words. Without this,
# a claim like "Stripe is available in 195 countries" would match any passage
# containing "is"/"in" and wrongly pull pricing context, defeating the
# must-abstain behavior on out-of-corpus claims.
_STOP = frozenset(
    "a an the is are was were be been being of to in on at for with and or "
    "you your our we they it its their this that these those will would can "
    "could should as by from per each only just no not any all more than up "
    "stripe stripes".split()
)


def content_tokens(s: str) -> set[str]:
    return {t for t in _WORD.findall((s or "").lower()) if t not in _STOP}


def overlap_score(query: str, text: str) -> float:
    """Fraction of the query's content words present in ``text``."""
    q = content_tokens(query)
    d = content_tokens(text)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


class LexicalRetriever:
    """Rank corpus passages by content-word overlap, with a relevance floor.

    ``min_score`` is the floor a grounded reviewer needs so it abstains on
    out-of-corpus claims instead of surfacing a loosely-related passage.
    Passages below the floor are dropped, which makes "empty context -> fail
    closed" reachable on the must-abstain split.
    """

    def __init__(self, min_score: float = 0.3) -> None:
        self.min_score = min_score

    def retrieve(self, query: str, corpus: Corpus, k: int) -> list[Doc]:
        scored = sorted(
            corpus.docs,
            key=lambda doc: (overlap_score(query, doc.text), doc.id),
            reverse=True,
        )
        relevant = [doc for doc in scored if overlap_score(query, doc.text) >= self.min_score]
        return relevant[:k]
