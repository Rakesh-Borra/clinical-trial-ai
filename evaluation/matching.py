"""
Compare what the app extracted from a protocol with hand-labelled ground truth.

Matching is deliberately simple and explainable:
two statements match when they share most of their words (Dice score)
AND every number in the ground-truth statement also appears in the
extracted one. Each ground-truth item can be matched at most once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from models.protocol_models import ProtocolAnalysis

CATEGORIES = (
    "inclusion_criteria",
    "exclusion_criteria",
    "visits",
    "procedures",
    "safety_information",
)

MATCH_THRESHOLD = 0.5

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "be", "is", "are", "will", "must", "who", "that", "at", "by", "as",
    "participants", "participant", "patients", "patient", "subjects", "subject",
}
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokens(text: str) -> set:
    """Lower-case content words and numbers."""
    return {t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS}


def numbers(text: str) -> set:
    return set(_NUMBER.findall(text))


def similarity(truth: str, extracted: str) -> float:
    """
    Dice word overlap between two statements (0 to 1).

    Returns 0 if any number in the ground truth is missing from the
    extracted statement, because a changed number is a different fact.
    """
    if not numbers(truth) <= numbers(extracted):
        return 0.0
    a, b = tokens(truth), tokens(extracted)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


@dataclass
class Item:
    text: str
    page: Optional[int] = None


@dataclass
class CategoryScore:
    category: str
    truth_count: int
    extracted_count: int
    matched: List[Tuple[Item, Item, float]] = field(default_factory=list)
    missed: List[Item] = field(default_factory=list)
    extra: List[Item] = field(default_factory=list)

    @property
    def recall(self) -> Optional[float]:
        return len(self.matched) / self.truth_count if self.truth_count else None

    @property
    def precision(self) -> Optional[float]:
        return len(self.matched) / self.extracted_count if self.extracted_count else None

    @property
    def page_accuracy(self) -> Optional[float]:
        with_pages = [(t, e) for t, e, _ in self.matched if t.page is not None]
        if not with_pages:
            return None
        correct = sum(1 for t, e in with_pages if t.page == e.page)
        return correct / len(with_pages)


def match_items(truth: Sequence[Item], extracted: Sequence[Item]) -> CategoryScore:
    """Greedy one-to-one matching, best pairs first."""
    pairs = []
    for ti, t in enumerate(truth):
        for ei, e in enumerate(extracted):
            score = similarity(t.text, e.text)
            if score >= MATCH_THRESHOLD:
                pairs.append((score, ti, ei))
    pairs.sort(reverse=True)

    used_t, used_e, matched = set(), set(), []
    for score, ti, ei in pairs:
        if ti in used_t or ei in used_e:
            continue
        used_t.add(ti)
        used_e.add(ei)
        matched.append((truth[ti], extracted[ei], score))

    return CategoryScore(
        category="",
        truth_count=len(truth),
        extracted_count=len(extracted),
        matched=matched,
        missed=[t for i, t in enumerate(truth) if i not in used_t],
        extra=[e for i, e in enumerate(extracted) if i not in used_e],
    )


def extracted_items(analysis: ProtocolAnalysis, category: str) -> List[Item]:
    """Turn the app's structured output into comparable text items."""
    if category in ("inclusion_criteria", "exclusion_criteria", "safety_information"):
        return [Item(i.description, i.source_page) for i in getattr(analysis, category)]
    if category == "procedures":
        return [
            Item(f"{p.name} {p.description or ''}".strip(), p.source_page)
            for p in analysis.procedures
        ]
    if category == "visits":
        return [
            Item(
                " ".join(x for x in [v.visit_name, v.timing or "", " ".join(v.procedures)] if x),
                v.source_page,
            )
            for v in analysis.visits
        ]
    raise ValueError(f"Unknown category: {category}")


def truth_items(ground_truth: dict, category: str) -> List[Item]:
    return [Item(x["text"], x.get("page")) for x in ground_truth.get(category, [])]


def score_protocol(analysis: ProtocolAnalysis, ground_truth: dict) -> List[CategoryScore]:
    scores = []
    for category in CATEGORIES:
        score = match_items(truth_items(ground_truth, category), extracted_items(analysis, category))
        score.category = category
        scores.append(score)
    return scores


def overall(scores: Iterable[CategoryScore]) -> dict:
    scores = list(scores)
    truth = sum(s.truth_count for s in scores)
    extracted = sum(s.extracted_count for s in scores)
    matched = sum(len(s.matched) for s in scores)
    paged = [(t, e) for s in scores for t, e, _ in s.matched if t.page is not None]
    return {
        "truth": truth,
        "extracted": extracted,
        "matched": matched,
        "recall": matched / truth if truth else None,
        "precision": matched / extracted if extracted else None,
        "page_accuracy": (sum(1 for t, e in paged if t.page == e.page) / len(paged)) if paged else None,
    }


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"
