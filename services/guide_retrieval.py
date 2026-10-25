"""
Lightweight BM25 retrieval over coordinator-approved patient guide items.

Beacon BOT concept reuse: lexical BM25 over short passages with citations,
no vector database. Corpus is the published, approved guide for the active
protocol only — pending, rejected, and unpublished items are never indexed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from models.plain_language_models import GuideItem, PatientGuide
from services.source_verifier import normalize_text

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "being", "with", "as", "at", "by", "from",
    "that", "this", "these", "those", "it", "its", "i", "me", "my", "we",
    "you", "your", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "about", "what", "when", "where", "who", "how",
    "which", "there", "their", "them", "have", "has", "had", "not", "no",
}

# Okapi BM25 defaults (same family as Beacon BOT / SslPipeline).
_K1 = 1.2
_B = 0.75

# Expand common patient wording before scoring (not used as factual content).
_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "testings": ("test", "testing", "procedure"),
    "testing": ("test", "procedure"),
    "tests": ("test", "procedure"),
    "test": ("testing", "procedure"),
    "happens": ("happen", "visit", "procedure", "monitor"),
    "happen": ("visit", "procedure", "monitor"),
    "during": ("visit", "procedure"),
    "end": ("visit", "day", "follow", "timing"),
    "ends": ("visit", "day", "follow", "timing"),
    "finish": ("end", "visit", "day", "timing"),
    "finished": ("end", "visit", "day", "timing"),
    "timeline": ("visit", "day", "timing", "schedule"),
    "duration": ("visit", "day", "timing"),
    "sugar": ("glucose",),
    "bp": ("blood", "pressure"),
    "bloodwork": ("blood", "test"),
    "a1c": ("hba1c", "hemoglobin", "glucose"),
    "hba1c": ("a1c", "hemoglobin", "glucose"),
    "renal": ("kidney",),
    "kidney": ("renal",),
    "shot": ("injection",),
    "injection": ("shot",),
}


@dataclass(frozen=True)
class RetrievedPassage:
    """One approved guide statement retrieved for a patient question."""

    item: GuideItem
    score: float
    item_id: str
    section: str
    plain_text: str
    source_page: Optional[int]
    source_pages: tuple[int, ...]


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumerics; drop stopwords and tiny tokens."""
    tokens: List[str] = []
    for raw in _TOKEN.findall(normalize_text(text or "")):
        if raw in _STOPWORDS:
            continue
        if len(raw) <= 2 and not raw.isdigit():
            continue
        tokens.append(raw)
    return tokens


def expand_query_tokens(tokens: Sequence[str]) -> List[str]:
    """Add synonym tokens for informal patient language."""
    expanded: List[str] = []
    seen = set()
    for token in tokens:
        for piece in (token, *_QUERY_SYNONYMS.get(token, ())):
            if piece not in seen:
                seen.add(piece)
                expanded.append(piece)
    return expanded


def approved_items(guide: Optional[PatientGuide]) -> List[GuideItem]:
    """Patient-facing corpus: approved and not rejected only."""
    if guide is None:
        return []
    return [
        item
        for item in guide.all_items()
        if item.approved and not item.rejected
    ]


def _passage_text(item: GuideItem) -> str:
    return f"{item.plain_text} {item.original_text}"


def _pages(item: GuideItem) -> tuple[int, ...]:
    pages: List[int] = []
    if item.source_page:
        pages.append(item.source_page)
    for page in item.source_pages or []:
        if page not in pages:
            pages.append(page)
    return tuple(pages)


def _document_freqs(docs: Sequence[Sequence[str]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for tokens in docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    return df


def _bm25_score(
    query_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    *,
    df: dict[str, int],
    doc_count: int,
    avg_len: float,
) -> float:
    if not query_tokens or not doc_tokens or doc_count <= 0:
        return 0.0
    tf: dict[str, int] = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1
    length = len(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        if token not in tf:
            continue
        freq = tf[token]
        doc_freq = df.get(token, 0)
        idf = math.log(((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0)
        denom = freq + _K1 * (1.0 - _B + _B * (length / max(avg_len, 1.0)))
        score += idf * ((freq * (_K1 + 1.0)) / denom)
    return score


def retrieve_approved_passages(
    guide: Optional[PatientGuide],
    query: str,
    *,
    top_k: int = 3,
    min_score: float = 0.35,
) -> List[RetrievedPassage]:
    """
    Rank approved guide statements for ``query`` with BM25.

    Returns at most ``top_k`` passages above ``min_score``, highest first.
    Empty when the guide has no approved items or nothing is relevant.
    """
    items = approved_items(guide)
    if not items or not (query or "").strip():
        return []

    query_tokens = expand_query_tokens(tokenize(query))
    if not query_tokens:
        return []

    docs = [tokenize(_passage_text(item)) for item in items]
    lengths = [len(doc) or 1 for doc in docs]
    avg_len = sum(lengths) / len(lengths)
    df = _document_freqs(docs)
    doc_count = len(docs)

    scored: List[RetrievedPassage] = []
    for item, doc_tokens in zip(items, docs):
        score = _bm25_score(
            query_tokens,
            doc_tokens,
            df=df,
            doc_count=doc_count,
            avg_len=avg_len,
        )
        if score < min_score:
            continue
        pages = _pages(item)
        scored.append(
            RetrievedPassage(
                item=item,
                score=score,
                item_id=item.item_id or "",
                section=item.section,
                plain_text=item.plain_text,
                source_page=item.source_page,
                source_pages=pages,
            )
        )

    scored.sort(key=lambda row: (-row.score, row.section, row.plain_text))
    return scored[: max(1, top_k)]


def passage_evidence_lines(passages: Iterable[RetrievedPassage]) -> List[str]:
    """Format retrieved passages for a grounded model prompt."""
    lines: List[str] = []
    for passage in passages:
        page = (
            f" (protocol page {passage.source_page})"
            if passage.source_page is not None
            else ""
        )
        identity = f" [{passage.item_id}]" if passage.item_id else ""
        lines.append(f"-{identity} {passage.plain_text}{page}")
    return lines
