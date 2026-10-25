"""
Explain why a statement failed its source check, for the coordinator.

This only reads text and compares words: it never marks anything verified.
It tells the coordinator whether the statement looks unsupported, only
differs in wording, is on a different page, or comes from an example.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional

from models.plain_language_models import GuideItem
from services.source_verifier import normalize_text, split_illustrative, _on_page

NOT_FOUND = "not_found"
NEAR_MATCH = "near_match"
OTHER_PAGE = "other_page"
ILLUSTRATIVE = "illustrative"
TIMING = "timing"
NO_TEXT = "no_page_text"

_WORD = re.compile(r"[a-z0-9]+")
_FILLER = {"the", "a", "an", "of", "and", "or", "to", "at", "in", "on", "for", "is", "are"}

EXPLANATIONS = {
    NOT_FOUND: (
        "Nothing on the cited page closely matches this statement. It may be "
        "unsupported or invented: edit it to match the protocol, or reject it."
    ),
    NEAR_MATCH: (
        "A similar line exists on the cited page, but the wording differs. "
        "Compare the lines below; if the protocol really says this, edit the "
        "statement to use the protocol's wording and recheck it."
    ),
    OTHER_PAGE: (
        "The text exists on a different page than the one cited. The page "
        "reference may be wrong; correct it by editing the statement."
    ),
    ILLUSTRATIVE: (
        "This only appears under an example or review-question heading, which "
        "is not protocol content. Reject it unless the protocol states it elsewhere."
    ),
    TIMING: (
        "The visit name was found, but its timing is not written together "
        "with it on the cited page."
    ),
    NO_TEXT: "The text of the cited page is not available in this saved review.",
}


def _tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(normalize_text(text)) if t not in _FILLER]


def closest_lines(statement: str, page_text: str, limit: int = 3) -> List[str]:
    """Page lines that share the most words with the statement."""
    wanted = set(_tokens(statement))
    if not wanted:
        return []
    scored = []
    for line in page_text.splitlines():
        have = set(_tokens(line))
        if have:
            scored.append((len(wanted & have) / len(wanted), line.strip()))
    scored.sort(key=lambda pair: -pair[0])
    return [line for score, line in scored[:limit] if score > 0.2]


def inspect_source(
    item: GuideItem, pages: Optional[Mapping[str, str]]
) -> Dict[str, object]:
    """Where the statement stands against its cited page: kind, text, lines."""
    page_text = (pages or {}).get(str(item.source_page)) if item.source_page else None
    if not page_text:
        return {"kind": NO_TEXT, "explanation": EXPLANATIONS[NO_TEXT],
                "page_text": "", "lines": []}
    authoritative, illustrative = split_illustrative(page_text)
    statement = item.original_text
    lines = closest_lines(statement, authoritative)
    if (item.source_issue or "").lower().startswith("timing"):
        kind = TIMING
    elif illustrative and _on_page(statement, normalize_text(illustrative)):
        kind = ILLUSTRATIVE
    elif any(
        _on_page(statement, normalize_text(split_illustrative(text)[0]))
        for page, text in (pages or {}).items()
        if page != str(item.source_page)
    ):
        kind = OTHER_PAGE
    else:
        wanted = set(_tokens(statement))
        best = max(
            (len(wanted & set(_tokens(l))) / len(wanted) for l in lines), default=0.0
        ) if wanted else 0.0
        kind = NEAR_MATCH if best >= 0.6 else NOT_FOUND
    return {
        "kind": kind,
        "explanation": EXPLANATIONS[kind],
        "page_text": authoritative,
        "lines": lines,
    }
