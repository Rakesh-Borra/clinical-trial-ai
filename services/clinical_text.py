"""
Deterministic meaning checks for rewritten clinical statements.

String comparison of numbers is not safe ("5" is found inside ".5"), and a
rewrite can keep every number while dropping a condition or flipping a
negation. These helpers compare numbers as decimals with their units, and
compare conditions and negations between an original and its rewrite.
They never call a model.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
)

_WORD_NUMBERS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
    if i >= 2
}

_UNITS = {
    "ml": "ml", "millilitre": "ml", "millilitres": "ml",
    "milliliter": "ml", "milliliters": "ml",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "mcg": "mcg", "µg": "mcg", "microgram": "mcg", "micrograms": "mcg",
    "g": "g", "gram": "g", "grams": "g", "kg": "kg",
    "day": "day", "days": "day", "week": "week", "weeks": "week",
    "month": "month", "months": "month", "year": "year", "years": "year",
    "hour": "hour", "hours": "hour", "minute": "minute", "minutes": "minute",
    "%": "%", "percent": "%", "mg/dl": "mg/dl",
}
_UNIT_RE = re.compile(
    r"\s*(mg/dl|millilit(?:re|er)s?|milligrams?|micrograms?|grams?|ml|mg|mcg|µg|kg|"
    r"days?|weeks?|months?|years?|hours?|minutes?|percent|%|g)(?![A-Za-z])",
    re.IGNORECASE,
)


def canonical_number(token: str) -> str:
    """'.5' -> '0.5', '1,000' -> '1000'. Trailing zeros are kept: 7.0 != 7."""
    cleaned = token.replace(",", "")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    return cleaned


def _with_number_words(text: str) -> str:
    def swap(match: re.Match) -> str:
        return _WORD_NUMBERS[match.group(0).lower()]

    pattern = r"\b(?:" + "|".join(_WORD_NUMBERS) + r")\b(?=\s*[-–]?\s*(?:mL|ml|mg|days?|weeks?|months?|years?|hours?|minutes?|visits?|times|tablets?|doses?|pills?|%))"
    return re.sub(pattern, swap, text, flags=re.IGNORECASE)


def numeric_facts(text: str) -> List[Tuple[str, str]]:
    """(canonical value, canonical unit or '') for every number in the text."""
    text = _with_number_words(text or "")
    facts: List[Tuple[str, str]] = []
    for match in NUMBER_RE.finditer(text):
        unit_match = _UNIT_RE.match(text, match.end())
        unit = _UNITS.get(unit_match.group(1).lower(), "") if unit_match else ""
        facts.append((canonical_number(match.group(0)), unit))
    return facts


def _label(fact: Tuple[str, str]) -> str:
    value, unit = fact
    return value


def _matches(fact: Tuple[str, str], pool: List[Tuple[str, str]]) -> bool:
    value, unit = fact
    return any(
        v == value and (not unit or not u or unit == u) for v, u in pool
    )


def missing_numbers(original: str, rewritten: str) -> List[str]:
    """Original numbers (with units) that the rewrite does not carry exactly."""
    pool = numeric_facts(rewritten)
    out: List[str] = []
    for fact in numeric_facts(original):
        label = _label(fact)
        if not _matches(fact, pool) and label not in out:
            out.append(label)
    return out


def added_numbers(original: str, rewritten: str) -> List[str]:
    """Rewrite numbers that the original does not carry exactly."""
    pool = numeric_facts(original)
    out: List[str] = []
    for fact in numeric_facts(rewritten):
        label = _label(fact)
        if not _matches(fact, pool) and label not in out:
            out.append(label)
    return out


# --- added or dropped detail -------------------------------------------------

_RISKY_DETAIL = re.compile(
    r"\b(within|twice|three times|once a|daily|weekly|monthly|every other day|"
    r"at least|at most|no more than|up to|no later than|before|after)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(r"\b(approximately|about|around|roughly)\b", re.IGNORECASE)


def added_details(original: str, rewritten: str) -> List[str]:
    """Timing/frequency wording in the rewrite that the original never says."""
    original_hits = {m.group(0).lower() for m in _RISKY_DETAIL.finditer(original)}
    out: List[str] = []
    for match in _RISKY_DETAIL.finditer(rewritten):
        word = match.group(0).lower()
        if word in ("before", "after"):
            continue
        if word not in original_hits and word not in out:
            out.append(word)
    return out


def dropped_hedge(original: str, rewritten: str) -> bool:
    return bool(_HEDGE.search(original)) and not _HEDGE.search(rewritten)


# --- conditions -------------------------------------------------------------

_ORIG_CONDITION = re.compile(
    r"\b(?:when|if|where|as)\s+(?:applicable|needed|indicated|relevant|required)\b",
    re.IGNORECASE,
)
_REWRITE_CONDITION_OK = re.compile(
    r"\b(applicable|if needed|as needed|when needed|if indicated|when indicated|"
    r"if required|when required|depending|only for some|not everyone|may not apply|"
    r"if relevant|when relevant)\b",
    re.IGNORECASE,
)
_ANY_CONDITION = re.compile(r"\b(if|when|unless|only if|in case|whether|depend)", re.IGNORECASE)
_ADDED_CONDITION = re.compile(r"\b(?:if|only if|unless)\s+you\b", re.IGNORECASE)


def condition_problems(original: str, rewritten: str) -> List[str]:
    problems: List[str] = []
    cond = _ORIG_CONDITION.search(original)
    if cond and not _REWRITE_CONDITION_OK.search(rewritten):
        problems.append(
            f'The original condition "{cond.group(0)}" is missing or changed in the rewrite'
        )
    added = _ADDED_CONDITION.search(rewritten)
    if added and not _ANY_CONDITION.search(original):
        problems.append(
            f'The rewrite adds a condition ("{added.group(0)} ...") that the original does not state'
        )
    return problems


# --- negation ---------------------------------------------------------------

_NEGATORS = {
    "not", "no", "never", "without", "cannot", "cant", "dont", "doesnt", "isnt",
    "arent", "wont", "shouldnt", "mustnt", "neither", "nor", "none", "nothing",
}
_STOP = {
    "a", "an", "the", "be", "is", "are", "was", "were", "to", "of", "for", "and",
    "or", "in", "on", "at", "by", "your", "you", "this", "that", "it", "any",
    "been", "being", "do", "does", "did", "have", "has", "will", "may", "might",
    "can", "could", "would", "should", "must", "with", "from", "as", "if",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[.;:,!?()]", text.lower().replace("’", "'"))


def negation_groups(text: str) -> List[Set[str]]:
    """Content stems that follow each negator inside the same clause."""
    words = [w.replace("'", "") for w in _words(text)]
    groups: List[Set[str]] = []
    for i, word in enumerate(words):
        if word not in _NEGATORS:
            continue
        stems: Set[str] = set()
        seen = 0
        for follower in words[i + 1:]:
            if follower in ".;:,!?()":
                break
            seen += 1
            if seen > 5:
                break
            if follower in _NEGATORS:
                continue
            if follower not in _STOP:
                stems.add(_stem(follower))
        if stems:
            groups.append(stems)
    return groups


def clause_negation_groups(text: str) -> List[Set[str]]:
    """Every content stem of each clause that contains a negator."""
    groups: List[Set[str]] = []
    for clause in re.split(r"[.;:!?,()]", text.lower().replace("’", "'")):
        words = [w.replace("'", "") for w in re.findall(r"[a-z]+(?:'[a-z]+)?", clause)]
        if any(w in _NEGATORS for w in words):
            stems = {_stem(w) for w in words if w not in _NEGATORS and w not in _STOP}
            if stems:
                groups.append(stems)
    return groups


def _all_stems(text: str) -> Set[str]:
    return {_stem(w.replace("'", "")) for w in _words(text) if w.isalpha() or "'" in w}


def negation_problems(original: str, rewritten: str) -> List[str]:
    """Flag a rewrite that reverses, drops, or invents a negation."""
    problems: List[str] = []
    original_groups = negation_groups(original)
    rewrite_groups = negation_groups(rewritten)
    rewrite_stems = _all_stems(rewritten)
    original_stems = _all_stems(original)

    rewrite_clauses = clause_negation_groups(rewritten)
    for group in original_groups:
        if any(group & other for other in rewrite_groups + rewrite_clauses):
            continue
        present = group & rewrite_stems
        if present:
            problems.append(
                "The original says not/no for "
                + ", ".join(sorted(present))
                + ", but the rewrite does not keep that negation"
            )
    for group in rewrite_groups:
        if any(group & other for other in original_groups):
            continue
        shared = group & original_stems
        if shared:
            problems.append(
                "The rewrite negates "
                + ", ".join(sorted(shared))
                + ", which the original does not"
            )
    return problems


_NEGATED_BEFORE = re.compile(
    r"\b(?:not|never|no|n't|cannot|without)\b[\w\s,'’-]{0,40}$", re.IGNORECASE
)


def is_negated_in_context(text: str, start: int) -> bool:
    """True when the phrase starting at `start` sits right after a negator."""
    window = text[max(0, start - 45):start]
    window = re.split(r"[.;:!?]", window)[-1]
    return bool(_NEGATED_BEFORE.search(window))


def original_restricts(original: str, phrase: str) -> bool:
    """True when the original itself negates the action named by the phrase."""
    verbs = {_stem(w) for w in re.findall(r"[a-z]+", phrase.lower()) if w not in _STOP}
    if not verbs:
        return False
    return any(verbs & group for group in negation_groups(original))
