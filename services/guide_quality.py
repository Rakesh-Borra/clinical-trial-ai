"""
Non-destructive quality audit for patient-guide statements.

Flags exact duplicates, near-duplicates, overly technical wording, and weak
source support for coordinator review. Never deletes, merges, approves, or
rejects statements. Dose, timing, visit, eligibility, and risk differences
are treated as clinically distinct.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from models.plain_language_models import GuideItem, PatientGuide
from services.source_verifier import normalize_text

# Tokens that usually mean two similar strings are still clinically different.
_CLINICAL_DISTINGUISHERS = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mmol|%|kg)|"
    r"day\s*-?\d+|week\s*\d+|month\s*\d+|"
    r"arm\s*[a-z]|cohort\s*\d+|dose|dosing|twice|once|daily|weekly)\b",
    re.IGNORECASE,
)

_TECHNICAL_MARKERS = (
    "protocol",
    "investigator",
    "sponsor",
    "case report form",
    "crf",
    "ich ",
    "gcp",
    "regulatory",
    "amendment",
    "pharmacokinetic",
    "pharmacodynamic",
    "randomisation code",
    "randomization code",
    "blinding envelope",
    "interactive web response",
    "iwrs",
    "ixrs",
)


@dataclass(frozen=True)
class QualityFinding:
    kind: str  # exact_duplicate | near_duplicate | technical | weak_source
    item_ids: tuple[str, ...]
    summary: str
    detail: str = ""


def _norm(text: str) -> str:
    return normalize_text(text or "")


def _has_clinical_distinguisher(text: str) -> bool:
    return bool(_CLINICAL_DISTINGUISHERS.search(text or ""))


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", (text or "").casefold()) if len(tok) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def audit_guide_quality(guide: PatientGuide) -> List[QualityFinding]:
    """Return reviewable quality concerns; empty when nothing suspicious."""
    items = [i for i in guide.all_items() if i.item_id and not i.rejected]
    findings: List[QualityFinding] = []

    # Exact duplicates (normalized plain text).
    by_text: dict[str, list[GuideItem]] = defaultdict(list)
    for item in items:
        key = _norm(item.plain_text)
        if key:
            by_text[key].append(item)
    for key, group in by_text.items():
        if len(group) < 2:
            continue
        ids = tuple(i.item_id for i in group if i.item_id)
        pages = sorted({i.source_page for i in group if i.source_page is not None})
        findings.append(QualityFinding(
            kind="exact_duplicate",
            item_ids=ids,
            summary=f"{len(group)} statements share the same wording",
            detail=f"Pages {pages}; ids {', '.join(ids[:6])}"
            + ("…" if len(ids) > 6 else ""),
        ))

    # Near-duplicates: high token overlap, same section, no clinical distinguishers.
    candidates = [
        i for i in items
        if i.item_id and not _has_clinical_distinguisher(i.plain_text)
        and not _has_clinical_distinguisher(i.original_text)
    ]
    seen_pairs: set[tuple[str, str]] = set()
    for index, left in enumerate(candidates):
        left_tok = _token_set(left.plain_text)
        for right in candidates[index + 1:]:
            if left.section != right.section:
                continue
            if _norm(left.plain_text) == _norm(right.plain_text):
                continue  # already reported as exact
            score = _jaccard(left_tok, _token_set(right.plain_text))
            if score < 0.85:
                continue
            pair = tuple(sorted((left.item_id, right.item_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            findings.append(QualityFinding(
                kind="near_duplicate",
                item_ids=pair,
                summary="Two statements look very similar (same section)",
                detail=f"Overlap {score:.0%}; pages "
                f"{left.source_page}/{right.source_page}",
            ))

    for item in items:
        blob = f"{item.plain_text} {item.original_text}".casefold()
        if any(marker in blob for marker in _TECHNICAL_MARKERS):
            findings.append(QualityFinding(
                kind="technical",
                item_ids=(item.item_id,),
                summary="May be coordinator/technical rather than patient-facing",
                detail=f"Page {item.source_page}; section {item.section}",
            ))
        if item.source_status in {"unverified", "unchecked"} or item.generation_failed:
            findings.append(QualityFinding(
                kind="weak_source",
                item_ids=(item.item_id,),
                summary="Lacks solid source verification",
                detail=(item.source_issue or item.review_reasons[0]
                        if item.review_reasons else item.source_status)
                or "unverified",
            ))

    return findings


def group_findings(findings: Sequence[QualityFinding]) -> dict[str, List[QualityFinding]]:
    """Bucket findings for compact UI display."""
    order = ("exact_duplicate", "near_duplicate", "technical", "weak_source")
    grouped = {key: [] for key in order}
    for finding in findings:
        grouped.setdefault(finding.kind, []).append(finding)
    return {key: grouped[key] for key in order if grouped.get(key)}
