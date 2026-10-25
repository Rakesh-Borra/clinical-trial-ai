"""
Completeness audit: which patient-relevant categories look missing or
uncertain after extraction. Pure rules, no model calls.

It compares the categories found in the protocol text (from section headings,
with content cues as a weaker signal) against what the extraction returned.
It never changes the analysis; it only reports what a coordinator should
double-check. Source text is read from the kept blocks, so skipped
administrative sections do not count as patient information.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from services.section_classifier import PATIENT_CATEGORIES

# Categories the extraction schema can hold, and how to count them.
def _info(analysis, *categories):
    return sum(
        1 for i in getattr(analysis, "patient_information", []) or []
        if i.category in categories
    )


_EXTRACTED = {
    "eligibility": lambda a: len(a.inclusion_criteria) + len(a.exclusion_criteria),
    "visits": lambda a: len(a.visits),
    "procedures": lambda a: len(a.procedures) + _info(a, "blood_collection"),
    "safety": lambda a: len(a.safety_information) + _info(a, "risks"),
    "risks": lambda a: len(a.safety_information) + _info(a, "risks"),
    "consent": lambda a: _info(a, "consent_withdrawal"),
    "withdrawal": lambda a: _info(a, "consent_withdrawal"),
}
_LABELS = {
    "purpose": "study purpose", "eligibility": "eligibility criteria",
    "visits": "visit schedule", "procedures": "procedures and tests",
    "risks": "risks and benefits", "safety": "safety information",
    "consent": "informed consent", "withdrawal": "withdrawal",
    "responsibilities": "participant responsibilities",
}


def audit_completeness(blocks: Iterable[Any], analysis: Any) -> Dict[str, Any]:
    kept = [b for b in blocks if b.keep and b.text.strip()]
    by_category: Dict[str, Dict[str, Any]] = {
        c: {"pages": set(), "uncertain_only": True} for c in PATIENT_CATEGORIES
    }
    for block in kept:
        for category in block.categories:
            if category in by_category:
                by_category[category]["pages"].add(block.page_number)
                if block.category_source == "heading":
                    by_category[category]["uncertain_only"] = False

    categories: Dict[str, Dict[str, Any]] = {}
    missing, uncertain = [], []
    for category in PATIENT_CATEGORIES:
        pages = sorted(p for p in by_category[category]["pages"] if p is not None)
        in_source = bool(by_category[category]["pages"])
        counter = _EXTRACTED.get(category)
        extracted = counter(analysis) if counter else None
        if counter is None:
            status = "in_source_not_extracted_field" if in_source else "not_found"
        elif extracted:
            status = "ok"
        elif in_source:
            status = "not_extracted"
        elif category in {"consent", "withdrawal"}:
            status = "not_found"  # optional in the source; flagged as uncertain below
        else:
            status = "missing"
        entry = {"status": status, "source_pages": pages, "extracted": extracted}
        if in_source and by_category[category]["uncertain_only"]:
            entry["classification"] = "content cues only (no recognized heading)"
        categories[category] = entry
        label = _LABELS[category]
        if status == "missing":
            missing.append(label)
        elif status == "not_extracted":
            uncertain.append(
                f"{label}: found in the protocol (pages {pages}) but nothing "
                "was extracted"
            )
        elif status == "not_found" and category in {"purpose", "consent", "withdrawal"}:
            uncertain.append(f"{label}: no section recognized in the protocol text")

    total = sum(len(b.text) for b in kept) or 1
    unclassified = sum(len(b.text) for b in kept if not b.categories)
    notes = []
    if unclassified / total > 0.5:
        notes.append(
            "More than half of the analyzed text has no recognized section "
            "heading, so section categories are uncertain."
        )
    return {
        "categories": categories,
        "missing": missing,
        "uncertain": uncertain,
        "notes": notes,
        "needs_attention": bool(missing or uncertain or notes),
    }
