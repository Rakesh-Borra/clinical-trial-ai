"""
Completeness of patient-facing coverage, kept separate from "did the analysis
run". Pure rules, no model calls.

It distinguishes four different facts that must never be confused:

- pages extracted:   pages that yielded text from the PDF
- pages analyzed:    pages whose content was actually sent to the model
- statements:        what the guide holds for each critical category
- verified:          statements whose source check passed (and approved ones)

Generating many statements does not mean every critical category is covered,
so publication also requires each critical category to be represented by at
least one approved, verified statement, or a recorded "not applicable" override
that passes the safeguards in `override_block_reason`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

# key -> (label, guide section attributes, audit categories in the source)
CRITICAL_CATEGORIES = {
    "eligibility": (
        "Eligibility", ("who_can_join", "who_cannot_join"), ("eligibility",),
    ),
    "visits": ("Study visits and schedule", ("visits",), ("visits",)),
    "procedures": (
        "Procedures and blood collection", ("procedures",), ("procedures",),
    ),
    "interventions": ("Study interventions", ("interventions",), ()),
    "risks": ("Risks and safety", ("safety",), ("risks", "safety")),
    "consent": (
        "Consent and withdrawal", ("consent",), ("consent", "withdrawal"),
    ),
}

MIN_OVERRIDE_REASON_CHARS = 10


def analysis_counts(analysis: Any) -> Dict[str, int]:
    """Extracted fact counts for each critical category."""
    info = list(getattr(analysis, "patient_information", []) or [])

    def count(*categories: str) -> int:
        return sum(1 for i in info if i.category in categories)

    return {
        "eligibility": len(analysis.inclusion_criteria)
        + len(analysis.exclusion_criteria),
        "visits": len(analysis.visits),
        "procedures": len(analysis.procedures) + count("blood_collection"),
        "interventions": count("interventions"),
        "risks": len(analysis.safety_information) + count("risks"),
        "consent": count("consent_withdrawal"),
    }


def _section_items(guide: Any, attrs: Iterable[str]) -> List[Any]:
    return [i for attr in attrs for i in getattr(guide, attr).items]


def build_coverage(
    pipeline_result: Mapping[str, Any],
    guide: Any = None,
) -> Dict[str, Any]:
    """Page counts and per-category coverage, with the gaps listed."""
    analysis = pipeline_result.get("analysis")
    counts = analysis_counts(analysis) if analysis is not None else {}
    audit = (pipeline_result.get("completeness_audit") or {}).get("categories", {})
    pages = dict(pipeline_result.get("pages") or {})

    categories: Dict[str, Dict[str, Any]] = {}
    gaps: List[str] = []
    for key, (label, attrs, audit_keys) in CRITICAL_CATEGORIES.items():
        evidence = sorted({
            p for a in audit_keys for p in (audit.get(a) or {}).get("source_pages", [])
        })
        entry: Dict[str, Any] = {
            "label": label,
            "extracted": counts.get(key, 0),
            "source_pages": evidence,
            "source_evidence": bool(evidence),
        }
        if guide is not None:
            items = _section_items(guide, attrs)
            kept = [i for i in items if not i.rejected]
            entry["statements"] = len(items)
            entry["verified"] = sum(
                1 for i in kept
                if i.source_status == "verified" and i.verification == "verified"
            )
            entry["approved"] = sum(
                1 for i in kept
                if i.approved
                and i.source_status == "verified"
                and i.verification == "verified"
            )
            entry["unverified"] = sum(
                1 for i in kept if i.source_status != "verified"
            )
        if not entry["extracted"]:
            entry["status"] = "missing"
        elif guide is not None and not entry["verified"]:
            entry["status"] = "unverified"
        else:
            entry["status"] = "covered"
        if entry["status"] != "covered":
            gaps.append(key)
        categories[key] = entry
    return {
        "pages": pages,
        "categories": categories,
        "gaps": gaps,
        # Without a completeness audit nothing proves a category is absent.
        "audit_available": isinstance(
            pipeline_result.get("completeness_audit"), Mapping
        ),
    }


def override_block_reason(
    coverage: Mapping[str, Any],
    key: str,
    reason: str,
    analysis_complete: bool,
) -> Optional[str]:
    """
    Why a "Not applicable" override is not allowed (None when it is).

    The override only records that the protocol genuinely has no such section.
    It can never hide extraction that failed, source contradictions, or a
    category the protocol text actually contains.
    """
    entry = (coverage.get("categories") or {}).get(key)
    if entry is None:
        return "Unknown category."
    if len((reason or "").strip()) < MIN_OVERRIDE_REASON_CHARS:
        return (
            "Record why this is not applicable (at least "
            f"{MIN_OVERRIDE_REASON_CHARS} characters), after checking the protocol."
        )
    if not coverage.get("audit_available"):
        return (
            "Not allowed: this analysis has no completeness audit, so the "
            "protocol cannot be shown to lack this information. Re-run the analysis."
        )
    if not analysis_complete:
        return (
            "Not allowed while the analysis is incomplete: the information "
            "may simply not have been extracted."
        )
    if entry.get("source_evidence"):
        pages = ", ".join(str(p) for p in entry["source_pages"])
        return (
            "Not allowed: the protocol text appears to contain this "
            f"information (page {pages}). Re-run the analysis or add the "
            "statement instead."
        )
    if entry.get("unverified"):
        return (
            "Not allowed: statements in this category failed source "
            "verification. Resolve them first."
        )
    return None


def coverage_block_reason(
    coverage: Mapping[str, Any],
    overrides: Optional[Mapping[str, Mapping[str, Any]]],
    analysis_complete: bool,
) -> Optional[str]:
    """Why critical-category coverage blocks publication (None when it does not)."""
    overrides = overrides or {}
    missing: List[str] = []
    for key, entry in (coverage.get("categories") or {}).items():
        if entry.get("approved", 0) > 0:
            continue
        recorded = overrides.get(key)
        if recorded and override_block_reason(
            coverage, key, recorded.get("reason", ""), analysis_complete
        ) is None and not entry.get("extracted"):
            continue
        missing.append(entry["label"])
    if missing:
        return (
            "Cannot publish: no approved, verified statement covers "
            + "; ".join(missing)
            + ". Approve or add a statement, or mark the category "
            "“Not applicable” with a reason if the protocol truly has none."
        )
    return None
