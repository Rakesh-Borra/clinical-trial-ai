
"""
Targeted eligibility rescue helpers.

Select eligibility-related protocol pages and decide when a focused
eligibility-only LLM pass is warranted. Never invents criteria.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from models.protocol_models import ProtocolAnalysis
from services.source_verifier import normalize_text

_PAGE_MARKER = re.compile(
    r"\[SOURCE PAGE (\d+)\]",
    flags=re.IGNORECASE,
)

# Keep aligned with services.protocol_accuracy clinical completeness markers.
_CLINICAL_MARKERS = (
    ("HbA1c", ("hba1c", "hb a1c")),
    ("eGFR", ("egfr",)),
)


def split_protocol_pages(protocol_text: str) -> list[tuple[int, str]]:
    """Split marked protocol text into (page_number, page_body) pairs."""
    parts = _PAGE_MARKER.split(protocol_text)
    # parts: [preamble, num1, body1, num2, body2, ...]
    pages: list[tuple[int, str]] = []
    if len(parts) < 3:
        return pages

    for index in range(1, len(parts), 2):
        page_number = int(parts[index])
        body = parts[index + 1].strip() if index + 1 < len(parts) else ""
        pages.append((page_number, body))
    return pages


def select_eligibility_sections(
    protocol_text: str,
) -> Tuple[Optional[str], List[str]]:
    """
    Return eligibility-focused protocol text with original page markers.

    Only pages that explicitly mention inclusion or exclusion are selected.
    If none are found confidently, returns (None, limitation warnings).
    """
    notes: list[str] = []
    pages = split_protocol_pages(protocol_text)

    if not pages:
        # Unmarked text: only accept when inclusion/exclusion headings exist.
        normalized = normalize_text(protocol_text)
        if "inclusion" in normalized or "exclusion" in normalized:
            return protocol_text.strip(), notes
        notes.append(
            "Eligibility rescue skipped: could not confidently locate "
            "eligibility sections (no [SOURCE PAGE N] markers and no "
            "inclusion/exclusion headings)."
        )
        return None, notes

    selected_chunks: list[str] = []
    for page_number, body in pages:
        normalized = normalize_text(body)
        if "inclusion" in normalized or "exclusion" in normalized:
            selected_chunks.append(
                f"[SOURCE PAGE {page_number}]\n{body}".strip()
            )

    if not selected_chunks:
        notes.append(
            "Eligibility rescue skipped: could not confidently locate "
            "eligibility sections among source pages."
        )
        return None, notes

    return "\n\n".join(selected_chunks), notes


def _eligibility_source_normalized(protocol_text: str) -> str:
    selected, _ = select_eligibility_sections(protocol_text)
    if selected:
        return normalize_text(selected)
    return ""


def _extracted_eligibility_normalized(analysis: ProtocolAnalysis) -> str:
    parts = [
        item.description for item in analysis.inclusion_criteria
    ] + [
        item.description for item in analysis.exclusion_criteria
    ]
    return normalize_text("\n".join(parts))


def missing_clinical_eligibility_markers(
    analysis: ProtocolAnalysis,
    protocol_text: str,
) -> list[str]:
    """
    Return clinical marker labels present in eligibility source text but
    absent from extracted criteria. Does not imply clinical correctness.
    """
    source = _eligibility_source_normalized(protocol_text)
    if not source:
        return []

    extracted = _extracted_eligibility_normalized(analysis)
    missing: list[str] = []
    for marker_label, variants in _CLINICAL_MARKERS:
        in_source = any(variant in source for variant in variants)
        in_extracted = any(variant in extracted for variant in variants)
        if in_source and not in_extracted:
            missing.append(marker_label)
    return missing


def should_trigger_eligibility_rescue(
    analysis: ProtocolAnalysis,
    protocol_text: str,
) -> bool:
    """
    Trigger rescue when inclusions are empty or conservative completeness
    checks suggest missing eligibility markers, and sections are locatable.
    """
    sections, _ = select_eligibility_sections(protocol_text)
    if sections is None:
        return False

    if not analysis.inclusion_criteria:
        return True

    return bool(missing_clinical_eligibility_markers(analysis, protocol_text))
