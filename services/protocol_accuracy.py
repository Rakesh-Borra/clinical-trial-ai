
"""
Deterministic post-LLM accuracy checks for protocol analysis.

These helpers ground metadata, repair unique source pages, and warn about
missing clinical markers. They never invent clinical content.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from models.protocol_models import ProtocolAnalysis
from services.source_verifier import normalize_text


# Reject placeholder / generic metadata even if a short substring matches.
_GENERIC_TITLES = {
    "example trial",
    "example study",
    "test trial",
    "test study",
    "untitled",
    "protocol title",
    "clinical trial",
    "study",
    "trial",
}

_GENERIC_IDS = {
    "example-protocol-123",
    "example",
    "n/a",
    "na",
    "unknown",
    "test",
    "protocol id",
}

_MIN_TITLE_CHARS = 12
_MIN_ID_CHARS = 4

_TITLE_LABELS = (
    "protocol title",
    "study title",
    "trial title",
    "title",
)

_ID_LABELS = (
    "protocol id",
    "protocol number",
    "study id",
    "trial id",
)

# Marker label -> substrings to search in eligibility-related source text.
_CLINICAL_MARKERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("HbA1c", ("hba1c", "hb a1c")),
    ("eGFR", ("egfr",)),
)

_ITEM_FIELDS = {
    "inclusion_criteria": "description",
    "exclusion_criteria": "description",
    "procedures": "name",
    "visits": "visit_name",
    "safety_information": "description",
}


def _page_texts(pdf_result: dict) -> dict[int, str]:
    return {
        page["page_number"]: normalize_text(page.get("text") or "")
        for page in pdf_result.get("pages", [])
    }


def _combined_protocol_text(pdf_result: dict, protocol_text: str = "") -> str:
    if protocol_text and protocol_text.strip():
        return normalize_text(protocol_text)
    parts = [page.get("text") or "" for page in pdf_result.get("pages", [])]
    return normalize_text("\n".join(parts))


def _labeled_values(protocol_text: str, labels: Iterable[str]) -> list[str]:
    """Return values following labels such as 'protocol title:' (line-based)."""
    values: list[str] = []
    for raw_line in protocol_text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        for label in labels:
            prefix = f"{normalize_text(label)}:"
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if value:
                    values.append(value)
                break
    return values


def _is_generic_metadata(value: str, generics: set[str]) -> bool:
    return normalize_text(value) in generics


def is_grounded_trial_title(value: Optional[str], protocol_text: str) -> bool:
    """Return True when title is explicitly supported by source text."""
    if not value or not value.strip():
        return False

    normalized_value = normalize_text(value)
    if _is_generic_metadata(normalized_value, _GENERIC_TITLES):
        return False

    labeled = _labeled_values(protocol_text, _TITLE_LABELS)
    if any(
        normalized_value == item or normalized_value in item
        for item in labeled
    ):
        return True

    # Unlabeled substring matches must be long enough to avoid coincidence.
    if len(normalized_value) < _MIN_TITLE_CHARS:
        return False
    return normalized_value in normalize_text(protocol_text)


def is_grounded_protocol_id(value: Optional[str], protocol_text: str) -> bool:
    """Return True when protocol ID is explicitly supported by source text."""
    if not value or not value.strip():
        return False

    normalized_value = normalize_text(value)
    if _is_generic_metadata(normalized_value, _GENERIC_IDS):
        return False

    labeled = _labeled_values(protocol_text, _ID_LABELS)
    if any(
        normalized_value == item or normalized_value in item
        for item in labeled
    ):
        return True

    if len(normalized_value) < _MIN_ID_CHARS:
        return False
    return normalized_value in normalize_text(protocol_text)


def _original_labeled_value(
    protocol_text: str,
    labels: Iterable[str],
) -> Optional[str]:
    """
    Return the original-cased value for the first matching label line.

    Only explicit 'LABEL: value' lines are accepted. Never invents values.
    """
    for raw_line in protocol_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or ":" not in stripped:
            continue
        normalized_line = normalize_text(stripped)
        for label in labels:
            prefix = f"{normalize_text(label)}:"
            if normalized_line.startswith(prefix):
                value = stripped.split(":", 1)[1].strip()
                if value:
                    return value
    return None


def recover_labeled_metadata(
    analysis: ProtocolAnalysis,
    protocol_text: str,
) -> List[str]:
    """
    Fill null trial_title / protocol_id from explicit labeled source fields.

    Recovered values must pass grounding validation. Never invents metadata.
    """
    warnings: list[str] = []

    if analysis.trial_title is None:
        # Prefer explicit protocol/study/trial title labels over bare "title".
        candidate = _original_labeled_value(
            protocol_text,
            ("protocol title", "study title", "trial title"),
        )
        if candidate and is_grounded_trial_title(candidate, protocol_text):
            analysis.trial_title = candidate
            warnings.append(
                f"trial_title recovered from labeled source field: {candidate!r}."
            )

    if analysis.protocol_id is None:
        candidate = _original_labeled_value(
            protocol_text,
            ("protocol id", "protocol number", "study id", "trial id"),
        )
        if candidate and is_grounded_protocol_id(candidate, protocol_text):
            analysis.protocol_id = candidate
            warnings.append(
                f"protocol_id recovered from labeled source field: {candidate!r}."
            )

    return warnings


def ground_protocol_metadata(
    analysis: ProtocolAnalysis,
    protocol_text: str,
) -> List[str]:
    """
    Null ungrounded trial_title / protocol_id and return accuracy warnings.
    """
    warnings: list[str] = []

    if analysis.trial_title is not None:
        if not is_grounded_trial_title(analysis.trial_title, protocol_text):
            warnings.append(
                "trial_title "
                f"{analysis.trial_title!r} was not grounded in source text "
                "and was set to null."
            )
            analysis.trial_title = None

    if analysis.protocol_id is not None:
        if not is_grounded_protocol_id(analysis.protocol_id, protocol_text):
            warnings.append(
                "protocol_id "
                f"{analysis.protocol_id!r} was not grounded in source text "
                "and was set to null."
            )
            analysis.protocol_id = None

    return warnings


def find_matching_pages(statement: str, page_texts: dict[int, str]) -> list[int]:
    """Return page numbers whose text contains the normalized statement."""
    needle = normalize_text(statement)
    if not needle:
        return []
    return sorted(
        page_number
        for page_number, text in page_texts.items()
        if needle in text
    )


def repair_source_pages(
    analysis: ProtocolAnalysis,
    pdf_result: dict,
) -> List[str]:
    """
    Repair source_page only for unique textual matches.

    - Unique match + missing/wrong page → set the unique page.
    - Ambiguous matches → leave unchanged and warn.
    - No matches → leave unchanged and warn as unsupported.
    """
    warnings: list[str] = []
    page_texts = _page_texts(pdf_result)

    for category, text_field in _ITEM_FIELDS.items():
        for item in getattr(analysis, category):
            statement = getattr(item, text_field)
            matches = find_matching_pages(statement, page_texts)
            current = item.source_page

            if len(matches) == 1:
                unique_page = matches[0]
                if current != unique_page:
                    item.source_page = unique_page
                    if current is None:
                        warnings.append(
                            f"{category}: repaired missing source_page for "
                            f"{statement!r} → page {unique_page}."
                        )
                    else:
                        warnings.append(
                            f"{category}: corrected source_page for "
                            f"{statement!r} from {current} → {unique_page}."
                        )
            elif len(matches) == 0:
                warnings.append(
                    f"{category}: unsupported statement {statement!r} "
                    "was not found on any source page."
                )
            else:
                if current not in matches:
                    warnings.append(
                        f"{category}: ambiguous source pages {matches} for "
                        f"{statement!r}; left source_page={current}."
                    )

    return warnings


def _eligibility_source_text(pdf_result: dict, protocol_text: str) -> str:
    """Collect source text from eligibility-related pages/sections only."""
    eligibility_chunks: list[str] = []

    for page in pdf_result.get("pages", []):
        text = page.get("text") or ""
        normalized = normalize_text(text)
        if "inclusion" in normalized or "exclusion" in normalized:
            eligibility_chunks.append(text)

    if eligibility_chunks:
        return normalize_text("\n".join(eligibility_chunks))

    # Fall back to whole protocol text only when no page headings matched.
    return _combined_protocol_text(pdf_result, protocol_text)


def _extracted_eligibility_text(analysis: ProtocolAnalysis) -> str:
    parts = [
        criterion.description
        for criterion in analysis.inclusion_criteria
    ] + [
        criterion.description
        for criterion in analysis.exclusion_criteria
    ]
    return normalize_text("\n".join(parts))


def check_eligibility_completeness(
    analysis: ProtocolAnalysis,
    pdf_result: dict,
    protocol_text: str = "",
) -> List[str]:
    """
    Warn when important clinical markers appear in eligibility source text
    but are absent from extracted criteria. Does not fabricate criteria.
    """
    warnings: list[str] = []
    source = _eligibility_source_text(pdf_result, protocol_text)
    extracted = _extracted_eligibility_text(analysis)

    if not source:
        return warnings

    for marker_label, variants in _CLINICAL_MARKERS:
        in_source = any(variant in source for variant in variants)
        in_extracted = any(variant in extracted for variant in variants)
        if in_source and not in_extracted:
            warnings.append(
                f"Completeness warning: source eligibility text mentions "
                f"{marker_label}, but no extracted inclusion/exclusion "
                "criterion contains it."
            )

    return warnings


def apply_protocol_accuracy_checks(
    analysis: ProtocolAnalysis,
    pdf_result: dict,
    protocol_text: str = "",
) -> List[str]:
    """
    Run deterministic accuracy checks in place on analysis.

    Returns accuracy warnings. Mutates analysis for grounding and page repair.
    """
    source_text = protocol_text or "\n".join(
        page.get("text") or "" for page in pdf_result.get("pages", [])
    )

    warnings: list[str] = []
    # Ground first so invented LLM metadata becomes null, then recover
    # only from explicit labeled source fields into those nulls.
    warnings.extend(ground_protocol_metadata(analysis, source_text))
    warnings.extend(recover_labeled_metadata(analysis, source_text))
    warnings.extend(repair_source_pages(analysis, pdf_result))
    warnings.extend(
        check_eligibility_completeness(analysis, pdf_result, source_text)
    )
    return warnings
