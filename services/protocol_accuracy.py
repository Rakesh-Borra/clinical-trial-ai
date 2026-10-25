
"""
Deterministic post-LLM accuracy checks for protocol analysis.

These helpers ground metadata, repair unique source pages, and warn about
missing clinical markers. They never invent clinical content.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from models.protocol_models import PatientInfoItem, ProtocolAnalysis, SafetyInformation
from services.source_verifier import (
    authoritative_page_lines, authoritative_page_texts, normalize_text,
    week_list_context,
)


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
    "patient_information": "description",
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


def prefer_labeled_title(
    analysis: ProtocolAnalysis,
    protocol_text: str,
) -> List[str]:
    """
    Use the explicit 'PROTOCOL TITLE:' line when the model's title differs.

    A labeled title printed in the source is more reliable than a title the
    model assembled from one chunk, so it wins. Never invents a title.
    """
    candidate = _original_labeled_value(
        protocol_text, ("protocol title", "study title", "trial title")
    )
    if not candidate or not is_grounded_trial_title(candidate, protocol_text):
        return []
    if analysis.trial_title and normalize_text(analysis.trial_title) == normalize_text(candidate):
        return []
    previous = analysis.trial_title
    analysis.trial_title = candidate
    if previous is None:
        return []
    return [
        f"trial_title {previous!r} replaced by the labeled source title "
        f"{candidate!r}."
    ]


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
    # Review questions and examples are not valid evidence for a page.
    page_texts = authoritative_page_texts(pdf_result)

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
            elif len(matches) == 0 and category == "visits" and any(
                week_list_context(statement, lines)
                for lines in authoritative_page_lines(pdf_result).values()
            ):
                continue  # named in the protocol's written visit list
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


_ACTIVITY = re.compile(
    r"\b(blood (?:draw|collection|sample)s?|venous|venipuncture|sample collection)\b",
    re.IGNORECASE,
)
_TIME_TOKEN = re.compile(r"\b(day|week)\s*(-?\d+)", re.IGNORECASE)


def _time_tokens(*texts: Optional[str]) -> set[str]:
    return {
        f"{m.group(1).lower()} {m.group(2)}"
        for t in texts if t for m in _TIME_TOKEN.finditer(t)
    }


def _visit_key(name: str) -> str:
    return re.sub(r"\s+visit$", "", normalize_text(name).rstrip("."))


def link_visit_activities(analysis: ProtocolAnalysis, pdf_result: dict) -> List[str]:
    """
    Fold "visits" that are really a blood draw at an existing visit into it.

    A model sometimes lists "Day 1 venous blood draw" as a visit of its own.
    It is only linked when its timing equals exactly one other scheduled
    visit and its wording is on its cited protocol page; otherwise it stays
    as extracted and keeps failing source verification.
    """
    pages = authoritative_page_texts(pdf_result)
    warnings: List[str] = []
    visits = list(analysis.visits)
    for activity in list(visits):
        if not _ACTIVITY.search(activity.visit_name):
            continue
        tokens = _time_tokens(activity.visit_name, activity.timing)
        if not tokens:
            continue
        hosts = [
            v for v in visits
            if v is not activity
            and not _ACTIVITY.search(v.visit_name)
            and tokens & _time_tokens(v.visit_name, v.timing)
        ]
        page_text = pages.get(activity.source_page or -1, "")
        if len(hosts) != 1 or not _on_page_loose(activity.visit_name, page_text):
            continue
        host = hosts[0]
        label = normalize_text(activity.visit_name)
        if label not in [normalize_text(p) for p in host.procedures]:
            host.procedures.append(activity.visit_name)
        visits.remove(activity)
        warnings.append(
            f'"{activity.visit_name}" is a procedure at the "{host.visit_name}" '
            "visit, not a separate visit; it was linked to that visit."
        )
    analysis.visits = visits
    return warnings


_LABELED_SECTIONS = (
    ("study_purpose", re.compile(
        r"(?:study purpose|purpose of the study|purpose|study objectives?|objectives?)",
        re.IGNORECASE)),
    ("consent_withdrawal", re.compile(
        r"(?:informed consent|consent|voluntary participation|participant withdrawal|"
        r"withdrawal|withdrawing from the study|right to withdraw|"
        r"privacy and confidentiality|confidentiality)", re.IGNORECASE)),
    ("safety_information", re.compile(
        r"(?:safety monitoring|safety and monitoring|adverse event reporting)",
        re.IGNORECASE)),
)


def recover_labeled_sections(analysis: ProtocolAnalysis, pdf_result: dict) -> List[str]:
    """
    Add protocol sentences written under a heading for study purpose, consent,
    withdrawal or safety monitoring when extraction left them out.

    Only verbatim sentences from the authoritative text of the cited page are
    added, each with its own page; nothing is paraphrased or inferred. A page
    title such as "016 Participant withdrawal" with no text under it adds
    nothing.
    """
    from services.source_verifier import split_illustrative

    def key(text: str) -> str:
        return normalize_text(text).rstrip(". ")

    # Statements the model already extracted; repeats of a new sentence on
    # other pages are all kept so every page citation survives de-duplication.
    known = {key(i.description) for i in analysis.patient_information} | {
        key(s.description) for s in analysis.safety_information
    }
    added: dict = {}
    for page in pdf_result.get("pages", []):
        lines = split_illustrative(page.get("text") or "")[0].splitlines()
        for index, line in enumerate(lines):
            heading = re.sub(r"^\d{1,3}\s+", "", line.strip()).rstrip(":").strip()
            category = next(
                (c for c, pattern in _LABELED_SECTIONS if pattern.fullmatch(heading)),
                None,
            )
            if category is None:
                continue
            block = []
            for follow in lines[index + 1:]:
                if not follow.strip():
                    break
                block.append(follow.strip())
            for sentence in re.split(r"(?<=[.!?])\s+", " ".join(block)):
                sentence = sentence.strip()
                if len(sentence.split()) < 3 or key(sentence) in known:
                    continue
                added.setdefault(category, []).append((sentence, page["page_number"]))
    warnings: List[str] = []
    for category, entries in added.items():
        for sentence, page_number in entries:
            if category == "safety_information":
                analysis.safety_information.append(
                    SafetyInformation(description=sentence, source_page=page_number)
                )
            else:
                analysis.patient_information.append(
                    PatientInfoItem(category=category, description=sentence,
                                    source_page=page_number)
                )
        warnings.append(
            f"{len(entries)} {category.replace('_', ' ')} statement(s) were recovered "
            "from labelled protocol sections the model left out."
        )
    return warnings


_SINGLE_WEEK_PLURAL = re.compile(r"^Weeks(\s+\d+)$")
_DATE_SENTENCE = re.compile(r"visit dates?\s+(?:are|is)\s+([^.]+)\.", re.IGNORECASE)


def normalize_visit_labels(analysis: ProtocolAnalysis, pdf_result: dict) -> List[str]:
    """
    Fix labels split out of a written week list, without inventing visits.

    "Weeks 2, 4, 8" becomes the visits "Week 2", "Week 4", "Week 8". A visit
    whose timing is only the first word of the protocol's own sentence
    ("Visit dates are approximate and confirmed by the study coordinator.")
    gets the sentence's full wording, so coordinator confirmation is kept.
    """
    pages = {
        page["page_number"]: re.sub(r"\s+", " ", page.get("text") or "")
        for page in pdf_result.get("pages", [])
    }
    warnings: List[str] = []
    for visit in analysis.visits:
        label = visit.visit_name.strip()
        fixed = _SINGLE_WEEK_PLURAL.sub(r"Week\1", label)
        if fixed != label:
            visit.visit_name = fixed
        timing = (visit.timing or "").strip().rstrip(".")
        if not timing:
            continue
        sentence = _DATE_SENTENCE.search(pages.get(visit.source_page, ""))
        if not sentence:
            continue
        full = sentence.group(1).strip()
        if (
            len(full) > len(timing)
            and full.casefold().startswith(timing.casefold())
        ):
            visit.timing = full
            warnings.append(
                f'Timing of "{visit.visit_name}" completed from the protocol '
                f'sentence "{sentence.group(0).strip()}" on page {visit.source_page}.'
            )
    return warnings


def dedupe_visits(analysis: ProtocolAnalysis) -> List[str]:
    """Merge visits that repeat the same name and timing (or timing and name)."""
    kept = []
    warnings: List[str] = []
    for visit in analysis.visits:
        key = _visit_key(visit.visit_name)
        timing = normalize_text(visit.timing or "")
        twin = next(
            (
                k for k in kept
                if _visit_key(k.visit_name) == key
                and normalize_text(k.timing or "") == timing
            ),
            None,
        )
        if twin is None:
            kept.append(visit)
            continue
        for procedure in visit.procedures:
            if normalize_text(procedure) not in [normalize_text(p) for p in twin.procedures]:
                twin.procedures.append(procedure)
        warnings.append(f'Duplicate visit "{visit.visit_name}" was merged into one.')
    analysis.visits = kept
    return warnings


def _on_page_loose(text: str, page_text: str) -> bool:
    from services.source_verifier import _on_page

    return bool(page_text) and _on_page(text, normalize_text(page_text))


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
    warnings.extend(prefer_labeled_title(analysis, source_text))
    warnings.extend(ground_protocol_metadata(analysis, source_text))
    warnings.extend(recover_labeled_metadata(analysis, source_text))
    warnings.extend(repair_source_pages(analysis, pdf_result))
    warnings.extend(recover_labeled_sections(analysis, pdf_result))
    warnings.extend(normalize_visit_labels(analysis, pdf_result))
    warnings.extend(dedupe_visits(analysis))
    warnings.extend(link_visit_activities(analysis, pdf_result))
    warnings.extend(
        check_eligibility_completeness(analysis, pdf_result, source_text)
    )
    return warnings
