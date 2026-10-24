
import re
from typing import Optional
from models.protocol_models import ProtocolAnalysis
from services.clinical_text import NUMBER_RE, canonical_number


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for comparison."""
    return re.sub(r"\s+", " ", text).strip().casefold()


# Statuses that mean "found in the authoritative text of the claimed page".
VERIFIED_STATUS = "exact_match"

# Headings after which a page holds review questions or worked examples, not
# protocol requirements. Text from such a heading to the end of the page is
# never accepted as evidence for an extracted protocol fact.
_ILLUSTRATIVE_HEADING = re.compile(
    r"^\W*(?:\d+[.)]?\s*)?(?:"
    r"(?:evaluation|review|sample|practice|quiz|comprehension|discussion|"
    r"self[- ]check|test)\s+questions?"
    r"|questions?\s+for\s+(?:evaluation|review)"
    r"|answer\s+key"
    r"|examples?(?:\s+(?:only|scenarios?))?"
    r"|illustrative\s+examples?"
    r")\W*$",
    re.IGNORECASE,
)

_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_NON_ALNUM = re.compile(r"[^\w±+\-.]+")


def compact(text: str) -> str:
    """Comparison form that ignores spacing, brackets and dash/±/to variants."""
    text = _DASHES.sub("-", normalize_text(text))
    text = text.replace("+/-", "±").replace("+-", "±").replace(" to ", "-")
    return _NON_ALNUM.sub("", text)


def split_illustrative(text: str) -> tuple[str, str]:
    """Return (authoritative text, illustrative text) for one page."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _ILLUSTRATIVE_HEADING.match(line.strip()):
            return "\n".join(lines[:index]), "\n".join(lines[index:])
    return text, ""


def authoritative_page_texts(pdf_result: dict) -> dict[int, str]:
    return {
        page["page_number"]: normalize_text(
            split_illustrative(page.get("text") or "")[0]
        )
        for page in pdf_result.get("pages", [])
    }


def authoritative_page_lines(pdf_result: dict) -> dict[int, list[str]]:
    """Normalized non-empty lines of each page's authoritative text."""
    return {
        page["page_number"]: [
            normalize_text(line)
            for line in split_illustrative(page.get("text") or "")[0].splitlines()
            if line.strip()
        ]
        for page in pdf_result.get("pages", [])
    }


def timing_belongs_to_visit(
    name: str,
    timing: str,
    lines: list[str],
    other_names: list[str] = (),
) -> bool:
    """
    True when the timing is written with this visit, not merely on the same page.

    Accepted: the name and timing on one line, or (table-style wrapping) the
    name on a line with no figures of its own and the timing on the next line,
    provided that next line does not start another visit.
    """
    wanted = compact(timing)
    variants = [compact(v) for v in _name_variants(name) if v]
    others = [compact(o) for o in other_names if o]
    compact_lines = [compact(line) for line in lines]
    for index, line in enumerate(compact_lines):
        hit = next((v for v in variants if v in line), None)
        if hit is None:
            continue
        if wanted in line:
            return True
        following = compact_lines[index + 1] if index + 1 < len(compact_lines) else ""
        if (
            wanted in following
            and not re.search(r"\d", line.replace(hit, "", 1))
            and not any(o in following for o in others)
        ):
            return True
    return False


# Words that carry no clinical claim; negations are deliberately NOT here.
_STOPWORDS = {
    "that", "this", "with", "from", "will", "have", "been", "your", "they",
    "their", "there", "than", "then", "when", "what", "which", "while", "also",
    "into", "such", "each", "must", "should", "would", "could", "about", "these",
    "those", "are", "the", "and", "for", "may", "can",
}
_NEGATIONS = {"no", "not", "never", "without", "non", "cannot", "none"}


def _numbers(text: str) -> set[str]:
    """Numbers as canonical decimals so '.5' never equals '5'."""
    return {canonical_number(m.group(0)) for m in NUMBER_RE.finditer(text)}


_CLAIM_SPLIT = re.compile(r"[;\n]|[.!?](?=\s|$)")


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.casefold())
    return {
        w if w in _NEGATIONS else (w[:-1] if w.endswith("s") and len(w) > 4 else w)[:5]
        for w in words
        if w in _NEGATIONS or (len(w) >= 4 and w not in _STOPWORDS)
    }


def unsupported_claims(text: str, page_text: str) -> list[str]:
    """
    Claims in `text` that the protocol page does not support.

    Every sentence must be matched by one single line (or two wrapped lines)
    of the page: all of its figures, the same negations (a dropped "not" flips
    the meaning), and at least 80% of its
    content words must appear in that same passage. Figures found elsewhere on
    the page therefore do not support a claim.
    """
    lines = [normalize_text(l) for l in page_text.splitlines() if l.strip()]
    segments = lines + [f"{a} {b}" for a, b in zip(lines, lines[1:])]
    segment_data = [
        (_numbers(seg), _content_stems(seg)) for seg in segments
    ]
    problems = []
    checked = 0
    for claim in (c.strip() for c in _CLAIM_SPLIT.split(text)):
        numbers = _numbers(claim)
        stems = _content_stems(claim)
        if not numbers and not stems:
            continue
        checked += 1
        negations = stems & _NEGATIONS
        supported = False
        for seg_numbers, seg_stems in segment_data:
            if not numbers <= seg_numbers or negations != seg_stems & _NEGATIONS:
                continue
            if not stems or len(stems & seg_stems) / len(stems) >= 0.8:
                supported = True
                break
        if not supported:
            problems.append(claim)
    if not checked:
        problems.append(text.strip())
    return problems


def _name_variants(name: str) -> list[str]:
    base = normalize_text(name).rstrip(".")
    variants = [base]
    stripped = re.sub(r"\s+visit$", "", base)
    if stripped and stripped != base:
        variants.append(stripped)
    return variants


def _singular_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[.][0-9]+)?", text.casefold())
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]


def _tokens_in_window(needle: list[str], haystack: list[str]) -> bool:
    """All needle tokens, in order, within a window barely longer than the needle."""
    if not needle:
        return False
    limit = len(needle) + 2
    for start, token in enumerate(haystack):
        if token != needle[0]:
            continue
        pos = 1
        for offset in range(1, min(limit, len(haystack) - start)):
            if pos < len(needle) and haystack[start + offset] == needle[pos]:
                pos += 1
        if pos == len(needle):
            return True
    return False


_WEEK_NAME = re.compile(r"^weeks?\s+(\d+)$")
_WEEK_LIST = re.compile(
    r"weeks?\s+\d+(?:\s*(?:,|and|&)\s*(?:and\s+)?\d+)*", re.IGNORECASE
)


_PLANNED_VISITS = re.compile(r"(?:planned )?visits?:\s*([^.]+)\.", re.IGNORECASE)


def week_list_context(name: str, lines: list[str]) -> Optional[str]:
    """
    Text following a written week list ("Weeks 2, 4, 8 and 12") that names this
    visit, or None. A visit such as "Week 12" is on the page when its number is
    one of the listed weeks, although the words "Week 12" never appear together.
    """
    wanted = normalize_text(name).rstrip(".")
    text = " ".join(lines)
    match = _WEEK_NAME.match(wanted)
    if not match:
        # A visit named in a "Planned visits: A, B, Weeks 2 and 4." sentence.
        for found in _PLANNED_VISITS.finditer(text):
            names = [
                part.strip().rstrip(".")
                for part in re.split(r",|\band\b", normalize_text(found.group(1)))
            ]
            if wanted in names:
                return text[found.start():found.end() + 200]
        return None
    for found in _WEEK_LIST.finditer(text):
        if match.group(1) in re.findall(r"\d+", found.group(0)):
            return text[found.start():found.end() + 200]
    return None


def _on_page(statement: str, page_text: str) -> bool:
    """Exact text match, or the same words in order ignoring plural endings."""
    variants = _name_variants(statement)
    if any(v and v in page_text for v in variants):
        return True
    haystack = _singular_tokens(page_text)
    return any(_tokens_in_window(_singular_tokens(v), haystack) for v in variants if len(v) > 3)


def verify_source_pages(
    analysis: ProtocolAnalysis,
    pdf_result: dict,
) -> list[dict]:
    """
    Check whether each extracted item appears on its claimed PDF page.

    Matching is exact after normalization. Text under review-question or
    example headings does not count. Visits must also show their stated
    timing on the same page. Paraphrases and clinical meaning are not
    verified here.

    Row statuses: exact_match, missing_page, invalid_page, not_verified,
    illustrative_only (found only under an example/question heading) and
    timing_not_found (visit name found, its timing is not).
    """

    full = {
        page["page_number"]: normalize_text(page["text"])
        for page in pdf_result["pages"]
    }
    page_texts = authoritative_page_texts(pdf_result)
    page_lines = authoritative_page_lines(pdf_result)

    results = []

    fields = {
        "inclusion_criteria": "description",
        "exclusion_criteria": "description",
        "procedures": "name",
        "visits": "visit_name",
        "safety_information": "description",
        "patient_information": "description",
    }

    for category, text_field in fields.items():
        for item in getattr(analysis, category):
            statement = getattr(item, text_field)
            source_page = item.source_page
            detail = None

            if source_page is None:
                status = "missing_page"
            elif source_page not in page_texts:
                status = "invalid_page"
            elif _on_page(statement, page_texts[source_page]) or (
                category == "visits"
                and week_list_context(statement, page_lines[source_page])
            ):
                status = VERIFIED_STATUS
                timing = getattr(item, "timing", None)
                listed = (
                    week_list_context(statement, page_lines[source_page])
                    if category == "visits" else None
                )
                if listed is not None:
                    if timing and compact(timing) not in compact(listed):
                        status = "timing_not_found"
                        detail = (
                            f"Timing {timing!r} is not written with the week "
                            f"list that names {statement!r} on page {source_page}."
                        )
                elif category == "visits" and timing:
                    others = [
                        v for other in analysis.visits if other is not item
                        for v in _name_variants(other.visit_name)
                    ]
                    if not timing_belongs_to_visit(
                        statement, timing, page_lines[source_page], others
                    ):
                        status = "timing_not_found"
                        detail = (
                            f"Timing {timing!r} is not written together with "
                            f"{statement!r} on page {source_page} of the protocol."
                        )
            elif _on_page(statement, full[source_page]):
                status = "illustrative_only"
                detail = (
                    f"Found on page {source_page} only under an example or "
                    "review-question heading, not in a protocol section."
                )
            else:
                status = "not_verified"

            row = {
                "category": category,
                "statement": statement,
                "source_page": source_page,
                "status": status,
            }
            if category == "visits":
                row["timing"] = item.timing
            if detail:
                row["detail"] = detail
            results.append(row)

    return results


def verify_saved_pages(analysis: ProtocolAnalysis, source_pages: dict) -> list[dict]:
    """Rerun the source check from the page text saved with a review."""
    pages = [
        {"page_number": int(number), "text": text or ""}
        for number, text in source_pages.items()
        if str(number).isdigit()
    ]
    return verify_source_pages(analysis, {"pages": pages})
