"""
Patient-relevance filter applied to protocol pages before LLM analysis.

The goal is fewer model calls without losing anything a participant needs. The
filter is deliberately conservative and uses no model calls:

- A page is skipped only when it looks clearly administrative (sponsor duties,
  monitoring, data management, record keeping, ...) AND it contains no
  patient-relevant terms AND no safety-critical terms AND no patient-facing
  section heading.
- Anything uncertain is kept.
- Pages that continue a patient-facing section (for example a long exclusion
  list) are kept until an administrative heading starts.
- The first page is always kept because it carries the title and protocol id.
- If filtering would remove every page, nothing is removed.

Skipped pages are reported (page number and reason) and the original protocol
text and page numbers are never modified, so citations and source
verification still use the complete document.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from services.protocol_chunker import detect_sections
from services.section_classifier import (
    classify_heading,
    classify_text,
    is_administrative_only,
)

# Terms that mean the page may affect a participant.
_PATIENT_TERMS = re.compile(
    r"\b(?:eligib\w*|inclusion|exclusion|criteria|consent|withdraw\w*|"
    r"discontinu\w*|visits?|schedule[sd]?|screening|baseline|follow[- ]?up|"
    r"procedures?|blood|urine|sample|biopsy|scan|imaging|ecg|fasting|"
    r"dose|dosing|dosage|treatment|placebo|randomi[sz]\w*|injection|"
    r"infusion|tablets?|diary|questionnaires?|weeks?|months?|"
    r"benefits?|risks?|side effects?|adverse|pregnan\w*|contracepti\w*|"
    r"breastfeed\w*|compensation|reimburse\w*|injury|cost|"
    r"purpose|objectives?|aims?|hospital\w*|emergency)\b",
    re.IGNORECASE,
)

# Safety-critical and participant-protection wording is never filtered,
# wherever it appears.
_SAFETY_TERMS = re.compile(
    r"\b(?:adverse|side effects?|toxicit\w*|contraindicat\w*|"
    r"life[- ]threatening|death|died|fatal|hospitali[sz]\w*|overdose|"
    r"allerg\w*|anaphyla\w*|hypoglyc\w*|hyperglyc\w*|pregnan\w*|"
    r"contracepti\w*|breastfeed\w*|stopping rules?|unblind\w*|"
    r"emergency|insurance|compensation|injur\w*|reimburse\w*|"
    r"informed consent|right to withdraw|withdraw(?:al)? from the study)\b",
    re.IGNORECASE,
)

_ADMIN_TERMS = re.compile(
    r"\b(?:sponsor|monitor(?:ing)? visits?|monitors?|source documents?|"
    r"case report forms?|crfs?|data management|data entry|database lock|"
    r"statistical analysis plan|sample size|investigator'?s? brochure|"
    r"regulatory (?:submission|authorit\w*|documents?)|delegation log|"
    r"staff training|site (?:qualification|initiation|personnel|staff)|"
    r"record retention|retention of records|audits?|inspections?|"
    r"publication policy|financial disclosure|document history|"
    r"version history|signature page|investigator responsibilit\w*|"
    r"drug accountability|supply (?:chain|management)|archiv\w*|"
    r"quality assurance|quality control|standard operating procedures?)\b",
    re.IGNORECASE,
)

_ADMIN_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:sponsor\b.*|study (?:staff|administration|"
    r"management)\b.*|monitoring\b.*|data management\b.*|"
    r"statistical (?:considerations|analysis)\b.*|investigator "
    r"responsibilities\b.*|regulatory\b.*|record (?:keeping|retention)\b.*|"
    r"publication policy|references|bibliography|abbreviations|"
    r"table of contents|signature page|document history|"
    r"protocol amendments?|quality (?:assurance|control)\b.*|"
    r"drug (?:supply|accountability)\b.*|appendix [a-z0-9]+[:.\- ]+"
    r"(?:regulatory|administrative|staff)\b.*)$",
    re.IGNORECASE,
)

_PATIENT_SECTIONS = {"eligibility", "exclusions", "safety", "visits", "procedures"}


@dataclass(frozen=True)
class PageDecision:
    page_number: Any
    keep: bool
    reason: str


@dataclass
class RelevanceResult:
    enabled: bool
    kept_pages: List[Mapping[str, Any]]
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    total_pages: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_pages": self.total_pages,
            "analyzed_pages": len(self.kept_pages),
            "skipped_pages": list(self.skipped),
        }


def filter_enabled() -> bool:
    return os.getenv("CTAI_PATIENT_RELEVANCE_FILTER", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }


def _has_admin_heading(text: str) -> bool:
    return any(
        _ADMIN_HEADING.match(line.strip().rstrip(":"))
        for line in text.splitlines()
        if 0 < len(line.strip()) <= 80
    )


def classify_pages(pages: Sequence[Mapping[str, Any]]) -> List[PageDecision]:
    """Decide keep/skip for each readable page, in order."""
    decisions: List[PageDecision] = []
    in_patient_section = False
    first_seen = False
    for page in pages:
        text = page.get("text") or ""
        number = page.get("page_number")
        if not text.strip():
            continue
        sections = set(detect_sections(text))
        admin_heading = _has_admin_heading(text)

        keep, reason = True, "kept"
        if not first_seen:
            first_seen = True
            reason = "first page (title and protocol id)"
        elif sections & _PATIENT_SECTIONS:
            reason = "patient-facing section heading"
        elif _SAFETY_TERMS.search(text):
            reason = "contains safety-related wording"
        elif in_patient_section and not admin_heading:
            reason = "continues a patient-facing section"
        else:
            patient_hits = len(_PATIENT_TERMS.findall(text))
            admin_hits = len(_ADMIN_TERMS.findall(text)) + (3 if admin_heading else 0)
            if admin_hits >= 3 and admin_hits >= 2 * patient_hits:
                keep, reason = False, "administrative content with no patient-relevant terms"
            elif patient_hits:
                reason = "patient-relevant terms"
            else:
                reason = "uncertain; kept"

        if sections & _PATIENT_SECTIONS:
            in_patient_section = True
        elif admin_heading:
            in_patient_section = False
        decisions.append(PageDecision(number, keep, reason))
    return decisions


def filter_protocol_pages(pages: Sequence[Mapping[str, Any]]) -> RelevanceResult:
    """Return the pages to analyze plus a record of anything skipped."""
    readable = [p for p in pages if (p.get("text") or "").strip()]
    if not filter_enabled():
        return RelevanceResult(False, list(pages), total_pages=len(readable))

    decisions = classify_pages(pages)
    skipped_numbers = {d.page_number for d in decisions if not d.keep}
    if not skipped_numbers or len(skipped_numbers) == len(decisions):
        return RelevanceResult(True, list(pages), total_pages=len(readable))

    return RelevanceResult(
        True,
        [p for p in pages if p.get("page_number") not in skipped_numbers],
        skipped=[
            {"page": d.page_number, "reason": d.reason}
            for d in decisions
            if not d.keep
        ],
        total_pages=len(readable),
    )


# ---------------------------------------------------------------------------
# Section-level filtering (blocks inside pages)
# ---------------------------------------------------------------------------
#
# Pages in real protocols mix patient-facing and administrative sections, so
# page-level filtering removes very little. Blocks are heading-delimited
# pieces of a page. A block is skipped only when its heading is clearly
# administrative, it holds no safety-critical wording, and it is not a
# sub-section of a patient-facing section. Block text is copied verbatim from
# the page, so source verification against the original page still works.

_NUM_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.*)$")

_PATIENT_HEADING = re.compile(
    r"purpose|objective|\baims?\b|background|rationale|study design|overview|"
    r"synopsis|summary|eligib|inclusion|exclusion|criteria|visit|schedule|"
    r"procedure|assessment|treatment|dosing|\bdose\b|risk|benefit|side effect|"
    r"adverse|safety|consent|withdraw|discontinu|responsibilit|amendment|"
    r"participant|subject|patient|follow[- ]?up|duration|compensation|injury|"
    r"confidential|population|blinding|randomi[sz]",
    re.IGNORECASE,
)

_PATIENT_ONLY_HEADING = re.compile(
    r"participant|subject|patient|safety|consent|eligib|risk|withdraw",
    re.IGNORECASE,
)

_ADMIN_BLOCK_HEADING = re.compile(
    r"sponsor|monitoring|data management|data handling|statistic|sample size|"
    r"record|archiv|audit|inspection|regulatory|publication|property|"
    r"intellectual|financial|conflict of interest|quality (?:assurance|control)|"
    r"references|bibliography|abbreviations|table of contents|signature|"
    r"document history|version history|drug (?:supply|accountability)|"
    r"\bimp\b (?:management|supply)|logistics|staff|training|delegation|"
    r"site (?:selection|initiation|closure|personnel)|investigator.s? brochure",
    re.IGNORECASE,
)

_STRONG_PATIENT = re.compile(
    r"\b(?:inclusion|exclusion|eligib\w*|visits?|screening|blood|urine|"
    r"procedures?|participants? (?:will|must)|subjects? (?:will|must))\b",
    re.IGNORECASE,
)


@dataclass
class Block:
    """A heading-delimited piece of one page, copied verbatim."""

    page_number: Any
    heading: str
    depth: int
    text: str
    keep: bool = True
    reason: str = "kept"
    categories: frozenset = frozenset()
    category_source: str = "none"  # "heading", "content" (uncertain) or "none"
    is_table: bool = False


@dataclass
class BlockFilterResult:
    enabled: bool
    blocks: List[Block]
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    total_blocks: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_blocks": self.total_blocks,
            "analyzed_blocks": len([b for b in self.blocks if b.keep]),
            "skipped_blocks": list(self.skipped),
        }


_MINOR_WORDS = frozenset({
    "of", "and", "the", "from", "for", "in", "to", "a", "an", "with", "or",
    "on", "by", "as", "at",
})


def parse_heading(line: str) -> Optional[tuple[int, str]]:
    """Return (depth, title) when the line looks like a section heading."""
    text = line.strip().rstrip(":").strip()
    if len(text) < 3 or text[-1] in ".,;?":
        return None
    depth, title = 0, text
    match = _NUM_PREFIX.match(text)
    if match:
        depth, title = match.group(1).count(".") + 1, match.group(2).strip()
    known = bool(classify_heading(title))
    # Long M11-style headings are accepted only when numbered or recognized.
    max_chars, max_words = (200, 25) if (depth or known) else (100, 10)
    words = title.split()
    if not words or len(text) > max_chars or len(words) > max_words:
        return None
    if any(c.isdigit() for c in title) and not known:
        return None
    letters = [c for c in title if c.isalpha()]
    if len(letters) < 3:
        return None
    if sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        return depth, title
    long_words = [
        w for w in words
        if len(w) > 3 and w[0].isalpha() and w.lower() not in _MINOR_WORDS
    ]
    if long_words and all(w[0].isupper() for w in long_words) and (
        depth or known or len(words) <= 6
    ):
        return depth, title
    return None


def _is_admin_heading(block: Block) -> bool:
    return bool(block.heading) and (
        bool(_ADMIN_BLOCK_HEADING.search(block.heading))
        or (
            block.category_source == "heading"
            and is_administrative_only(block.categories)
        )
    )


def _trim_to_safety_paragraphs(block: Block) -> bool:
    """
    In an administrative section, keep only the heading and the paragraphs
    with safety-related wording (verbatim). Sections without paragraph breaks
    are kept whole, because their boundaries cannot be told apart safely.
    """
    heading_line, _, body = block.text.partition("\n")
    paragraphs = re.split(r"\n\s*\n", body)
    if len(paragraphs) < 2:
        return False
    kept = [heading_line] + [p for p in paragraphs if _SAFETY_TERMS.search(p)]
    if len(kept) == len(paragraphs) + 1:
        return False
    block.text = "\n\n".join(kept)
    return True


def _classify_block(block: Block) -> None:
    """Heading first; fall back to (uncertain) content cues."""
    block.categories = classify_heading(block.heading)
    if block.categories:
        block.category_source = "heading"
        return
    cues = classify_text(block.text)
    if cues:
        block.categories, block.category_source = cues, "content"


def render_table(rows: Sequence[Sequence[Any]]) -> str:
    """Render table rows as 'cell | cell' lines; cell text is kept verbatim."""
    lines = []
    for row in rows:
        cells = [" ".join(str(c).split()) if c is not None else "" for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def table_blocks(page: Mapping[str, Any]) -> List[Block]:
    """One block per table the PDF extractor found on this page."""
    blocks = []
    for table in page.get("tables") or []:
        body = render_table(table.get("rows") or [])
        if body:
            blocks.append(Block(
                page.get("page_number"), "Table", 0,
                f"[TABLE]\n{body}\n[/TABLE]", is_table=True,
            ))
    return blocks


def split_page_blocks(page_number: Any, text: str) -> List[Block]:
    """Split a page into blocks at heading lines, preserving every character."""
    blocks: List[Block] = []
    heading, depth, lines = "", 0, []

    def flush() -> None:
        body = "\n".join(lines).strip("\n")
        if body.strip():
            blocks.append(Block(page_number, heading, depth, body))

    for line in text.splitlines():
        parsed = parse_heading(line)
        if parsed is not None:
            flush()
            heading, depth, lines = parsed[1], parsed[0], [line]
        else:
            lines.append(line)
    flush()
    return blocks


def filter_protocol_blocks(
    pages: Sequence[Mapping[str, Any]],
) -> BlockFilterResult:
    """Split pages into blocks and drop only clearly administrative ones."""
    blocks: List[Block] = []
    first_page = True
    for page in pages:
        text = page.get("text") or ""
        if not text.strip():
            # Keep the marker so unreadable pages stay visible to the model.
            blocks.append(Block(page.get("page_number"), "", 0, "",
                                reason="no extractable text"))
            continue
        page_blocks = split_page_blocks(page.get("page_number"), text)
        page_blocks.extend(table_blocks(page))
        if first_page:
            for block in page_blocks:
                block.reason = "first page (title and protocol id)"
            first_page = False
            blocks.extend(page_blocks)
            continue
        blocks.extend(page_blocks)

    for block in blocks:
        _classify_block(block)

    if not filter_enabled():
        return BlockFilterResult(False, blocks, total_blocks=len(blocks))

    section_depth: Optional[int] = None
    previous_kept = True
    skipped: List[Dict[str, Any]] = []
    for block in blocks:
        if block.reason.startswith("first page"):
            previous_kept = True
            continue
        text = block.text
        strong = len(_STRONG_PATIENT.findall(text))
        patient_categories = block.categories - {"administrative"}
        if block.is_table:
            block.reason = "table kept verbatim"
        elif _SAFETY_TERMS.search(text):
            block.reason = "contains safety-related wording"
            if _is_admin_heading(block) and _trim_to_safety_paragraphs(block):
                block.reason = "administrative section trimmed to safety paragraphs"
        elif block.category_source == "heading" and patient_categories:
            section_depth = block.depth or 1
            block.reason = "patient-facing section"
        elif (
            block.heading
            and _PATIENT_HEADING.search(block.heading)
            and not (
                _ADMIN_BLOCK_HEADING.search(block.heading)
                and not _PATIENT_ONLY_HEADING.search(block.heading)
            )
        ):
            section_depth = block.depth or 1
            block.reason = "patient-facing heading"
        elif block.heading and (
            _ADMIN_BLOCK_HEADING.search(block.heading)
            or (
                block.category_source == "heading"
                and is_administrative_only(block.categories)
            )
        ):
            inside_patient_section = (
                section_depth is not None and (block.depth or 1) > section_depth
            )
            if strong >= 3 or inside_patient_section:
                block.reason = "uncertain; kept"
            else:
                block.keep = False
                block.reason = "administrative heading, no patient-relevant content"
                section_depth = None
        elif not block.heading and not previous_kept and strong < 3:
            block.keep = False
            block.reason = "continues an administrative section"
        previous_kept = block.keep
        if not block.keep:
            skipped.append({
                "page": block.page_number,
                "heading": block.heading,
                "reason": block.reason,
                "characters": len(block.text),
            })

    if blocks and not any(b.keep for b in blocks):
        for block in blocks:
            block.keep, block.reason = True, "kept; nothing else remained"
        skipped = []
    return BlockFilterResult(True, blocks, skipped, total_blocks=len(blocks))
