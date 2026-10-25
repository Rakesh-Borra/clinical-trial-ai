"""Deterministic page-aware chunking for extracted clinical protocols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_PAGE_MARKER = re.compile(
    r"^[ \t]*\[SOURCE PAGE ([1-9]\d*)\][ \t]*(?:\r?\n|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_PAGE_MARKER_PREFIX = re.compile(r"^[ \t]*\[SOURCE PAGE\b", re.IGNORECASE)
_PAGE_MARKER_LINE = re.compile(
    r"^[ \t]*\[SOURCE PAGE ([1-9]\d*)\][ \t]*$",
    flags=re.IGNORECASE,
)
_HEADING_PREFIX = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")
_PROTECTED_CLINICAL_VALUE = re.compile(
    r"""
    (?<!\w)
    (?:(?:[<>]=?|≥|≤)\s*)?
    \d+(?:[.,]\d+)*
    (?:\s+(?:to|through|[-–—])\s+\d+(?:[.,]\d+)*(?:\s*%?)?)?
    (?:\s*(?:%|mmol/mol|mmol/L|mg/dL|mL/min(?:/1\.73m2)?|
    mg/kg|mcg/kg|µg/kg|mL/kg|mg|mcg|µg|mL|kg/m2|kg|years?|months?|weeks?|days?))?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_SECTION_PATTERNS = (
    (
        "eligibility",
        re.compile(
            r"(?:inclusion criteria|eligibility criteria|"
            r"eligibility requirements|who may participate|inclusion)",
            re.IGNORECASE,
        ),
    ),
    (
        "exclusions",
        re.compile(
            r"(?:exclusion criteria|exclusion|excluded participants|"
            r"who cannot participate)",
            re.IGNORECASE,
        ),
    ),
    (
        "safety",
        re.compile(
            r"(?:safety information|safety|risks? to (?:subjects|"
            r"participants)|adverse events?)",
            re.IGNORECASE,
        ),
    ),
    (
        "visits",
        re.compile(
            r"(?:visit schedule|schedule of assessments|study calendar|"
            r"visits)",
            re.IGNORECASE,
        ),
    ),
    (
        "procedures",
        re.compile(
            r"(?:study procedures|procedures|study assessments|"
            r"assessments)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class ProtocolPage:
    """Extracted text associated with an original PDF page, if known."""

    page_number: Optional[int]
    text: str
    sections: tuple[str, ...]
    continuation: bool = False
    overlap: bool = False


@dataclass(frozen=True)
class ProtocolChunk:
    """One bounded chunk; page fragments retain their original page numbers."""

    index: int
    pages: tuple[ProtocolPage, ...]
    text: str
    sections: tuple[str, ...]
    # Section categories of the blocks in this chunk (empty when unknown), and
    # whether every block was classified from its heading (not just content).
    categories: tuple[str, ...] = ()
    fully_classified: bool = False
    has_table: bool = False
    # Stable ids of the kept source blocks packed into this chunk (empty when
    # chunks were built from raw page text rather than filtered blocks).
    source_block_ids: tuple[int, ...] = ()

    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class ChunkingResult:
    """Chunks and non-fatal input limitations detected during chunking."""

    chunks: tuple[ProtocolChunk, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PageAtom:
    page_index: int
    page_number: Optional[int]
    text: str
    sections: tuple[str, ...]
    continuation: bool

    @property
    def rendered(self) -> str:
        if self.page_number is None:
            return self.text
        return f"[SOURCE PAGE {self.page_number}]\n{self.text}"


def detect_sections(page_text: str) -> tuple[str, ...]:
    """Recognize section labels on heading-shaped lines only."""
    headings = []
    for line in page_text.splitlines():
        heading = line.strip()
        heading = _HEADING_PREFIX.sub("", heading, count=1)
        heading = heading.rstrip(":").strip()
        if heading:
            headings.append(heading)

    return tuple(
        category
        for category, pattern in _SECTION_PATTERNS
        if any(pattern.fullmatch(heading) for heading in headings)
    )


def _parse_pages(protocol_text: str) -> tuple[list[ProtocolPage], list[str]]:
    markers = list(_PAGE_MARKER.finditer(protocol_text))
    warnings: list[str] = []
    lines = protocol_text.splitlines()

    for line_number, line in enumerate(lines, start=1):
        if (
            _PAGE_MARKER_PREFIX.match(line)
            and not _PAGE_MARKER_LINE.fullmatch(line)
        ):
            warnings.append(
                f"Malformed source-page marker on input line {line_number}; "
                "the line was retained as page text."
            )

    if not markers:
        if protocol_text:
            warnings.append(
                "No valid source-page markers found; input was retained as "
                "one page with an unknown page number."
            )
            return [
                ProtocolPage(
                    page_number=None,
                    text=protocol_text,
                    sections=detect_sections(protocol_text),
                )
            ], warnings
        return [], warnings

    pages: list[ProtocolPage] = []
    preamble = protocol_text[:markers[0].start()]
    if preamble:
        pages.append(
            ProtocolPage(
                page_number=None,
                text=preamble,
                sections=detect_sections(preamble),
            )
        )

    for index, marker in enumerate(markers):
        body_start = marker.end()
        body_end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(protocol_text)
        )
        body = protocol_text[body_start:body_end]
        pages.append(
            ProtocolPage(
                page_number=int(marker.group(1)),
                text=body,
                sections=detect_sections(body),
            )
        )

    return pages, warnings


def _page_prefix(page_number: Optional[int]) -> str:
    return f"[SOURCE PAGE {page_number}]\n" if page_number is not None else ""


def _safe_split_end(text: str, start: int, limit: int) -> int:
    """Prefer paragraph/line boundaries, then whitespace outside values."""
    hard_limit = min(start + limit, len(text))
    if hard_limit == len(text):
        return hard_limit

    segment = text[start:hard_limit]
    protected_spans = [
        (match.start(), match.end())
        for match in _PROTECTED_CLINICAL_VALUE.finditer(text)
        if match.start() < hard_limit and match.end() > start
    ]

    def safe(boundary: int) -> bool:
        return not any(left < boundary < right for left, right in protected_spans)

    paragraph_boundaries = [
        start + match.end()
        for match in re.finditer(r"\n[ \t]*\n", segment)
        if match.end() > 1
    ]
    safe_paragraphs = [boundary for boundary in paragraph_boundaries if safe(boundary)]
    if safe_paragraphs:
        return safe_paragraphs[-1]

    line_boundaries = [
        start + match.end()
        for match in re.finditer(r"\n", segment)
        if match.end() > 1
    ]
    safe_lines = [boundary for boundary in line_boundaries if safe(boundary)]
    if safe_lines:
        return safe_lines[-1]

    whitespace_boundaries = [
        index + 1
        for index in range(start, hard_limit)
        if text[index].isspace()
    ]
    safe_boundaries = [boundary for boundary in whitespace_boundaries if safe(boundary)]
    if safe_boundaries:
        return safe_boundaries[-1]

    # A single token/value can exceed the configured size. Preserve every
    # character and let continuation metadata identify the hard split.
    return hard_limit


def _split_pages(
    pages: list[ProtocolPage],
    max_chars: int,
) -> list[_PageAtom]:
    atoms: list[_PageAtom] = []
    for page_index, page in enumerate(pages):
        prefix = _page_prefix(page.page_number)
        body_limit = max_chars - len(prefix)
        if body_limit <= 0:
            raise ValueError(
                "max_chars must be larger than the source-page marker."
            )

        if not page.text:
            atoms.append(
                _PageAtom(
                    page_index=page_index,
                    page_number=page.page_number,
                    text="",
                    sections=page.sections,
                    continuation=False,
                )
            )
            continue

        start = 0
        while start < len(page.text):
            end = _safe_split_end(page.text, start, body_limit)
            atoms.append(
                _PageAtom(
                    page_index=page_index,
                    page_number=page.page_number,
                    text=page.text[start:end],
                    sections=page.sections,
                    continuation=start > 0,
                )
            )
            start = end
    return atoms


def _render_atoms(atoms: list[_PageAtom]) -> str:
    return "\n\n".join(atom.rendered for atom in atoms)


def _page_groups(atoms: list[_PageAtom]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    while start < len(atoms):
        page_index = atoms[start].page_index
        end = start + 1
        while end < len(atoms) and atoms[end].page_index == page_index:
            end += 1
        groups.append((start, end))
        start = end
    return groups


def _select_overlap_groups(
    atoms: list[_PageAtom],
    groups: list[tuple[int, int]],
    previous_bounds: tuple[int, int],
    cursor: int,
    overlap_pages: int,
    max_chars: int,
) -> tuple[list[_PageAtom], bool]:
    prev_start, prev_end = previous_bounds
    prior_groups = [
        (start, end)
        for start, end in groups
        if prev_start <= start and end <= prev_end and end <= cursor
    ]

    selected: list[tuple[int, int]] = []
    overlap_atoms: list[_PageAtom] = []
    for group in reversed(prior_groups):
        candidate = [
            _PageAtom(
                page_index=atoms[index].page_index,
                page_number=atoms[index].page_number,
                text=atoms[index].text,
                sections=atoms[index].sections,
                continuation=atoms[index].continuation,
            )
            for index in range(*group)
        ]
        if len(_render_atoms(candidate + overlap_atoms)) >= max_chars:
            break
        selected.insert(0, group)
        overlap_atoms = candidate + overlap_atoms
        if len(selected) >= overlap_pages:
            break

    unavailable = len(selected) < overlap_pages
    return overlap_atoms, unavailable


def chunk_protocol(
    protocol_text: str,
    *,
    max_chars: int = 12_000,
    overlap_pages: int = 0,
) -> ChunkingResult:
    """
    Split page-marked text into bounded, consecutive chunks.

    Oversized pages are split at deterministic character boundaries and keep
    the original page number on every fragment. Optional overlap duplicates
    complete pages from the preceding chunk only when they fit without
    exceeding ``max_chars``. Headings annotate chunks; they never select or
    discard pages.
    """
    if not isinstance(protocol_text, str):
        raise TypeError("protocol_text must be a string.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer.")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if isinstance(overlap_pages, bool) or not isinstance(overlap_pages, int):
        raise TypeError("overlap_pages must be an integer.")
    if overlap_pages < 0:
        raise ValueError("overlap_pages cannot be negative.")

    if not protocol_text:
        return ChunkingResult(
            chunks=(),
            warnings=("Input is empty; no chunks were created.",),
        )

    pages, warnings = _parse_pages(protocol_text)
    atoms = _split_pages(pages, max_chars)
    groups = _page_groups(atoms)
    chunks: list[ProtocolChunk] = []
    previous_bounds: Optional[tuple[int, int]] = None
    cursor = 0

    while cursor < len(atoms):
        overlap_atoms: list[_PageAtom] = []
        if overlap_pages and previous_bounds is not None:
            overlap_atoms, overlap_unavailable = _select_overlap_groups(
                atoms,
                groups,
                previous_bounds,
                cursor,
                overlap_pages,
                max_chars,
            )
            if overlap_unavailable:
                warnings.append(
                    "Requested page overlap was unavailable or reduced; "
                    "original pages remain in their primary chunks."
                )

        current_atoms = list(overlap_atoms)
        start_cursor = cursor
        while cursor < len(atoms):
            candidate = current_atoms + [atoms[cursor]]
            if len(_render_atoms(candidate)) > max_chars:
                break
            current_atoms.append(atoms[cursor])
            cursor += 1

        if cursor == start_cursor:
            if overlap_atoms:
                current_atoms = []
                overlap_atoms = []
                while cursor < len(atoms):
                    candidate = current_atoms + [atoms[cursor]]
                    if len(_render_atoms(candidate)) > max_chars:
                        break
                    current_atoms.append(atoms[cursor])
                    cursor += 1
                warnings.append(
                    "Requested page overlap was omitted because it left no "
                    "room for new source text."
                )
            if cursor == start_cursor:
                raise ValueError(
                    "A page fragment could not fit within max_chars."
                )

        chunk_pages = tuple(
            ProtocolPage(
                page_number=atom.page_number,
                text=atom.text,
                sections=atom.sections,
                continuation=atom.continuation,
                overlap=index < len(overlap_atoms),
            )
            for index, atom in enumerate(current_atoms)
        )
        chunk_text = _render_atoms(current_atoms)
        chunk_sections = tuple(
            category
            for category, _pattern in _SECTION_PATTERNS
            if any(category in page.sections for page in chunk_pages)
        )
        chunks.append(
            ProtocolChunk(
                index=len(chunks),
                pages=chunk_pages,
                text=chunk_text,
                sections=chunk_sections,
            )
        )
        previous_bounds = (start_cursor, cursor)

    return ChunkingResult(chunks=tuple(chunks), warnings=tuple(warnings))
