"""
Structure- and token-aware chunking of patient-relevant protocol blocks.

Chunks are packed from whole heading-delimited blocks, so a section is not cut
in the middle unless it alone exceeds the token budget; such a block is split
at paragraph, then line, then whitespace boundaries (never inside a clinical
value) and the pieces are marked as continuations. Every page fragment keeps
its original page number and a [SOURCE PAGE n] marker.

Token counts are estimates (no tokenizer dependency): clinical text with
numbers and abbreviations averages roughly three characters per token, so the
estimate is deliberately conservative.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

from services.patient_relevance import Block
from services.protocol_chunker import (
    ProtocolChunk,
    ProtocolPage,
    _safe_split_end,
    detect_sections,
)

CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def _marker(page_number: Optional[int]) -> str:
    return f"[SOURCE PAGE {page_number}]\n" if page_number is not None else ""


def _is_whole_table(block: Block) -> bool:
    text = block.text.strip()
    return block.is_table and text.startswith("[TABLE]") and text.endswith("[/TABLE]")


def _split_table(block: Block, max_chars: int) -> List[Block]:
    """Split a table between rows, repeating the header row in every piece."""
    lines = block.text.split("\n")
    body = [ln for ln in lines if ln not in ("[TABLE]", "[/TABLE]")]
    if len(body) < 2:
        return []
    header, rows = body[0], body[1:]
    pieces, current = [], []

    def close() -> None:
        if current:
            text = "\n".join(["[TABLE]", header, *current, "[/TABLE]"])
            pieces.append(Block(
                block.page_number, block.heading, block.depth, text,
                reason="continuation" if pieces else block.reason,
                categories=block.categories,
                category_source=block.category_source, is_table=True,
            ))

    for row in rows:
        candidate = len(header) + len(row) + sum(len(r) + 1 for r in current) + 24
        if current and candidate > max_chars:
            close()
            current = []
        current.append(row)
    close()
    # A single row larger than the budget cannot be split safely here.
    if any(len(p.text) > max_chars for p in pieces):
        return []
    return pieces


def _split_oversized(block: Block, max_chars: int) -> List[Block]:
    """Split one block that cannot fit a chunk, keeping its page number."""
    if block.is_table:
        pieces = _split_table(block, max_chars)
        if pieces:
            return pieces
    pieces = []
    start = 0
    while start < len(block.text):
        end = _safe_split_end(block.text, start, max_chars)
        if end <= start:
            end = min(start + max_chars, len(block.text))
        pieces.append(
            Block(block.page_number, block.heading, block.depth,
                  block.text[start:end],
                  reason="continuation" if start else block.reason,
                  categories=block.categories,
                  category_source=block.category_source,
                  is_table=block.is_table)
        )
        start = end
    return pieces


def build_chunks(blocks: Iterable[Block], max_tokens: int) -> List[ProtocolChunk]:
    """Pack kept blocks, in document order, into token-bounded chunks."""
    if max_tokens < 50:
        raise ValueError("max_tokens must be at least 50.")
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    pieces: List[tuple[Block, bool]] = []
    for block in blocks:
        if not block.keep:
            continue
        marker_len = len(_marker(block.page_number))
        if len(block.text) + marker_len > max_chars:
            parts = _split_oversized(block, max_chars - marker_len)
            # Row-wise table pieces are complete (header repeated), so they are
            # not continuations; character splits of any block are.
            pieces.extend(
                (part, index > 0 and not _is_whole_table(part))
                for index, part in enumerate(parts)
            )
        else:
            pieces.append((block, False))

    chunks: List[ProtocolChunk] = []
    current: List[tuple[Block, bool]] = []
    used = 0

    def flush() -> None:
        nonlocal current, used
        if not current:
            return
        pages: List[ProtocolPage] = []
        texts: List[str] = []
        index = 0
        while index < len(current):
            page_number = current[index][0].page_number
            group = [current[index]]
            while (
                index + len(group) < len(current)
                and current[index + len(group)][0].page_number == page_number
            ):
                group.append(current[index + len(group)])
            body = "\n\n".join(b.text for b, _ in group)
            pages.append(ProtocolPage(
                page_number=page_number,
                text=body,
                sections=detect_sections(body),
                continuation=group[0][1],
            ))
            texts.append(_marker(page_number) + body)
            index += len(group)
        text = "\n\n".join(texts)
        sections = tuple(sorted({s for p in pages for s in p.sections}))
        blocks_here = [b for b, _ in current]
        chunks.append(ProtocolChunk(
            len(chunks), tuple(pages), text, sections,
            categories=tuple(sorted({c for b in blocks_here for c in b.categories})),
            fully_classified=all(
                b.category_source == "heading" for b in blocks_here
            ),
            has_table=any(b.is_table for b in blocks_here),
        ))
        current, used = [], 0

    for block, continuation in pieces:
        cost = len(block.text) + len(_marker(block.page_number)) + 2
        if current and used + cost > max_chars:
            flush()
        current.append((block, continuation))
        used += cost
    flush()
    return chunks
