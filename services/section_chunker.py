"""
Structure- and token-aware chunking of patient-relevant protocol blocks.

Chunks are packed from whole heading-delimited blocks, so a section is not cut
in the middle unless it alone exceeds the token budget; such a block is split
at paragraph, then line, then whitespace boundaries (never inside a clinical
value) and the pieces are marked as continuations. Limited character overlap
is applied only when an oversized section is split, so boundary context is not
lost. Every page fragment keeps its original page number and a
[SOURCE PAGE n] marker.

Token counts are estimates (no tokenizer dependency): clinical text with
numbers and abbreviations averages roughly three characters per token, so the
estimate is deliberately conservative.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from services.patient_relevance import Block
from services.protocol_chunker import (
    ProtocolChunk,
    ProtocolPage,
    _safe_split_end,
    detect_sections,
)

CHARS_PER_TOKEN = 3.2
# Default overlap when an oversized section must be split across chunks.
DEFAULT_OVERLAP_CHARS = 400


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


def _split_oversized(
    block: Block,
    max_chars: int,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[Block]:
    """Split one block that cannot fit a chunk, keeping its page number."""
    if block.is_table:
        pieces = _split_table(block, max_chars)
        if pieces:
            return pieces
    pieces = []
    start = 0
    text = block.text
    while start < len(text):
        end = _safe_split_end(text, start, max_chars)
        if end <= start:
            end = min(start + max_chars, len(text))
        pieces.append(
            Block(block.page_number, block.heading, block.depth,
                  text[start:end],
                  reason="continuation" if start else block.reason,
                  categories=block.categories,
                  category_source=block.category_source,
                  is_table=block.is_table)
        )
        if end >= len(text):
            break
        next_start = end
        if overlap_chars > 0:
            overlapped = end - overlap_chars
            if overlapped > start:
                next_start = overlapped
        if next_start <= start:
            next_start = end
        start = next_start
    return pieces


def build_chunks(
    blocks: Iterable[Block],
    max_tokens: int,
    *,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[ProtocolChunk]:
    """Pack kept blocks, in document order, into token-bounded chunks."""
    if max_tokens < 50:
        raise ValueError("max_tokens must be at least 50.")
    if isinstance(overlap_chars, bool) or not isinstance(overlap_chars, int):
        raise TypeError("overlap_chars must be an integer.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    # (block_piece, is_continuation, source_block_id)
    pieces: List[Tuple[Block, bool, int]] = []
    for block_id, block in enumerate(blocks):
        if not block.keep:
            continue
        marker_len = len(_marker(block.page_number))
        if len(block.text) + marker_len > max_chars:
            parts = _split_oversized(
                block, max_chars - marker_len, overlap_chars=overlap_chars,
            )
            # Row-wise table pieces are complete (header repeated), so they are
            # not continuations; character splits of any block are.
            pieces.extend(
                (part, index > 0 and not _is_whole_table(part), block_id)
                for index, part in enumerate(parts)
            )
        else:
            pieces.append((block, False, block_id))

    chunks: List[ProtocolChunk] = []
    current: List[Tuple[Block, bool, int]] = []
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
            body = "\n\n".join(b.text for b, _, _ in group)
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
        blocks_here = [b for b, _, _ in current]
        source_ids = tuple(dict.fromkeys(block_id for _, _, block_id in current))
        chunks.append(ProtocolChunk(
            len(chunks), tuple(pages), text, sections,
            categories=tuple(sorted({c for b in blocks_here for c in b.categories})),
            fully_classified=all(
                b.category_source == "heading" for b in blocks_here
            ),
            has_table=any(b.is_table for b in blocks_here),
            source_block_ids=source_ids,
        ))
        current, used = [], 0

    for block, continuation, block_id in pieces:
        cost = len(block.text) + len(_marker(block.page_number)) + 2
        if current and used + cost > max_chars:
            flush()
        current.append((block, continuation, block_id))
        used += cost
    flush()
    return chunks


def uncovered_source_blocks(
    blocks: Sequence[Block],
    chunks: Sequence[ProtocolChunk],
) -> List[str]:
    """
    Return warnings for kept blocks that were not packed into any chunk.

    ``build_chunks`` records ``source_block_ids`` for every packed piece,
    including character-split continuations, so a missing id means silent
    truncation. When a block fits in one piece, its exact text must also
    appear in the chunk corpus.
    """
    corpus = "\n\n".join(chunk.text for chunk in chunks)
    covered_ids = {
        block_id for chunk in chunks for block_id in chunk.source_block_ids
    }
    warnings: List[str] = []
    for block_id, block in enumerate(blocks):
        if not block.keep or not block.text.strip():
            continue
        if block_id not in covered_ids:
            warnings.append(
                f"Source block {block_id} (page {block.page_number}, "
                f"heading {block.heading!r}) was not packed into any chunk."
            )
            continue
        # Unsplit blocks must appear verbatim; split blocks are exact slices
        # recorded under the same source_block_id across one or more chunks.
        piece_count = sum(
            1 for chunk in chunks if block_id in chunk.source_block_ids
        )
        if piece_count == 1 and block.text not in corpus:
            warnings.append(
                f"Source block {block_id} (page {block.page_number}, "
                f"heading {block.heading!r}) is not fully covered by chunks."
            )
    return warnings
