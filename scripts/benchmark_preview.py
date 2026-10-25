"""
Ollama-free preview of how each protocol PDF would be processed.

Runs the app's real extraction, relevance filtering and chunking, then applies
the same limit checks as analyze_protocol_in_chunks, WITHOUT calling any
model. PDFs are read in place; nothing is copied or written to the project.

    python scripts/benchmark_preview.py "/path/to/Protocol samples" [out.json]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.protocol_analysis_agent import route_passes  # noqa: E402
from services import protocol_pipeline as pipe  # noqa: E402
from services.patient_relevance import filter_protocol_blocks  # noqa: E402
from services.pdf_processor import extract_pdf  # noqa: E402
from services.protocol_chunker import chunk_protocol  # noqa: E402
from services.section_chunker import build_chunks  # noqa: E402


def preview(pdf_path: Path) -> dict:
    started = time.perf_counter()
    pdf = extract_pdf(str(pdf_path))
    extract_seconds = time.perf_counter() - started

    pages = pdf["pages"]
    full_text = "\n\n".join(
        f"[SOURCE PAGE {p['page_number']}]\n{p.get('text') or ''}" for p in pages
    )
    started = time.perf_counter()
    blocks = filter_protocol_blocks(pages)
    chunks = build_chunks(blocks.blocks, pipe.PROTOCOL_CHUNK_TOKENS)
    filter_seconds = time.perf_counter() - started
    relevant_text = "\n\n".join(c.text for c in chunks)
    is_long = len(full_text) > pipe.LONG_PROTOCOL_THRESHOLD_CHARS

    row = {
        "file": pdf_path.name,
        "pages": len(pages),
        "pages_with_text": sum(1 for p in pages if (p.get("text") or "").strip()),
        "extracted_chars": len(full_text),
        "relevant_chars": len(relevant_text),
        "extract_seconds": round(extract_seconds, 2),
        "filter_chunk_seconds": round(filter_seconds, 2),
        "extraction_warnings": len(pdf.get("warnings") or []),
        "path": "long (chunked)" if is_long else "short (2 passes + rescue)",
    }

    if not is_long:
        row.update(
            chunks_filtered=1,
            chunks_unfiltered=1,
            estimated_calls_min=2,
            estimated_calls_max=3,
            fits=True,
            limit_note="short path: no chunk/call/input caps apply",
        )
        return row

    plans = [
        route_passes(
            c.text, i == 0, getattr(c, "categories", ()),
            getattr(c, "fully_classified", False), getattr(c, "has_table", False),
        )
        for i, c in enumerate(chunks)
    ]
    routed = sum(a + b for a, b in plans)
    # The real cap check uses the worst case (2 calls per chunk + 1 rescue).
    worst = len(chunks) * 2 + 1
    max_chunks = (pipe.PROTOCOL_MAX_CALLS - 1) // 2
    problems = []
    if len(relevant_text) > pipe.PROTOCOL_MAX_INPUT_CHARS:
        problems.append(
            f"input {len(relevant_text)} > {pipe.PROTOCOL_MAX_INPUT_CHARS} chars"
        )
    if len(chunks) > max_chunks:
        problems.append(
            f"{len(chunks)} chunks > {max_chunks} allowed "
            f"(worst case {worst} calls > cap {pipe.PROTOCOL_MAX_CALLS})"
        )
    row.update(
        chunks_filtered=len(chunks),
        chunks_unfiltered=len(
            chunk_protocol(full_text, max_chars=pipe.PROTOCOL_CHUNK_CHARS).chunks
        ),
        estimated_calls_min=routed,
        estimated_calls_max=routed + 1,
        worst_case_calls=worst,
        fits=not problems,
        limit_note="; ".join(problems) or "within configured limits",
    )
    return row


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if folder is None or not folder.is_dir():
        sys.exit(__doc__)
    rows = []
    for pdf in sorted(folder.glob("*.pdf"), key=lambda p: p.name):
        print(f"previewing {pdf.name} ...", file=sys.stderr, flush=True)
        try:
            rows.append(preview(pdf))
        except Exception as exc:  # report, keep going
            rows.append({"file": pdf.name, "error": f"{type(exc).__name__}: {exc}"})
    text = json.dumps(
        {
            "limits": {
                "long_threshold_chars": pipe.LONG_PROTOCOL_THRESHOLD_CHARS,
                "chunk_tokens": pipe.PROTOCOL_CHUNK_TOKENS,
                "max_input_chars": pipe.PROTOCOL_MAX_INPUT_CHARS,
                "max_calls": pipe.PROTOCOL_MAX_CALLS,
                "max_seconds": pipe.ANALYSIS_MAX_SECONDS,
            },
            "results": rows,
        },
        indent=2,
    )
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
