
import os
import time

from services import benchmark_log, performance
from services.analysis_progress import report_stage
from services import analysis_cache
from services.patient_relevance import filter_protocol_blocks
from services.completeness_audit import audit_completeness
from services.section_chunker import build_chunks
from services.protocol_chunker import chunk_protocol
from services.pdf_processor import extract_pdf
from services.source_verifier import verify_source_pages
from services.protocol_accuracy import apply_protocol_accuracy_checks
from agents.protocol_analysis_agent import (
    analyze_protocol_in_chunks,
    analyze_protocol_with_rescue,
)

LONG_PROTOCOL_THRESHOLD_CHARS = int(
    os.getenv("CTAI_LONG_PROTOCOL_THRESHOLD_CHARS", "12000")
)
PROTOCOL_CHUNK_CHARS = int(os.getenv("CTAI_PROTOCOL_CHUNK_CHARS", "8000"))
PROTOCOL_MAX_INPUT_CHARS = int(
    os.getenv("CTAI_PROTOCOL_MAX_INPUT_CHARS", "320000")
)
PROTOCOL_MAX_CALLS = int(os.getenv("CTAI_PROTOCOL_MAX_CALLS", "65"))
# Token budget per chunk (estimated); the default matches ~8000 characters.
PROTOCOL_CHUNK_TOKENS = int(os.getenv("CTAI_PROTOCOL_CHUNK_TOKENS", "2500"))
ANALYSIS_MAX_SECONDS = float(os.getenv("CTAI_ANALYSIS_MAX_SECONDS", "1800"))
PROTOCOL_RESCUE_MAX_CHARS = int(
    os.getenv("CTAI_PROTOCOL_RESCUE_MAX_CHARS", "8000")
)


def _analysis_settings() -> dict:
    from agents import protocol_analysis_agent as agent
    from services.model_provider import provider_display_name

    return {
        "provider": provider_display_name(),
        "model": agent.MODEL_NAME,
        "version": agent.ANALYSIS_VERSION,
        "think": agent.EXTRACTION_THINK,
        "num_predict": agent.NUM_PREDICT,
        "num_ctx": agent.NUM_CTX,
        "chunk_tokens": PROTOCOL_CHUNK_TOKENS,
        "route": True,
    }


def _relevance_summary(pages, block_result) -> dict:
    """Summarize filtering; a page counts as skipped only if all its blocks were."""
    kept_pages = {b.page_number for b in block_result.blocks if b.keep}
    skipped_pages = [
        {"page": page["page_number"], "reason": "all sections administrative"}
        for page in pages
        if page["page_number"] not in kept_pages
    ]
    return {
        "enabled": block_result.enabled,
        "total_pages": len(pages),
        "analyzed_pages": len(pages) - len(skipped_pages),
        "skipped_pages": skipped_pages,
        "blocks": block_result.summary(),
    }


def analyze_protocol_pdf(file_path: str):
    """
    Extract clinical trial information from a PDF,
    analyze it using Qwen, and verify source pages.

    Clearly administrative sections are skipped before the model runs (see
    services/patient_relevance.py); the result reports what was skipped and
    includes lightweight performance measurements.
    """
    label = benchmark_log.label_for(file_path)
    started = time.perf_counter()
    try:
        with performance.track() as tracker:
            result = _analyze_protocol_pdf(file_path, tracker)
    except Exception as exc:
        benchmark_log.record(
            "analysis", label, success=False,
            seconds=round(time.perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    metrics = tracker.summary()
    metrics.update(result.pop("_chunk_counts", {}))
    result["performance"] = metrics
    if benchmark_log.enabled():
        benchmark_log.record(
            "analysis", label,
            **{
                **metrics,
                "success": bool(result.get("analysis_complete")),
                "seconds": round(time.perf_counter() - started, 3),
                "verification_passed": result.get("verification_passed"),
                "unsupported_statements": result.get("unsupported_statements"),
                "pages_extracted": (result.get("pages") or {}).get("extracted"),
            },
        )
    return result


def _analyze_protocol_pdf(file_path: str, tracker: performance.PerfTracker):

    # Step 1: Extract text from the PDF
    with tracker.stage("extract"):
        pdf_result = extract_pdf(file_path)

    # Step 2: Combine readable pages
    protocol_pages = []

    for page in pdf_result["pages"]:
        if page["text"].strip():
            protocol_pages.append(
                f"[SOURCE PAGE {page['page_number']}]\n"
                f"{page['text']}"
            )

    if not protocol_pages:
        warning_detail = ""
        extraction_warnings = pdf_result.get("warnings") or []
        if extraction_warnings:
            warning_detail = (
                " Extraction warnings: "
                + "; ".join(extraction_warnings)
            )
        raise ValueError(
            "The protocol PDF contains no extractable text "
            "after selectable-text extraction and OCR fallback."
            + warning_detail
        )

    # Step 3: Combine all pages (the full text is kept for accuracy checks)
    protocol_text = "\n\n".join(protocol_pages)

    full_page_text = "\n\n".join(
        f"[SOURCE PAGE {page['page_number']}]\n{page.get('text') or ''}"
        for page in pdf_result["pages"]
    )
    is_long_protocol = len(full_page_text) > LONG_PROTOCOL_THRESHOLD_CHARS

    # Patient-relevance filtering works on blocks of ALL original pages, so a
    # page mixing patient and administrative content is never dropped whole.
    # Only the text sent to the model changes; the PDF result is untouched.
    with tracker.stage("relevance_filter"):
        block_result = filter_protocol_blocks(pdf_result["pages"])
        chunks = build_chunks(block_result.blocks, PROTOCOL_CHUNK_TOKENS)
        relevance = _relevance_summary(pdf_result["pages"], block_result)
    chunk_text = "\n\n".join(chunk.text for chunk in chunks)
    chunk_counts = {
        "chunks_without_filter": len(
            chunk_protocol(full_page_text, max_chars=PROTOCOL_CHUNK_CHARS).chunks
        ) if is_long_protocol else 1,
        "chunks_analyzed": len(chunks) if is_long_protocol else 1,
        "pages_total": len(pdf_result["pages"]),
        "pages_analyzed": relevance["analyzed_pages"],
        "blocks_total": block_result.total_blocks,
        "blocks_analyzed": block_result.summary()["analyzed_blocks"],
        "chars_without_filter": len(full_page_text),
        "chars_analyzed": len(chunk_text),
    }

    analysis_started = time.perf_counter()
    if is_long_protocol:
        cache_key = analysis_cache.make_key(chunk_text, _analysis_settings())
        cached = analysis_cache.load(cache_key)
        if cached is not None:
            analysis = cached["analysis"]
            long_warnings = list(cached["warnings"]) + [
                "Model analysis was reused from the cache (same document "
                "text and settings)."
            ]
            failed_chunks, incomplete_reasons, llm_call_count = [], [], 0
        else:
            (
                analysis,
                long_warnings,
                failed_chunks,
                incomplete_reasons,
                llm_call_count,
            ) = analyze_protocol_in_chunks(
                chunk_text,
                chunk_chars=PROTOCOL_CHUNK_CHARS,
                max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
                max_calls=PROTOCOL_MAX_CALLS,
                rescue_max_chars=PROTOCOL_RESCUE_MAX_CHARS,
                chunks=chunks,
                route=True,
                max_seconds=ANALYSIS_MAX_SECONDS,
            )
            if not failed_chunks and not incomplete_reasons:
                analysis_cache.store(cache_key, analysis, long_warnings)
        unreadable_pages = [
            page["page_number"]
            for page in pdf_result["pages"]
            if not (page.get("text") or "").strip()
        ]
        if unreadable_pages:
            warning = (
                "Pages with no extractable text were not available to the "
                f"analysis: {unreadable_pages}."
            )
            long_warnings.append(warning)
            incomplete_reasons.append(warning)
        if pdf_result.get("warnings"):
            incomplete_reasons.extend(
                warning for warning in pdf_result["warnings"]
                if warning not in incomplete_reasons
            )
        accuracy_warnings = list(long_warnings)
        reported_complete = getattr(analysis, "analysis_complete", None)
        if reported_complete is not True and not incomplete_reasons:
            warning = (
                "Long-protocol analysis completeness could not be verified."
            )
            long_warnings.append(warning)
            incomplete_reasons.append(warning)
            accuracy_warnings.append(warning)
        analysis_complete = (
            reported_complete is True
            and not incomplete_reasons
        )
    else:
        # Preserve the existing focused two-pass path for shorter protocols.
        analysis, rescue_notes = analyze_protocol_with_rescue(chunk_text)
        accuracy_warnings = list(rescue_notes)
        failed_chunks = []
        incomplete_reasons = []
        llm_call_count = None
        analysis_complete = True

    tracker.stage_seconds["llm_analysis"] = (
        time.perf_counter() - analysis_started
    )
    analysis.analysis_complete = analysis_complete

    report_stage("accuracy")
    # Step 5: Deterministic grounding, page repair, and completeness checks
    with tracker.stage("accuracy_checks"):
        accuracy_warnings.extend(
            apply_protocol_accuracy_checks(
                analysis,
                pdf_result,
                protocol_text,
            )
        )

    report_stage("verify")
    # Step 6: Verify source references (after page repair)
    with tracker.stage("verification"):
        verification = verify_source_pages(
            analysis,
            pdf_result,
        )

    # "Ran to completion" and "everything extracted checked out" are different
    # facts: a failed source check blocks only the affected statements.
    unsupported = [
        row for row in verification if row.get("status") != "exact_match"
    ]
    analysis.verification_passed = not unsupported
    pages_with_text = [
        page["page_number"] for page in pdf_result["pages"]
        if (page.get("text") or "").strip()
    ]
    result = {
        "analysis": analysis,
        "analysis_complete": analysis_complete,
        "execution_complete": analysis_complete,
        "verification_passed": not unsupported,
        "unsupported_statements": len(unsupported),
        "pages": {
            "total": len(pdf_result["pages"]),
            "extracted": len(pages_with_text),
            "analyzed": relevance["analyzed_pages"],
            "unreadable": [
                page["page_number"] for page in pdf_result["pages"]
                if not (page.get("text") or "").strip()
            ],
        },
        # Cited page text lets an edited statement be rechecked against its
        # source without re-reading the PDF.
        "source_pages": {
            str(page["page_number"]): page.get("text") or ""
            for page in pdf_result["pages"]
        },
        "verification": verification,
        "extraction_warnings": list(pdf_result.get("warnings") or []),
        "accuracy_warnings": accuracy_warnings,
        "relevance": relevance,
        "completeness_audit": audit_completeness(block_result.blocks, analysis),
        "_chunk_counts": chunk_counts,
    }
    if is_long_protocol:
        result.update({
            "incomplete_chunks": failed_chunks,
            "incomplete_reasons": incomplete_reasons,
            "llm_call_count": llm_call_count,
        })
    return result
