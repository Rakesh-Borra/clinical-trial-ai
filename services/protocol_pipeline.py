
import os

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
PROTOCOL_RESCUE_MAX_CHARS = int(
    os.getenv("CTAI_PROTOCOL_RESCUE_MAX_CHARS", "8000")
)


def analyze_protocol_pdf(file_path: str):
    """
    Extract clinical trial information from a PDF,
    analyze it using Qwen, and verify source pages.
    """

    # Step 1: Extract text from the PDF
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

    # Step 3: Combine all pages
    protocol_text = "\n\n".join(protocol_pages)

    all_page_text = "\n\n".join(
        f"[SOURCE PAGE {page['page_number']}]\n{page.get('text') or ''}"
        for page in pdf_result["pages"]
    )
    is_long_protocol = len(all_page_text) > LONG_PROTOCOL_THRESHOLD_CHARS

    if is_long_protocol:
        (
            analysis,
            long_warnings,
            failed_chunks,
            incomplete_reasons,
            llm_call_count,
        ) = analyze_protocol_in_chunks(
            all_page_text,
            chunk_chars=PROTOCOL_CHUNK_CHARS,
            max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
            max_calls=PROTOCOL_MAX_CALLS,
            rescue_max_chars=PROTOCOL_RESCUE_MAX_CHARS,
        )
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
        analysis, rescue_notes = analyze_protocol_with_rescue(protocol_text)
        accuracy_warnings = list(rescue_notes)
        failed_chunks = []
        incomplete_reasons = []
        llm_call_count = None
        analysis_complete = True

    analysis.analysis_complete = analysis_complete

    # Step 5: Deterministic grounding, page repair, and completeness checks
    accuracy_warnings.extend(
        apply_protocol_accuracy_checks(
            analysis,
            pdf_result,
            protocol_text,
        )
    )

    # Step 6: Verify source references (after page repair)
    verification = verify_source_pages(
        analysis,
        pdf_result,
    )

    result = {
        "analysis": analysis,
        "analysis_complete": analysis_complete,
        "verification": verification,
        "extraction_warnings": list(pdf_result.get("warnings") or []),
        "accuracy_warnings": accuracy_warnings,
    }
    if is_long_protocol:
        result.update({
            "incomplete_chunks": failed_chunks,
            "incomplete_reasons": incomplete_reasons,
            "llm_call_count": llm_call_count,
        })
    return result
