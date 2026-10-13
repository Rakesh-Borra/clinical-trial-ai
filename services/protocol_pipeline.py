
from services.pdf_processor import extract_pdf
from services.source_verifier import verify_source_pages
from agents.protocol_analysis_agent import analyze_protocol


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

    # Step 4: Analyze the clinical trial protocol
    analysis = analyze_protocol(protocol_text)

    # Step 5: Verify source references
    verification = verify_source_pages(
        analysis,
        pdf_result,
    )

    return {
        "analysis": analysis,
        "verification": verification,
        "extraction_warnings": list(pdf_result.get("warnings") or []),
    }
