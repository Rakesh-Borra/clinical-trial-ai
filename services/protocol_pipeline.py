from services.pdf_processor import extract_pdf
from agents.protocol_analysis_agent import analyze_protocol


def analyze_protocol_pdf(file_path: str):
    """
    Extract text from a clinical trial protocol PDF
    and send it to the Protocol Analysis Agent.
    """

    # Step 1: Read the protocol PDF
    pdf_result = extract_pdf(file_path)

    # Step 2: Combine text from all readable pages
    protocol_pages = []

    for page in pdf_result["pages"]:
        if page["text"].strip():
            protocol_pages.append(
                f"[SOURCE PAGE {page['page_number']}]\n"
                f"{page['text']}"
            )

    if not protocol_pages:
        raise ValueError(
            "The protocol PDF contains no extractable text. "
            "OCR may be required."
        )

    # Step 3: Combine pages into one document
    protocol_text = "\n\n".join(protocol_pages)

    # Step 4: Ask our AI agent to analyze the protocol
    analysis = analyze_protocol(protocol_text)

    return analysis