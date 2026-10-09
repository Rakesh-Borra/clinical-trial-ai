
import pymupdf
import pytest

from services.pdf_processor import (
    PDFProcessingError,
    extract_pdf,
)


def test_extract_multipage_pdf(tmp_path):
    pdf_path = tmp_path / "sample_protocol.pdf"

    document = pymupdf.open()

    for text in [
        "Clinical trial eligibility criteria",
        "Visit schedule and procedures",
    ]:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(pdf_path)
    document.close()

    result = extract_pdf(str(pdf_path))

    assert result["total_pages"] == 2
    assert result["pages"][0]["page_number"] == 1
    assert "eligibility" in result["pages"][0]["text"]
    assert result["pages"][1]["requires_ocr"] is False


def test_missing_pdf():
    with pytest.raises(PDFProcessingError):
        extract_pdf("nonexistent_protocol.pdf")


def test_scanned_or_empty_page(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    result = extract_pdf(str(pdf_path))

    assert result["pages"][0]["requires_ocr"] is True



from services.pdf_processor import extract_structured_pdf


def test_structured_pdf_extraction(tmp_path):
    pdf_path = tmp_path / "structured_protocol.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Inclusion Criteria: Adults aged 18 or older."
    )

    document.save(pdf_path)
    document.close()

    result = extract_structured_pdf(str(pdf_path))

    assert result["total_pages"] == 1
    assert len(result["pages"][0]["blocks"]) > 0

    first_block = result["pages"][0]["blocks"][0]

    assert first_block["block_id"] == "p1_b1"
    assert "Inclusion Criteria" in first_block["text"]
    assert len(first_block["bbox"]) == 4
