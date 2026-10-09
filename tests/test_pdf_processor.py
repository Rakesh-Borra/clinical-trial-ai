
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
