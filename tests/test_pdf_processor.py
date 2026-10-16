
from unittest.mock import patch

import pymupdf
import pytest

from services.pdf_processor import (
    PDFProcessingError,
    extract_pdf,
    extract_structured_pdf,
)


def _make_text_pdf(path, page_texts):
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _make_image_only_page(document, text, dpi=150):
    """Create a page that contains only a rendered image of text."""
    temp = pymupdf.open()
    source = temp.new_page()
    source.insert_text((72, 72), text, fontsize=14)
    pixmap = source.get_pixmap(dpi=dpi)
    temp.close()

    page = document.new_page(
        width=pixmap.width,
        height=pixmap.height,
    )
    page.insert_image(page.rect, pixmap=pixmap)
    return page


def test_extract_multipage_pdf(tmp_path):
    pdf_path = tmp_path / "sample_protocol.pdf"

    _make_text_pdf(
        pdf_path,
        [
            "Clinical trial eligibility criteria",
            "Visit schedule and procedures",
        ],
    )

    result = extract_pdf(str(pdf_path))

    assert result["total_pages"] == 2
    assert result["pages"][0]["page_number"] == 1
    assert "eligibility" in result["pages"][0]["text"]
    assert result["pages"][1]["requires_ocr"] is False
    assert result["pages"][0]["ocr_applied"] is False
    assert result["pages"][1]["ocr_applied"] is False
    assert result["warnings"] == []


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
    assert result["pages"][0]["ocr_applied"] is False
    assert result["pages"][0]["text"] == ""


def test_ocr_fallback_on_scanned_page(tmp_path):
    pdf_path = tmp_path / "scanned_protocol.pdf"

    document = pymupdf.open()
    _make_image_only_page(
        document,
        "Screening Visit: Day 0",
    )
    document.save(pdf_path)
    document.close()

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="Screening Visit: Day 0",
    ) as mock_ocr:
        result = extract_pdf(str(pdf_path))

    mock_ocr.assert_called_once()
    page = result["pages"][0]

    assert page["page_number"] == 1
    assert "Screening Visit: Day 0" in page["text"]
    assert page["ocr_applied"] is True
    assert page["requires_ocr"] is False
    assert result["warnings"] == []


def test_mixed_selectable_and_scanned_pdf(tmp_path):
    pdf_path = tmp_path / "mixed_protocol.pdf"

    document = pymupdf.open()
    text_page = document.new_page()
    text_page.insert_text(
        (72, 72),
        "PROTOCOL TITLE: Mixed OCR Study",
    )
    _make_image_only_page(
        document,
        "Follow-up Visit: Day 30",
    )
    document.save(pdf_path)
    document.close()

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="Follow-up Visit: Day 30",
    ) as mock_ocr:
        result = extract_pdf(str(pdf_path))

    mock_ocr.assert_called_once()

    assert result["total_pages"] == 2
    assert result["pages"][0]["page_number"] == 1
    assert "Mixed OCR Study" in result["pages"][0]["text"]
    assert result["pages"][0]["ocr_applied"] is False

    assert result["pages"][1]["page_number"] == 2
    assert "Follow-up Visit: Day 30" in result["pages"][1]["text"]
    assert result["pages"][1]["ocr_applied"] is True
    assert result["pages"][1]["requires_ocr"] is False


def test_ocr_dependency_missing_records_warning(tmp_path):
    pdf_path = tmp_path / "scanned_missing_ocr.pdf"

    document = pymupdf.open()
    _make_image_only_page(document, "Hidden scanned text")
    document.save(pdf_path)
    document.close()

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        side_effect=PDFProcessingError(
            "OCR requires the Tesseract OCR binary on PATH."
        ),
    ):
        result = extract_pdf(str(pdf_path))

    page = result["pages"][0]
    assert page["text"] == ""
    assert page["requires_ocr"] is True
    assert page["ocr_applied"] is False
    assert len(result["warnings"]) == 1
    assert "Tesseract" in result["warnings"][0]


def test_ocr_empty_result_does_not_invent_text(tmp_path):
    pdf_path = tmp_path / "blank_scan.pdf"

    document = pymupdf.open()
    _make_image_only_page(document, "   ")
    document.save(pdf_path)
    document.close()

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="   ",
    ):
        result = extract_pdf(str(pdf_path))

    page = result["pages"][0]
    assert page["text"] == ""
    assert page["ocr_applied"] is False
    assert page["requires_ocr"] is True
    assert "OCR returned no text" in result["warnings"][0]


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _tesseract_available(),
    reason="Tesseract OCR binary or pytesseract not available",
)
def test_live_ocr_on_synthetic_scanned_pdf(tmp_path):
    pdf_path = tmp_path / "live_scanned.pdf"

    document = pymupdf.open()
    _make_image_only_page(document, "Inclusion Criteria: Age 18 or older")
    document.save(pdf_path)
    document.close()

    result = extract_pdf(str(pdf_path))
    page = result["pages"][0]

    assert page["ocr_applied"] is True
    assert page["requires_ocr"] is False
    assert "Inclusion" in page["text"] or "Age" in page["text"]
    assert result["warnings"] == []


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


# ---------------------------------------------------------------------------
# Scrambled-text detection (added after evaluation on real protocols:
# NCT02879383 extracts as "3URWRFRO" instead of "Protocol" on 84/86 pages).
# ---------------------------------------------------------------------------

from unittest.mock import patch as _patch

from services.pdf_processor import PDFProcessingError as _PDFError
from services.pdf_processor import _looks_garbled

_READABLE = (
    "Subjects must be at least 18 years of age and must have type 2 diabetes "
    "for at least one year. Subjects with a history of kidney disease will be "
    "excluded from the study. All subjects will be asked to sign the informed "
    "consent form before any procedures are performed at the screening visit."
)


def _scramble(text: str) -> str:
    # The same shift the broken font in NCT02879383 produces ("P" -> "3").
    return "".join(chr(ord(c) - 29) if c.isalpha() else c for c in text)


def test_looks_garbled_flags_scrambled_text_only():
    assert _looks_garbled(_scramble(_READABLE)) is True
    assert _looks_garbled(_READABLE) is False
    # Short text (titles, headers) is never judged.
    assert _looks_garbled(_scramble("Protocol Title Page")) is False


def _pdf_with_text(tmp_path, text):
    pdf_path = tmp_path / "scrambled.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    document.save(pdf_path)
    document.close()
    return pdf_path


def test_scrambled_page_is_replaced_by_ocr_text(tmp_path):
    pdf_path = _pdf_with_text(tmp_path, _scramble(_READABLE))

    with _patch("services.pdf_processor._ocr_pdf_page", return_value=_READABLE):
        result = extract_pdf(str(pdf_path))

    page = result["pages"][0]
    assert page["garbled_text_detected"] is True
    assert page["ocr_applied"] is True
    assert "type 2 diabetes" in page["text"]
    assert result["warnings"] == []


def test_scrambled_page_without_ocr_is_warned(tmp_path):
    pdf_path = _pdf_with_text(tmp_path, _scramble(_READABLE))

    with _patch(
        "services.pdf_processor._ocr_pdf_page",
        side_effect=_PDFError("OCR requires the Tesseract OCR binary on PATH."),
    ):
        result = extract_pdf(str(pdf_path))

    page = result["pages"][0]
    assert page["garbled_text_detected"] is True
    assert page["ocr_applied"] is False
    assert any("scrambled" in w for w in result["warnings"])


def test_readable_page_is_not_sent_to_ocr(tmp_path):
    pdf_path = _pdf_with_text(tmp_path, _READABLE)

    with _patch("services.pdf_processor._ocr_pdf_page") as ocr:
        result = extract_pdf(str(pdf_path))

    ocr.assert_not_called()
    assert result["pages"][0]["garbled_text_detected"] is False
