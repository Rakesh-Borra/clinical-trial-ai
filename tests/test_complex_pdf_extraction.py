
from unittest.mock import patch

import pymupdf

from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    TrialVisit,
)
from services.pdf_processor import extract_pdf
from services.protocol_pipeline import analyze_protocol_pdf
from services.source_verifier import verify_source_pages


def _insert_lines(page, lines, start_y=72, line_height=16, fontsize=11):
    y = start_y
    for line in lines:
        page.insert_text((72, y), line, fontsize=fontsize)
        y += line_height
    return y


def _make_image_only_page(document, text, dpi=150):
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


def _build_complex_protocol_pdf(path):
    """
    Synthetic multi-page protocol with headings, paragraphs, a visit
    table, clinical values, and one scanned page. No patient data.
    """
    document = pymupdf.open()

    page1 = document.new_page()
    _insert_lines(
        page1,
        [
            "PROTOCOL TITLE: Complex Layout Diabetes Study",
            "PROTOCOL ID: CLX-205",
            "",
            "INCLUSION CRITERIA:",
            "1. Participants must be between 18 and 65 years old.",
            "2. HbA1c greater than or equal to 7.0%.",
            "3. eGFR at least 60 mL/min/1.73m2.",
            "",
            "EXCLUSION CRITERIA:",
            "1. Participants with severe kidney disease are excluded.",
            "2. eGFR below 30 mL/min/1.73m2.",
        ],
    )

    page2 = document.new_page()
    # Table-like visit schedule with spaced columns (reading order by y).
    _insert_lines(
        page2,
        [
            "VISIT SCHEDULE:",
            "Visit Name          Timing      Procedures",
            "Screening Visit     Day 0       Informed consent",
            "Baseline Visit      Day 1       Blood glucose testing",
            "Follow-up Visit     Day 30      Blood pressure measurement",
            "",
            "DOSAGE:",
            "Study drug 10 mg once daily with food.",
            "",
            "SAFETY INFORMATION:",
            "Participants will be monitored for adverse events.",
            "Hypoglycemia below 70 mg/dL requires clinical review.",
        ],
    )

    # Page 3: scanned image-only content.
    _make_image_only_page(
        document,
        "LAB THRESHOLDS: Fasting glucose 126 mg/dL or higher.",
    )

    document.save(path)
    document.close()


def test_complex_layout_preserves_clinical_values(tmp_path):
    pdf_path = tmp_path / "complex_protocol.pdf"
    _build_complex_protocol_pdf(pdf_path)

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="LAB THRESHOLDS: Fasting glucose 126 mg/dL or higher.",
    ):
        result = extract_pdf(str(pdf_path))

    assert result["total_pages"] == 3
    assert [page["page_number"] for page in result["pages"]] == [1, 2, 3]

    page1_text = result["pages"][0]["text"]
    page2_text = result["pages"][1]["text"]
    page3_text = result["pages"][2]["text"]

    # Reading order: title before inclusion criteria.
    assert page1_text.index("PROTOCOL TITLE") < page1_text.index(
        "INCLUSION CRITERIA"
    )
    assert page1_text.index("INCLUSION CRITERIA") < page1_text.index(
        "EXCLUSION CRITERIA"
    )

    # Clinical values preserved exactly.
    assert "18 and 65" in page1_text
    assert "7.0%" in page1_text
    assert "60 mL/min/1.73m2" in page1_text
    assert "30 mL/min/1.73m2" in page1_text

    assert "Screening Visit" in page2_text
    assert "Day 0" in page2_text
    assert "Day 1" in page2_text
    assert "Day 30" in page2_text
    assert page2_text.index("Day 0") < page2_text.index("Day 30")
    assert "10 mg once daily" in page2_text
    assert "70 mg/dL" in page2_text

    assert result["pages"][0]["ocr_applied"] is False
    assert result["pages"][1]["ocr_applied"] is False
    assert result["pages"][2]["ocr_applied"] is True
    assert "126 mg/dL" in page3_text
    assert result["warnings"] == []


def test_table_like_visit_schedule_reading_order(tmp_path):
    pdf_path = tmp_path / "visit_table.pdf"

    document = pymupdf.open()
    page = document.new_page()
    # Insert lower row first to prove sort=True restores reading order.
    page.insert_text((72, 140), "Follow-up Visit     Day 30")
    page.insert_text((72, 100), "Screening Visit     Day 0")
    page.insert_text((72, 72), "VISIT SCHEDULE:")
    document.save(pdf_path)
    document.close()

    result = extract_pdf(str(pdf_path))
    text = result["pages"][0]["text"]

    assert text.index("VISIT SCHEDULE") < text.index("Screening Visit")
    assert text.index("Screening Visit") < text.index("Follow-up Visit")
    assert text.index("Day 0") < text.index("Day 30")


def test_suspiciously_incomplete_text_warning(tmp_path):
    pdf_path = tmp_path / "incomplete_page.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Title only")
    # Embed an image so short selectable text looks incomplete.
    temp = pymupdf.open()
    src = temp.new_page()
    src.insert_text((72, 72), "Hidden body text with dosage 5 mg")
    pixmap = src.get_pixmap(dpi=120)
    temp.close()
    page.insert_image(pymupdf.Rect(72, 120, 500, 400), pixmap=pixmap)
    document.save(pdf_path)
    document.close()

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="Title only",
    ):
        result = extract_pdf(str(pdf_path))

    assert any(
        "suspiciously incomplete" in warning
        for warning in result["warnings"]
    )
    # Still does not invent the hidden OCR-only body if OCR returns little.
    assert "5 mg" not in result["pages"][0]["text"]


def test_source_page_attribution_for_complex_protocol(tmp_path):
    pdf_path = tmp_path / "attribution_protocol.pdf"
    _build_complex_protocol_pdf(pdf_path)

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="LAB THRESHOLDS: Fasting glucose 126 mg/dL or higher.",
    ):
        pdf_result = extract_pdf(str(pdf_path))

    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description=(
                    "Participants must be between 18 and 65 years old."
                ),
                source_page=1,
            )
        ],
        visits=[
            TrialVisit(
                visit_name="Screening Visit",
                timing="Day 0",
                source_page=2,
            )
        ],
    )

    verification = verify_source_pages(analysis, pdf_result)

    assert verification[0]["status"] == "exact_match"
    assert verification[0]["source_page"] == 1
    assert verification[1]["status"] == "exact_match"
    assert verification[1]["source_page"] == 2


def test_pipeline_source_markers_and_warnings_for_mixed_pdf(tmp_path):
    pdf_path = tmp_path / "pipeline_mixed.pdf"
    _build_complex_protocol_pdf(pdf_path)

    mock_analysis = ProtocolAnalysis(
        trial_title="Complex Layout Diabetes Study",
        inclusion_criteria=[
            EligibilityCriterion(
                description=(
                    "Participants must be between 18 and 65 years old."
                ),
                source_page=1,
            )
        ],
        visits=[
            TrialVisit(
                visit_name="Follow-up Visit",
                timing="Day 30",
                source_page=2,
            )
        ],
    )

    with patch(
        "services.pdf_processor._ocr_pdf_page",
        return_value="LAB THRESHOLDS: Fasting glucose 126 mg/dL or higher.",
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(mock_analysis, []),
    ) as mock_agent:
        result = analyze_protocol_pdf(str(pdf_path))

    prompt = mock_agent.call_args.args[0]
    assert "[SOURCE PAGE 1]" in prompt
    assert "[SOURCE PAGE 2]" in prompt
    assert "[SOURCE PAGE 3]" in prompt
    assert "7.0%" in prompt
    assert "10 mg once daily" in prompt
    assert "126 mg/dL" in prompt

    assert result["extraction_warnings"] == []
    assert all(
        item["status"] == "exact_match" for item in result["verification"]
    )
