from unittest.mock import patch

import pytest

from services.protocol_pipeline import analyze_protocol_pdf
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
)


def test_protocol_pdf_pipeline():
    sample_pdf_result = {
        "filename": "test_protocol.pdf",
        "total_pages": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "PROTOCOL TITLE: Diabetes Study",
                "character_count": 30,
                "requires_ocr": False,
            },
            {
                "page_number": 2,
                "text": "INCLUSION: Age 18 to 65",
                "character_count": 23,
                "requires_ocr": False,
            },
        ],
        "warnings": [],
    }

    mock_analysis = ProtocolAnalysis(
        trial_title="Diabetes Study"
    )

    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value=sample_pdf_result,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(mock_analysis, []),
    ) as mock_agent:

        result = analyze_protocol_pdf("test_protocol.pdf")

        assert result["analysis"].trial_title == "Diabetes Study"
        assert isinstance(result["verification"], list)
        assert result["extraction_warnings"] == []
        assert result["accuracy_warnings"] == []
        text_sent_to_agent = mock_agent.call_args.args[0]

        assert "[SOURCE PAGE 1]" in text_sent_to_agent
        assert "[SOURCE PAGE 2]" in text_sent_to_agent
        assert "Age 18 to 65" in text_sent_to_agent


def test_pipeline_applies_accuracy_checks():
    sample_pdf_result = {
        "filename": "accuracy_protocol.pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "PROTOCOL TITLE: Accuracy Study\n"
                    "PROTOCOL ID: ACC-001\n"
                    "INCLUSION CRITERIA:\n"
                    "1. Participants must be 18 years or older.\n"
                    "2. HbA1c greater than or equal to 7.0%.\n"
                    "3. eGFR at least 60 mL/min/1.73m2."
                ),
                "character_count": 180,
                "requires_ocr": False,
                "ocr_applied": False,
            }
        ],
        "warnings": [],
    }

    mock_analysis = ProtocolAnalysis(
        trial_title="Example Trial",
        protocol_id="ACC-001",
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be 18 years or older.",
                source_page=None,
            )
        ],
    )

    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value=sample_pdf_result,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(mock_analysis, ["Eligibility rescue skipped: criteria appear complete."]),
    ):
        result = analyze_protocol_pdf("accuracy_protocol.pdf")

    # Invented title grounded to null, then recovered from PROTOCOL TITLE.
    assert result["analysis"].trial_title == "Accuracy Study"
    assert result["analysis"].protocol_id == "ACC-001"
    assert result["analysis"].inclusion_criteria[0].source_page == 1
    assert result["verification"][0]["status"] == "exact_match"
    assert any("trial_title" in w for w in result["accuracy_warnings"])
    assert any(
        "recovered from labeled source field" in w
        for w in result["accuracy_warnings"]
    )
    assert any("HbA1c" in w for w in result["accuracy_warnings"])
    assert any("eGFR" in w for w in result["accuracy_warnings"])
    assert result["extraction_warnings"] == []


def test_pipeline_propagates_extraction_warnings():
    sample_pdf_result = {
        "filename": "warn_protocol.pdf",
        "total_pages": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "PROTOCOL TITLE: Warning Study",
                "character_count": 29,
                "requires_ocr": False,
                "ocr_applied": False,
            },
            {
                "page_number": 2,
                "text": "",
                "character_count": 0,
                "requires_ocr": True,
                "ocr_applied": False,
            },
        ],
        "warnings": [
            "Page 2: no selectable text and OCR returned no text.",
        ],
    }

    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value=sample_pdf_result,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(ProtocolAnalysis(trial_title="Warning Study"), []),
    ):
        result = analyze_protocol_pdf("warn_protocol.pdf")

    assert result["extraction_warnings"] == sample_pdf_result["warnings"]
    assert "OCR returned no text" in result["extraction_warnings"][0]


def test_pipeline_includes_warnings_when_no_text():
    sample_pdf_result = {
        "filename": "empty_protocol.pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "",
                "character_count": 0,
                "requires_ocr": True,
                "ocr_applied": False,
            }
        ],
        "warnings": [
            "Page 1: OCR requires the Tesseract OCR binary on PATH.",
        ],
    }

    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value=sample_pdf_result,
    ):
        with pytest.raises(ValueError) as exc_info:
            analyze_protocol_pdf("empty_protocol.pdf")

    message = str(exc_info.value)
    assert "no extractable text" in message
    assert "Tesseract" in message


def test_pipeline_preserves_rescue_notes_in_accuracy_warnings():
    sample_pdf_result = {
        "filename": "rescue_notes.pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "PROTOCOL TITLE: Rescue Note Study",
                "character_count": 40,
                "requires_ocr": False,
            }
        ],
        "warnings": [],
    }

    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value=sample_pdf_result,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(
            ProtocolAnalysis(trial_title="Rescue Note Study"),
            ["Eligibility rescue triggered: running eligibility-only extraction on selected source pages."],
        ),
    ):
        result = analyze_protocol_pdf("rescue_notes.pdf")

    assert any(
        "Eligibility rescue triggered" in warning
        for warning in result["accuracy_warnings"]
    )
