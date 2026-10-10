from unittest.mock import patch

from services.protocol_pipeline import analyze_protocol_pdf
from models.protocol_models import ProtocolAnalysis


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
    }

    mock_analysis = ProtocolAnalysis(
        trial_title="Diabetes Study"
    )

    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value=sample_pdf_result,
    ), patch(
        "services.protocol_pipeline.analyze_protocol",
        return_value=mock_analysis,
    ) as mock_agent:

        result = analyze_protocol_pdf("test_protocol.pdf")

        assert result.trial_title == "Diabetes Study"

        text_sent_to_agent = mock_agent.call_args.args[0]

        assert "[SOURCE PAGE 1]" in text_sent_to_agent
        assert "[SOURCE PAGE 2]" in text_sent_to_agent
        assert "Age 18 to 65" in text_sent_to_agent