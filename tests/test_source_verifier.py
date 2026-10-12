
from models.protocol_models import (
    ProtocolAnalysis,
    EligibilityCriterion,
)
from services.source_verifier import verify_source_pages


def test_source_verifier_exact_match():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be 18 years or older.",
                source_page=1,
            )
        ]
    )

    pdf_result = {
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "INCLUSION CRITERIA:\n"
                    "Participants must be 18 years or older."
                ),
            }
        ]
    }

    results = verify_source_pages(analysis, pdf_result)

    assert len(results) == 1
    assert results[0]["status"] == "exact_match"


def test_source_verifier_wrong_page():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be 18 years or older.",
                source_page=2,
            )
        ]
    )

    pdf_result = {
        "pages": [
            {
                "page_number": 1,
                "text": "Participants must be 18 years or older.",
            },
            {
                "page_number": 2,
                "text": "Study procedures.",
            },
        ]
    }

    results = verify_source_pages(analysis, pdf_result)

    assert results[0]["status"] == "not_verified"
