from models.protocol_models import (
    ProtocolAnalysis,
    EligibilityCriterion,
)


def test_protocol_analysis_model():
    analysis = ProtocolAnalysis(
        trial_title="Example Diabetes Clinical Trial",
        protocol_id="TRIAL-001",
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be 18 years or older.",
                source_page=12,
            )
        ],
    )

    assert analysis.trial_title == "Example Diabetes Clinical Trial"
    assert len(analysis.inclusion_criteria) == 1
    assert analysis.inclusion_criteria[0].source_page == 12


def test_empty_protocol_analysis():
    analysis = ProtocolAnalysis()

    assert analysis.inclusion_criteria == []
    assert analysis.exclusion_criteria == []
    assert analysis.visits == []
    assert analysis.safety_information == []