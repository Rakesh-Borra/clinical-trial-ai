import json
from unittest.mock import patch

from agents.protocol_analysis_agent import (
    MetadataEligibilitySafetyExtraction,
    VisitsProceduresExtraction,
    analyze_protocol,
    merge_focused_extractions,
)
from models.protocol_models import (
    EligibilityCriterion,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)


def _pass_a_response(**overrides):
    payload = {
        "trial_title": "Timing Regression Study",
        "protocol_id": "TIME-001",
        "inclusion_criteria": [],
        "exclusion_criteria": [],
        "safety_information": [],
    }
    payload.update(overrides)
    return {"message": {"content": json.dumps(payload)}}


def _pass_b_response(**overrides):
    payload = {
        "procedures": [],
        "visits": [
            {
                "visit_name": "Screening Visit",
                "timing": "Day 0",
                "procedures": [],
                "source_page": 1,
            },
            {
                "visit_name": "Follow-up Visit",
                "timing": "Day 30",
                "procedures": [],
                "source_page": 1,
            },
        ],
    }
    payload.update(overrides)
    return {"message": {"content": json.dumps(payload)}}


def test_analyze_protocol_extracts_visit_timings():
    """Regression: visit schedule days must populate timing, not null."""

    protocol_text = """
    [SOURCE PAGE 1]
    PROTOCOL TITLE: Timing Regression Study
    PROTOCOL ID: TIME-001

    VISIT SCHEDULE:
    Screening Visit: Day 0.
    Follow-up Visit: Day 30.
    """

    with patch("services.model_provider.ollama.Client") as mock_client:
        mock_chat = mock_client.return_value.chat
        mock_chat.side_effect = [_pass_a_response(), _pass_b_response()]
        analysis = analyze_protocol(protocol_text)

        assert mock_chat.call_count == 2
        pass_b_prompt = mock_chat.call_args_list[1].kwargs["messages"][0]["content"]

        assert "Pass B" in pass_b_prompt
        # Misleading few-shot values made the model invent visits and tests.
        for invented in ("Day 30", "blood glucose", "blood pressure", "Blood pressure"):
            assert invented not in pass_b_prompt.split("PROTOCOL TEXT:")[0]
        assert "Never leave visit timing null" in pass_b_prompt

        visits_by_name = {
            visit.visit_name: visit.timing for visit in analysis.visits
        }

        assert visits_by_name["Screening Visit"] == "Day 0"
        assert visits_by_name["Follow-up Visit"] == "Day 30"
        assert None not in visits_by_name.values()


def test_prompt_requires_grounded_metadata_and_lab_criteria():
    protocol_text = """
    [SOURCE PAGE 1]
    PROTOCOL TITLE: Prompt Guidance Study
    PROTOCOL ID: PGS-001
    """

    with patch("services.model_provider.ollama.Client") as mock_client:
        mock_chat = mock_client.return_value.chat
        mock_chat.side_effect = [
            _pass_a_response(
                trial_title="Prompt Guidance Study",
                protocol_id="PGS-001",
            ),
            _pass_b_response(visits=[]),
        ]
        analyze_protocol(protocol_text)
        pass_a_prompt = mock_chat.call_args_list[0].kwargs["messages"][0]["content"]
        pass_b_prompt = mock_chat.call_args_list[1].kwargs["messages"][0]["content"]

    assert "Pass A" in pass_a_prompt
    assert 'Never invent titles such as "Example Trial"' in pass_a_prompt
    assert "Never invent IDs" in pass_a_prompt
    assert "HbA1c" in pass_a_prompt
    assert "eGFR" in pass_a_prompt
    assert "CLX-205" not in pass_a_prompt
    assert "Do not omit lab-based eligibility criteria" in pass_a_prompt
    assert "[SOURCE PAGE 1]" in pass_a_prompt
    assert "Pass B" in pass_b_prompt
    assert "Do not extract trial_title" in pass_b_prompt


def test_two_pass_orchestration_merges_results():
    protocol_text = "[SOURCE PAGE 1]\nPROTOCOL TITLE: Merge Study\n"

    with patch("services.model_provider.ollama.Client") as mock_client:
        mock_chat = mock_client.return_value.chat
        mock_chat.side_effect = [
            _pass_a_response(
                trial_title="Merge Study",
                protocol_id="MRG-1",
                inclusion_criteria=[
                    {
                        "description": "Age 18 or older.",
                        "source_page": 1,
                    }
                ],
                safety_information=[
                    {
                        "description": "Monitor adverse events.",
                        "source_page": 1,
                    }
                ],
            ),
            _pass_b_response(
                procedures=[
                    {
                        "name": "Blood glucose testing.",
                        "description": None,
                        "source_page": 1,
                    }
                ],
            ),
        ]
        analysis = analyze_protocol(protocol_text)

    assert mock_chat.call_count == 2
    assert analysis.trial_title == "Merge Study"
    assert analysis.protocol_id == "MRG-1"
    assert len(analysis.inclusion_criteria) == 1
    assert len(analysis.safety_information) == 1
    assert len(analysis.procedures) == 1
    assert len(analysis.visits) == 2


def test_merge_focused_extractions_deduplicates():
    pass_a = MetadataEligibilitySafetyExtraction(
        trial_title="Dedupe Study",
        protocol_id="DED-1",
        inclusion_criteria=[
            EligibilityCriterion(description="HbA1c >= 7.0%.", source_page=1),
            EligibilityCriterion(description="HbA1c >= 7.0%.", source_page=1),
            EligibilityCriterion(description="eGFR >= 60.", source_page=1),
        ],
        safety_information=[
            SafetyInformation(description="Monitor AEs.", source_page=2),
            SafetyInformation(description="Monitor AEs.", source_page=2),
        ],
    )
    pass_b = VisitsProceduresExtraction(
        procedures=[
            TrialProcedure(name="Blood glucose testing.", source_page=2),
            TrialProcedure(name="Blood glucose testing.", source_page=2),
        ],
        visits=[
            TrialVisit(visit_name="Screening Visit", timing="Day 0", source_page=2),
            TrialVisit(visit_name="Screening Visit", timing="Day 0", source_page=2),
            TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=2),
        ],
    )

    merged = merge_focused_extractions(pass_a, pass_b)

    assert len(merged.inclusion_criteria) == 2
    assert len(merged.safety_information) == 1
    assert len(merged.procedures) == 1
    assert len(merged.visits) == 2
    assert merged.visits[0].timing == "Day 0"
    assert merged.visits[1].timing == "Day 30"
