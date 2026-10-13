
import json
from unittest.mock import patch

from agents.protocol_analysis_agent import (
    EligibilityRescueExtraction,
    analyze_protocol_with_rescue,
    merge_eligibility_rescue,
    maybe_run_eligibility_rescue,
)
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
)
from services.eligibility_rescue import (
    missing_clinical_eligibility_markers,
    select_eligibility_sections,
    should_trigger_eligibility_rescue,
)


COMPLEX_PROTOCOL = """
[SOURCE PAGE 1]
PROTOCOL TITLE: Complex Layout Diabetes Study
PROTOCOL ID: CLX-205

INCLUSION CRITERIA:
1. Participants must be between 18 and 65 years old.
2. HbA1c greater than or equal to 7.0%.
3. eGFR at least 60 mL/min/1.73m2.

EXCLUSION CRITERIA:
1. eGFR below 30 mL/min/1.73m2.

[SOURCE PAGE 2]
VISIT SCHEDULE:
Screening Visit: Day 0.
SAFETY INFORMATION:
Participants will be monitored for adverse events.
"""


def test_select_eligibility_sections_preserves_page_markers():
    selected, notes = select_eligibility_sections(COMPLEX_PROTOCOL)

    assert notes == []
    assert selected is not None
    assert "[SOURCE PAGE 1]" in selected
    assert "INCLUSION CRITERIA" in selected
    assert "VISIT SCHEDULE" not in selected


def test_select_eligibility_sections_reports_limitation_when_missing():
    text = """
    [SOURCE PAGE 1]
    PROTOCOL TITLE: No Eligibility Here
    PROTOCOL ID: NE-1
    """
    selected, notes = select_eligibility_sections(text)

    assert selected is None
    assert any("could not confidently locate" in note for note in notes)


def test_rescue_triggered_for_missing_inclusions():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[],
        exclusion_criteria=[
            EligibilityCriterion(description="eGFR below 30.", source_page=1)
        ],
    )

    assert should_trigger_eligibility_rescue(analysis, COMPLEX_PROTOCOL) is True


def test_rescue_triggered_for_missing_hba1c_egfr():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be between 18 and 65 years old.",
                source_page=1,
            )
        ]
    )

    missing = missing_clinical_eligibility_markers(analysis, COMPLEX_PROTOCOL)
    assert "HbA1c" in missing
    assert "eGFR" in missing
    assert should_trigger_eligibility_rescue(analysis, COMPLEX_PROTOCOL) is True


def test_rescue_skipped_when_criteria_complete():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be between 18 and 65 years old.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="HbA1c greater than or equal to 7.0%.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="eGFR at least 60 mL/min/1.73m2.",
                source_page=1,
            ),
        ],
        exclusion_criteria=[
            EligibilityCriterion(
                description="eGFR below 30 mL/min/1.73m2.",
                source_page=1,
            )
        ],
    )

    assert should_trigger_eligibility_rescue(analysis, COMPLEX_PROTOCOL) is False

    with patch(
        "agents.protocol_analysis_agent._pass_eligibility_rescue",
    ) as mock_rescue:
        updated, notes = maybe_run_eligibility_rescue(
            analysis,
            COMPLEX_PROTOCOL,
        )

    mock_rescue.assert_not_called()
    assert updated.inclusion_criteria == analysis.inclusion_criteria
    assert any("criteria appear complete" in note for note in notes)


def test_merge_eligibility_rescue_deduplicates():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="HbA1c greater than or equal to 7.0%.",
                source_page=1,
            )
        ]
    )
    rescue = EligibilityRescueExtraction(
        inclusion_criteria=[
            EligibilityCriterion(
                description="HbA1c greater than or equal to 7.0%.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="eGFR at least 60 mL/min/1.73m2.",
                source_page=1,
            ),
        ],
        exclusion_criteria=[
            EligibilityCriterion(
                description="eGFR below 30 mL/min/1.73m2.",
                source_page=1,
            )
        ],
    )

    merged = merge_eligibility_rescue(analysis, rescue)

    assert len(merged.inclusion_criteria) == 2
    assert len(merged.exclusion_criteria) == 1


def test_maybe_run_rescue_merges_when_inclusions_empty():
    analysis = ProtocolAnalysis(inclusion_criteria=[])
    rescue_payload = {
        "inclusion_criteria": [
            {
                "description": "HbA1c greater than or equal to 7.0%.",
                "source_page": 1,
            },
            {
                "description": "eGFR at least 60 mL/min/1.73m2.",
                "source_page": 1,
            },
        ],
        "exclusion_criteria": [
            {
                "description": "eGFR below 30 mL/min/1.73m2.",
                "source_page": 1,
            }
        ],
    }

    with patch(
        "agents.protocol_analysis_agent._pass_eligibility_rescue",
        return_value=EligibilityRescueExtraction.model_validate(rescue_payload),
    ) as mock_rescue:
        updated, notes = maybe_run_eligibility_rescue(
            analysis,
            COMPLEX_PROTOCOL,
        )

    mock_rescue.assert_called_once()
    prompt_text = mock_rescue.call_args.args[0]
    assert "[SOURCE PAGE 1]" in prompt_text
    assert "VISIT SCHEDULE" not in prompt_text
    assert len(updated.inclusion_criteria) == 2
    assert any("Eligibility rescue triggered" in note for note in notes)


def test_analyze_protocol_with_rescue_three_calls_when_needed():
    pass_a = {
        "message": {
            "content": json.dumps(
                {
                    "trial_title": "Complex Layout Diabetes Study",
                    "protocol_id": "CLX-205",
                    "inclusion_criteria": [],
                    "exclusion_criteria": [],
                    "safety_information": [],
                }
            )
        }
    }
    pass_b = {
        "message": {
            "content": json.dumps(
                {
                    "procedures": [],
                    "visits": [
                        {
                            "visit_name": "Screening Visit",
                            "timing": "Day 0",
                            "procedures": [],
                            "source_page": 2,
                        }
                    ],
                }
            )
        }
    }
    rescue = {
        "message": {
            "content": json.dumps(
                {
                    "inclusion_criteria": [
                        {
                            "description": (
                                "HbA1c greater than or equal to 7.0%."
                            ),
                            "source_page": 1,
                        }
                    ],
                    "exclusion_criteria": [],
                }
            )
        }
    }

    with patch(
        "agents.protocol_analysis_agent.ollama.chat",
        side_effect=[pass_a, pass_b, rescue],
    ) as mock_chat:
        analysis, notes = analyze_protocol_with_rescue(COMPLEX_PROTOCOL)

    assert mock_chat.call_count == 3
    assert analysis.inclusion_criteria[0].description.startswith("HbA1c")
    assert any("rescue triggered" in note for note in notes)
