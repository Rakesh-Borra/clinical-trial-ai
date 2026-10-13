import json
from unittest.mock import patch

from agents.protocol_analysis_agent import analyze_protocol


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

    mock_response = {
        "message": {
            "content": json.dumps(
                {
                    "trial_title": "Timing Regression Study",
                    "protocol_id": "TIME-001",
                    "inclusion_criteria": [],
                    "exclusion_criteria": [],
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
                    "safety_information": [],
                }
            )
        }
    }

    with patch(
        "agents.protocol_analysis_agent.ollama.chat",
        return_value=mock_response,
    ) as mock_chat:
        analysis = analyze_protocol(protocol_text)

        prompt = mock_chat.call_args.kwargs["messages"][0]["content"]

        assert 'timing: the stated schedule' in prompt
        assert '"Day 0"' in prompt
        assert '"Day 30"' in prompt
        assert "Use null for timing only when" in prompt

        visits_by_name = {
            visit.visit_name: visit.timing for visit in analysis.visits
        }

        assert visits_by_name["Screening Visit"] == "Day 0"
        assert visits_by_name["Follow-up Visit"] == "Day 30"
        assert None not in visits_by_name.values()
