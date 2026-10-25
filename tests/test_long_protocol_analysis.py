import re
from unittest.mock import patch

import pytest

from agents.protocol_analysis_agent import (
    EligibilityRescueExtraction,
    MetadataEligibilitySafetyExtraction,
    VisitsProceduresExtraction,
    _merge_chunk_analyses,
    analyze_protocol_in_chunks,
)
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from services.protocol_chunker import chunk_protocol
from services.protocol_pipeline import analyze_protocol_pdf


def _page_number(text):
    match = re.search(r"\[SOURCE PAGE (\d+)\]", text)
    return int(match.group(1)) if match else None


def test_long_analysis_processes_all_chunks_and_preserves_page_citations():
    protocol = (
        "[SOURCE PAGE 4]\nINCLUSION CRITERIA:\n"
        + "Age must be at least eighteen years.\n" * 5
        + "[SOURCE PAGE 9]\nEXCLUSION CRITERIA:\n"
        + "Prior condition is excluded from the study.\n" * 5
    )
    expected_chunks = chunk_protocol(protocol, max_chars=110).chunks
    pass_a_calls = []
    pass_b_calls = []

    def pass_a(text):
        page = _page_number(text)
        pass_a_calls.append(text)
        return MetadataEligibilitySafetyExtraction(
            inclusion_criteria=[
                EligibilityCriterion(
                    description=f"Criterion from source page {page}.",
                    source_page=page,
                )
            ],
        )

    def pass_b(text):
        page = _page_number(text)
        pass_b_calls.append(text)
        return VisitsProceduresExtraction(
            procedures=[
                TrialProcedure(
                    name=f"Procedure from source page {page}.",
                    source_page=page,
                )
            ],
        )

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        side_effect=pass_a,
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        side_effect=pass_b,
    ):
        analysis, warnings, failed, incomplete, calls = analyze_protocol_in_chunks(
            protocol,
            chunk_chars=110,
            max_input_chars=5000,
            max_calls=65,
            rescue_max_chars=1000,
        )

    assert len(expected_chunks) > 2
    assert len(pass_a_calls) == len(expected_chunks)
    assert len(pass_b_calls) == len(expected_chunks)
    assert calls == len(expected_chunks) * 2
    assert failed == []
    assert incomplete == []
    assert analysis.analysis_complete is True
    assert any("Oversized source pages were split" in warning for warning in warnings)
    assert {item.source_page for item in analysis.inclusion_criteria} <= {4, 9}
    assert {item.source_page for item in analysis.procedures} <= {4, 9}


def test_chunk_merge_preserves_clinically_distinct_payloads_and_pages():
    first = ProtocolAnalysis(
        trial_title="Long Protocol",
        inclusion_criteria=[
            EligibilityCriterion(description="HbA1c >= 7.0%.", source_page=3),
            EligibilityCriterion(description="HbA1c >= 7.0%.", source_page=3),
        ],
        safety_information=[
            SafetyInformation(description="Monitor liver enzymes.", source_page=8),
            SafetyInformation(description="Monitor liver enzymes.", source_page=8),
        ],
        procedures=[
            TrialProcedure(name="Study drug", description="10 mg daily.", source_page=12),
            TrialProcedure(name="Study drug", description="20 mg daily.", source_page=13),
        ],
        visits=[
            TrialVisit(visit_name="Follow-up", timing="Day 30", source_page=14),
            TrialVisit(visit_name="Follow-up", timing="Day 60", source_page=15),
            TrialVisit(
                visit_name="Follow-up",
                timing="Day 30",
                procedures=["Blood draw"],
                source_page=14,
            ),
        ],
    )
    second = ProtocolAnalysis(
        trial_title="Long Protocol",
        inclusion_criteria=[
            EligibilityCriterion(description="HbA1c >= 7.0%.", source_page=4),
        ],
        procedures=[
            TrialProcedure(name="Study drug", description="10 mg daily.", source_page=12),
        ],
    )

    merged, warnings = _merge_chunk_analyses([first, second])

    assert any("Conflicting procedure details" in w for w in warnings)
    assert any("Conflicting visit schedule details" in w for w in warnings)
    assert [(x.description, x.source_page) for x in merged.inclusion_criteria] == [
        ("HbA1c >= 7.0%.", 3),
        ("HbA1c >= 7.0%.", 4),
    ]
    assert [(x.description, x.source_page) for x in merged.safety_information] == [
        ("Monitor liver enzymes.", 8),
    ]
    assert {
        (x.description, x.source_page) for x in merged.procedures
    } == {("10 mg daily.", 12), ("20 mg daily.", 13)}
    assert {
        (x.timing, tuple(x.procedures), x.source_page) for x in merged.visits
    } == {
        ("Day 30", (), 14),
        ("Day 60", (), 15),
        ("Day 30", ("Blood draw",), 14),
    }


def test_failed_chunk_pass_is_reported_and_successful_pass_retained():
    protocol = (
        "[SOURCE PAGE 2]\nINCLUSION CRITERIA:\n"
        + "Age must be at least eighteen years.\n" * 5
        + "[SOURCE PAGE 5]\nMore source text.\n"
        + "This line is long enough to create another chunk.\n" * 5
    )

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        side_effect=[
            RuntimeError("simulated pass A failure"),
            RuntimeError("simulated pass A failure"),
            MetadataEligibilitySafetyExtraction(
                inclusion_criteria=[
                    EligibilityCriterion(
                        description="Age 18 or older.",
                        source_page=5,
                    )
                ],
            ),
            MetadataEligibilitySafetyExtraction(),
        ],
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ):
        _analysis, warnings, failed, incomplete, calls = (
            analyze_protocol_in_chunks(
                protocol,
                chunk_chars=100,
                max_input_chars=5000,
                max_calls=65,
                rescue_max_chars=1000,
                pass_retries=1,
            )
        )

    assert calls > 0
    assert failed
    assert failed[0]["failed_passes"] == [
        "Pass A failed: simulated pass A failure"
    ]
    assert any("Chunk 1" in warning and "Pass A failed" in warning for warning in warnings)
    assert any("did not complete both extraction passes" in reason for reason in incomplete)


def test_failed_pass_retries_once_then_succeeds_without_duplicate():
    protocol = (
        "[SOURCE PAGE 2]\nINCLUSION CRITERIA:\n"
        + "Age must be at least eighteen years.\n" * 5
    )

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        side_effect=[
            RuntimeError("transient failure"),
            MetadataEligibilitySafetyExtraction(
                inclusion_criteria=[
                    EligibilityCriterion(
                        description="Age 18 or older.",
                        source_page=2,
                    )
                ],
            ),
        ],
    ) as pass_a, patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ):
        analysis, warnings, failed, incomplete, calls = (
            analyze_protocol_in_chunks(
                protocol,
                chunk_chars=500,
                max_input_chars=5000,
                max_calls=65,
                rescue_max_chars=1000,
                pass_retries=1,
            )
        )

    assert pass_a.call_count == 2
    assert calls >= 3  # pass A twice + pass B once
    assert failed == []
    assert incomplete == []
    assert len(analysis.inclusion_criteria) == 1
    assert any("retrying" in warning for warning in warnings)


def test_chunk_and_input_caps_fail_before_any_extraction_calls():
    protocol = "[SOURCE PAGE 1]\n" + "Long source text. " * 100

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety"
    ) as pass_a, patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures"
    ) as pass_b:
        with pytest.raises(ValueError, match="overall resource cap"):
            analyze_protocol_in_chunks(
                protocol,
                chunk_chars=100,
                max_input_chars=50,
                max_calls=65,
                rescue_max_chars=100,
            )
        with pytest.raises(ValueError, match="exceeding the configured cap"):
            analyze_protocol_in_chunks(
                protocol,
                chunk_chars=100,
                max_input_chars=5000,
                max_calls=3,
                rescue_max_chars=100,
            )

    pass_a.assert_not_called()
    pass_b.assert_not_called()


def test_eligibility_rescue_runs_once_after_merge_and_within_budget():
    protocol = (
        "[SOURCE PAGE 1]\nINCLUSION CRITERIA:\nAge 18 or older.\n"
        "[SOURCE PAGE 2]\nEXCLUSION CRITERIA:\nPrior disease."
    )
    rescue_result = {
        "inclusion_criteria": [
            {"description": "Age 18 or older.", "source_page": 1},
        ],
        "exclusion_criteria": [
            {"description": "Prior disease.", "source_page": 2},
        ],
    }

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        return_value=MetadataEligibilitySafetyExtraction(),
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ), patch(
        "agents.protocol_analysis_agent._pass_eligibility_rescue",
        return_value=EligibilityRescueExtraction.model_validate(rescue_result),
    ) as rescue:
        analysis, warnings, failed, incomplete, calls = (
            analyze_protocol_in_chunks(
                protocol,
                chunk_chars=1000,
                max_input_chars=5000,
                max_calls=3,
                rescue_max_chars=1000,
            )
        )

    rescue.assert_called_once()
    assert calls == 3
    assert failed == []
    assert incomplete == []
    assert len(analysis.inclusion_criteria) == 1
    assert len(analysis.exclusion_criteria) == 1
    assert any("ran once" in warning for warning in warnings)


def test_over_budget_eligibility_rescue_is_reported_without_call():
    protocol = (
        "[SOURCE PAGE 1]\nINCLUSION CRITERIA:\n"
        + "Age criteria listed in this eligibility section.\n" * 10
    )

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        return_value=MetadataEligibilitySafetyExtraction(),
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ), patch(
        "agents.protocol_analysis_agent._pass_eligibility_rescue"
    ) as rescue:
        _analysis, warnings, _failed, incomplete, calls = (
            analyze_protocol_in_chunks(
                protocol,
                chunk_chars=1000,
                max_input_chars=5000,
                max_calls=3,
                rescue_max_chars=20,
            )
        )

    rescue.assert_not_called()
    assert calls == 2
    assert any("Eligibility rescue was skipped" in warning for warning in warnings)
    assert any("Eligibility rescue was skipped" in reason for reason in incomplete)


def test_pipeline_uses_legacy_short_path_and_long_path_marks_unreadable():
    short_pages = [
        {
            "page_number": 1,
            "text": "Short protocol text.",
            "requires_ocr": False,
        },
    ]
    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value={"pages": short_pages, "warnings": []},
    ), patch(
        "services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS",
        1000,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(ProtocolAnalysis(), []),
    ) as legacy, patch(
        "services.protocol_pipeline.analyze_protocol_in_chunks"
    ) as chunked:
        short_result = analyze_protocol_pdf("short.pdf")

    legacy.assert_called_once()
    chunked.assert_not_called()
    assert short_result["analysis_complete"] is True
    assert short_result["analysis"].analysis_complete is True

    long_pages = [
        {
            "page_number": 1,
            "text": "PROTOCOL TITLE: Long Study\n" + "Long content. " * 30,
            "requires_ocr": False,
        },
        {
            "page_number": 2,
            "text": "",
            "requires_ocr": True,
        },
    ]
    long_analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Long content.",
                source_page=1,
            )
        ]
    )
    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value={
            "pages": long_pages,
            "warnings": ["Page 2: OCR returned no text."],
        },
    ), patch(
        "services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS",
        100,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_in_chunks",
        return_value=(long_analysis, [], [], [], 2),
    ) as chunked:
        long_result = analyze_protocol_pdf("long.pdf")

    sent_text = chunked.call_args.args[0]
    assert "[SOURCE PAGE 1]" in sent_text
    assert "[SOURCE PAGE 2]\n" in sent_text
    assert long_result["analysis_complete"] is False
    assert long_result["analysis"].analysis_complete is False
    assert long_result["incomplete_chunks"] == []
    assert any("Pages with no extractable text" in w for w in long_result["accuracy_warnings"])
    assert any("OCR returned no text" in w for w in long_result["incomplete_reasons"])
    assert long_result["verification"][0]["source_page"] == 1
