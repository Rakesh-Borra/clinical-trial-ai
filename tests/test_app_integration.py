"""
Mocked Streamlit integration tests for Phase 6 app wiring.

Does not call Ollama or write durable artifacts beyond temp cleanup checks.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.coordinator_models import CoordinatorAnswer, CoordinatorCitation
from models.patient_models import PatientAnswer, PatientCitation
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)


def _sample_pipeline_result() -> dict:
    analysis = ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be between 18 and 65 years old.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="HbA1c greater than or equal to 7.0%.",
                source_page=1,
            ),
        ],
        exclusion_criteria=[
            EligibilityCriterion(
                description="eGFR below 30 mL/min/1.73m2.",
                source_page=1,
            )
        ],
        procedures=[
            TrialProcedure(name="Blood glucose testing.", source_page=2),
        ],
        visits=[
            TrialVisit(
                visit_name="Screening Visit",
                timing="Day 0",
                source_page=2,
            ),
            TrialVisit(
                visit_name="Follow-up Visit",
                timing="Day 30",
                source_page=2,
            ),
        ],
        safety_information=[
            SafetyInformation(
                description="Participants will be monitored for adverse events.",
                source_page=2,
            )
        ],
    )
    verification = [
        {
            "category": "inclusion_criteria",
            "statement": "Participants must be between 18 and 65 years old.",
            "source_page": 1,
            "status": "exact_match",
        },
        {
            "category": "inclusion_criteria",
            "statement": "HbA1c greater than or equal to 7.0%.",
            "source_page": 1,
            "status": "exact_match",
        },
        {
            "category": "exclusion_criteria",
            "statement": "eGFR below 30 mL/min/1.73m2.",
            "source_page": 1,
            "status": "exact_match",
        },
        {
            "category": "procedures",
            "statement": "Blood glucose testing.",
            "source_page": 2,
            "status": "exact_match",
        },
        {
            "category": "visits",
            "statement": "Screening Visit",
            "source_page": 2,
            "status": "exact_match",
        },
        {
            "category": "visits",
            "statement": "Follow-up Visit",
            "source_page": 2,
            "status": "exact_match",
        },
        {
            "category": "safety_information",
            "statement": "Participants will be monitored for adverse events.",
            "source_page": 2,
            "status": "exact_match",
        },
    ]
    return {
        "analysis": analysis,
        "verification": verification,
        "extraction_warnings": [],
        "accuracy_warnings": [
            "Eligibility rescue skipped: criteria appear complete.",
        ],
    }


def test_research_disclaimer_present():
    import app

    assert "educational" in app.RESEARCH_DISCLAIMER.casefold()
    assert "medical advice" in app.RESEARCH_DISCLAIMER.casefold()
    assert "eligible" in app.RESEARCH_DISCLAIMER.casefold()


def test_persist_uploaded_pdf_and_cleanup(tmp_path):
    import app

    payload = b"%PDF-1.4 fake protocol content"
    uploaded = SimpleNamespace(
        name="sample.pdf",
        size=len(payload),
        getbuffer=lambda: memoryview(payload),
    )

    path = app.persist_uploaded_pdf(uploaded)
    try:
        assert path.endswith(".pdf")
        with open(path, "rb") as handle:
            assert handle.read() == payload
    finally:
        app.cleanup_temp_pdf(path)
        assert not __import__("os").path.exists(path)


def test_upload_fingerprint_changes_with_file():
    import app

    a = SimpleNamespace(name="a.pdf", size=10)
    b = SimpleNamespace(name="b.pdf", size=10)
    c = SimpleNamespace(name="a.pdf", size=11)
    assert app.upload_fingerprint(a) != app.upload_fingerprint(b)
    assert app.upload_fingerprint(a) != app.upload_fingerprint(c)


def test_format_analysis_error_invalid_pdf_and_ollama():
    import app

    empty = app.format_analysis_error(
        ValueError("The protocol PDF contains no extractable text")
    )
    assert "Could not analyze this PDF" in empty

    ollama = app.format_analysis_error(
        ConnectionError("Failed to connect to Ollama at http://127.0.0.1:11434")
    )
    assert "Ollama" in ollama
    assert "qwen3:4b" in ollama


def test_provider_display_name_is_safe_and_configuration_aware(monkeypatch):
    from services.model_provider import provider_display_name

    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    assert provider_display_name() == "OpenRouter (cloud)"
    assert "secret-test-key" not in provider_display_name()

    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "invalid")
    assert provider_display_name() == "Unconfigured model provider"


def test_incomplete_and_unknown_analysis_status_fail_closed():
    import app

    complete, complete_message = app.analysis_completeness_message(
        {"analysis_complete": True}
    )
    incomplete, incomplete_message = app.analysis_completeness_message(
        {
            "analysis_complete": False,
            "incomplete_chunks": [2, 3],
            "incomplete_reasons": ["OCR was incomplete."],
        }
    )
    unknown, unknown_message = app.analysis_completeness_message({})

    assert complete
    assert "Analysis complete" in complete_message
    assert not incomplete
    assert "2 chunk(s) did not complete" in incomplete_message
    assert "OCR was incomplete." in incomplete_message
    assert not unknown
    assert "could not verify" in unknown_message


def test_run_protocol_analysis_delegates_to_pipeline():
    import app

    expected = _sample_pipeline_result()
    with patch("app.analyze_protocol_pdf", return_value=expected) as mocked:
        result = app.run_protocol_analysis("/tmp/fake.pdf")
    mocked.assert_called_once_with("/tmp/fake.pdf")
    assert result is expected


def test_analysis_is_cached_once_per_upload():
    """Simulate analyze-once caching: second ask must not re-call pipeline."""
    import app

    pipeline = _sample_pipeline_result()
    call_count = {"n": 0}

    def fake_analyze(_path: str):
        call_count["n"] += 1
        return pipeline

    with patch("app.analyze_protocol_pdf", side_effect=fake_analyze):
        first = app.run_protocol_analysis("/tmp/protocol.pdf")
        # App caches in session_state; assistants consume the cached dict.
        second_coord = __import__(
            "services.coordinator_query", fromlist=["answer_from_pipeline_result"]
        ).answer_from_pipeline_result("What are the inclusion criteria?", first)
        second_patient = __import__(
            "services.patient_query", fromlist=["answer_patient_from_pipeline_result"]
        ).answer_patient_from_pipeline_result("Who can join this study?", first)

    assert call_count["n"] == 1
    assert "7.0%" in second_coord.answer_text
    assert "7.0%" in second_patient.answer_text
    assert second_coord.citations
    assert second_patient.audience == "patient"


def test_coordinator_and_patient_use_cached_pipeline_result():
    from services.coordinator_query import answer_from_pipeline_result
    from services.patient_query import answer_patient_from_pipeline_result

    pipeline = _sample_pipeline_result()

    with patch(
        "services.coordinator_query.answer_coordinator_query"
    ) as coord_mock, patch(
        "services.patient_query.answer_patient_query"
    ) as patient_mock:
        coord_mock.return_value = CoordinatorAnswer(
            answer_text="Inclusion facts",
            intent="eligibility_inclusion",
            citations=[
                CoordinatorCitation(
                    category="inclusion_criteria",
                    statement="HbA1c greater than or equal to 7.0%.",
                    source_page=1,
                    verification_status="exact_match",
                )
            ],
            confidence="high",
            limitations=[],
            disclaimers=["Coordinator disclaimer"],
        )
        patient_mock.return_value = PatientAnswer(
            answer_text="Patient-friendly inclusion facts",
            intent="eligibility_inclusion",
            citations=[
                PatientCitation(
                    category="inclusion_criteria",
                    statement="HbA1c greater than or equal to 7.0%.",
                    source_page=1,
                    verification_status="exact_match",
                    plain_language='One requirement is: "HbA1c greater than or equal to 7.0%."',
                )
            ],
            confidence="high",
            limitations=[],
            disclaimers=["Patient disclaimer"],
            audience="patient",
        )

        coord = answer_from_pipeline_result("inclusion?", pipeline)
        patient = answer_patient_from_pipeline_result("Who can join?", pipeline)

    coord_mock.assert_called_once()
    patient_mock.assert_called_once()
    assert coord.confidence == "high"
    assert patient.audience == "patient"
    assert coord_mock.call_args.args[1] is pipeline["analysis"]
    assert patient_mock.call_args.args[1] is pipeline["analysis"]


def test_graceful_error_when_pipeline_raises():
    import app

    with patch(
        "app.analyze_protocol_pdf",
        side_effect=ValueError("The protocol PDF contains no extractable text"),
    ):
        with pytest.raises(ValueError):
            app.run_protocol_analysis("/tmp/bad.pdf")

    message = app.format_analysis_error(
        ValueError("The protocol PDF contains no extractable text")
    )
    assert "Could not analyze this PDF" in message


def test_session_helpers_clear_cached_analysis():
    import app

    class _Session(dict):
        """Minimal stand-in for st.session_state attribute access."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

        def get(self, key, default=None):
            return dict.get(self, key, default)

    fake_state = _Session(
        pipeline_result=_sample_pipeline_result(),
        analysis_error="old",
        temp_pdf_path=None,
        uploaded_name="x.pdf",
        uploaded_fingerprint="x.pdf:1",
        coordinator_history=[{"q": 1}],
        patient_history=[{"q": 1}],
    )

    mock_st = MagicMock()
    mock_st.session_state = fake_state

    with patch.object(app, "st", mock_st), patch.object(
        app, "cleanup_temp_pdf"
    ) as cleanup:
        app.clear_analysis_state(keep_upload=False)

    assert fake_state.pipeline_result is None
    assert fake_state.analysis_error is None
    assert fake_state.coordinator_history == []
    assert fake_state.patient_history == []
    assert fake_state.uploaded_name is None
    cleanup.assert_called_once_with(None)


def test_verification_lookup_indexes_rows():
    import app

    index = app.verification_lookup(
        [
            {
                "category": "visits",
                "statement": "Screening Visit",
                "source_page": 2,
                "status": "exact_match",
            }
        ]
    )
    assert index[("visits", "Screening Visit")]["status"] == "exact_match"


def test_filter_hides_informational_rescue_warnings():
    import app

    warnings = [
        "Eligibility rescue skipped: criteria appear complete.",
        (
            "Completeness warning: source eligibility text mentions HbA1c, "
            "but no extracted inclusion/exclusion criterion contains it."
        ),
    ]
    filtered = app.filter_user_facing_warnings(warnings)
    assert not any("rescue skipped" in item.casefold() for item in filtered)
    assert any("HbA1c" in item for item in filtered)
    assert app.is_informational_warning(
        "Eligibility rescue skipped: criteria appear complete."
    )


def test_classify_safety_monitoring_vs_risk():
    import app

    assert (
        app.classify_safety_statement(
            "Participants will be monitored for adverse events."
        )
        == "monitoring"
    )
    assert (
        app.classify_safety_statement(
            "Hypoglycemia below 70 mg/dL requires clinical review."
        )
        == "risk"
    )


def test_display_answer_reduces_duplication_and_keeps_thresholds():
    import app

    answer = PatientAnswer(
        answer_text=(
            "Based on the verified study summary, people may be able to join "
            "if these listed requirements apply. Exact wording is preserved:\n"
            '- One requirement listed in the study summary is: '
            '"HbA1c greater than or equal to 7.0%." '
            "(source page 1; verification: exact_match)"
        ),
        intent="eligibility_inclusion",
        citations=[
            PatientCitation(
                category="inclusion_criteria",
                statement="HbA1c greater than or equal to 7.0%.",
                source_page=1,
                verification_status="exact_match",
                plain_language=(
                    'One requirement listed in the study summary is: '
                    '"HbA1c greater than or equal to 7.0%."'
                ),
            )
        ],
        confidence="high",
        limitations=[],
        disclaimers=["Patient disclaimer"],
        audience="patient",
    )

    display = app.format_display_answer_text(answer, audience="patient")
    assert "7.0%" in display
    assert 'HbA1c greater than or equal to 7.0%.' in display
    # Page/verification stay out of the main body (shown in expanders).
    assert "source page 1" not in display.casefold()
    assert "verification: exact_match" not in display.casefold()
    # Avoid repeating the long plain-language wrapper in the body.
    assert "One requirement listed in the study summary is:" not in display


def test_display_eligibility_refusal_has_spaced_headings():
    import app

    answer = PatientAnswer(
        answer_text=(
            "I cannot tell you whether you personally are eligible for this "
            "study. Eligibility depends on your individual situation and must "
            "be decided with the study team.\n"
            "Here are general criteria from the verified study summary that "
            "you can discuss with them:\n"
            "Requirements that may allow someone to join:\n"
            '- One requirement listed in the study summary is: '
            '"Participants must be between 18 and 65 years old." '
            "(source page 1; verification: exact_match)\n"
            "Reasons someone may not be able to join:\n"
            '- The study summary lists this as a reason someone may not be '
            'able to join: "eGFR below 30 mL/min/1.73m2." '
            "(source page 1; verification: exact_match)"
        ),
        intent="eligibility_decision_refused",
        citations=[
            PatientCitation(
                category="inclusion_criteria",
                statement="Participants must be between 18 and 65 years old.",
                source_page=1,
                verification_status="exact_match",
            ),
            PatientCitation(
                category="exclusion_criteria",
                statement="eGFR below 30 mL/min/1.73m2.",
                source_page=1,
                verification_status="exact_match",
            ),
        ],
        confidence="high",
        limitations=["This assistant does not evaluate personal eligibility."],
        disclaimers=["Patient disclaimer"],
        audience="patient",
    )

    display = app.format_display_answer_text(answer, audience="patient")
    assert "cannot tell you whether you personally are eligible" in (
        display.casefold()
    )
    assert "**Requirements that may allow someone to join:**" in display
    assert "**Reasons someone may not be able to join:**" in display
    # Heading and first bullet are separated (markdown-friendly).
    assert re_search_blank_line_before_bullet(
        display,
        "Requirements that may allow someone to join",
    )
    assert "18 and 65" in display
    assert "eGFR below 30 mL/min/1.73m2." in display
    assert "source page 1" not in display.casefold()


def re_search_blank_line_before_bullet(display: str, header_fragment: str) -> bool:
    pattern = rf"\*\*{re.escape(header_fragment)}:\*\*\n\n- "
    return re.search(pattern, display) is not None


def test_display_safety_distinguishes_risks_and_monitoring():
    import app

    monitoring_only = PatientAnswer(
        answer_text="Here is safety-related information...",
        intent="safety",
        citations=[
            PatientCitation(
                category="safety_information",
                statement="Participants will be monitored for adverse events.",
                source_page=2,
                verification_status="exact_match",
            )
        ],
        confidence="high",
        limitations=[],
        disclaimers=[],
        audience="patient",
    )
    monitoring_display = app.format_display_answer_text(
        monitoring_only,
        audience="patient",
    )
    assert "**Adverse-event monitoring:**" in monitoring_display
    assert "not specified in the extracted summary" in monitoring_display
    assert "does not necessarily mean the full protocol lacks" in (
        monitoring_display
    )
    assert "**Study risk information:**" not in monitoring_display

    with_risk = PatientAnswer(
        answer_text="Here is safety-related information...",
        intent="safety",
        citations=[
            PatientCitation(
                category="safety_information",
                statement="Participants will be monitored for adverse events.",
                source_page=2,
                verification_status="exact_match",
            ),
            PatientCitation(
                category="safety_information",
                statement=(
                    "Hypoglycemia below 70 mg/dL requires clinical review."
                ),
                source_page=2,
                verification_status="exact_match",
            ),
        ],
        confidence="high",
        limitations=[],
        disclaimers=[],
        audience="patient",
    )
    risk_display = app.format_display_answer_text(with_risk, audience="patient")
    assert "**Study risk information:**" in risk_display
    assert "70 mg/dL" in risk_display
    assert "not specified in the extracted summary" not in risk_display


def test_display_preserves_refusal_without_inventing_facts():
    import app

    answer = PatientAnswer(
        answer_text=(
            "I cannot provide medical advice, diagnosis, or treatment "
            "recommendations."
        ),
        intent="medical_advice_refused",
        citations=[],
        confidence="unavailable",
        limitations=["Medical advice and personal clinical decisions are out of scope."],
        disclaimers=["Patient disclaimer"],
        audience="patient",
    )
    display = app.format_display_answer_text(answer, audience="patient")
    assert "cannot provide medical advice" in display.casefold()
    assert "7.0%" not in display
    assert "eGFR" not in display
