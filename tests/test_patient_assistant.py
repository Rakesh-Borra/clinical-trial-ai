
from agents.patient_agent import answer_patient_query, classify_patient_intent
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from services.patient_query import answer_patient_from_pipeline_result


def _sample_analysis() -> ProtocolAnalysis:
    return ProtocolAnalysis(
        trial_title="Complex Layout Diabetes Study",
        protocol_id="CLX-205",
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
        procedures=[
            TrialProcedure(
                name="Blood glucose testing.",
                source_page=2,
            )
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


def _sample_verification() -> list[dict]:
    return [
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
            "category": "inclusion_criteria",
            "statement": "eGFR at least 60 mL/min/1.73m2.",
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


def test_classify_patient_intents():
    assert classify_patient_intent("Who can join this study?") == (
        "eligibility_inclusion"
    )
    assert classify_patient_intent("Who cannot join?") == "eligibility_exclusion"
    assert classify_patient_intent("What visits will I have?") == "visits"
    assert classify_patient_intent("What procedures are done?") == "procedures"
    assert classify_patient_intent("What are the safety risks?") == "safety"
    assert classify_patient_intent("What is this study called?") == "study_overview"


def test_inclusion_preserves_hba1c_and_egfr():
    answer = answer_patient_query(
        "Who can join this study?",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "eligibility_inclusion"
    assert answer.audience == "patient"
    assert answer.confidence == "high"
    assert "7.0%" in answer.answer_text
    assert "eGFR at least 60 mL/min/1.73m2." in answer.answer_text
    assert all(c.verification_status == "exact_match" for c in answer.citations)
    assert all(c.source_page == 1 for c in answer.citations)
    assert len({c.statement for c in answer.citations}) == len(answer.citations)


def test_exclusion_visits_procedures_safety_and_overview():
    analysis = _sample_analysis()
    verification = _sample_verification()

    exclusion = answer_patient_query(
        "Who cannot join the study?",
        analysis,
        verification,
    )
    assert exclusion.intent == "eligibility_exclusion"
    assert "eGFR below 30 mL/min/1.73m2." in exclusion.answer_text

    visits = answer_patient_query(
        "What is the visit schedule?",
        analysis,
        verification,
    )
    assert visits.intent == "visits"
    assert "Day 0" in visits.answer_text
    assert "Day 30" in visits.answer_text
    assert "2 distinct scheduled visits" not in visits.answer_text

    visit_count = answer_patient_query(
        "How many visits will I have?",
        analysis,
        verification,
    )
    assert visit_count.intent == "visits"
    assert "2 distinct scheduled visits" in visit_count.answer_text
    assert "Day 0" in visit_count.answer_text
    assert "Day 30" in visit_count.answer_text
    assert all(c.verification_status == "exact_match" for c in visit_count.citations)

    procedures = answer_patient_query(
        "What procedures will happen?",
        analysis,
        verification,
    )
    assert procedures.intent == "procedures"
    assert "Blood glucose testing." in procedures.answer_text

    safety = answer_patient_query(
        "What safety information should I know?",
        analysis,
        verification,
    )
    assert safety.intent == "safety"
    assert "adverse events" in safety.answer_text

    overview = answer_patient_query(
        "What is this study called?",
        analysis,
        verification,
    )
    assert overview.intent == "study_overview"
    assert "Complex Layout Diabetes Study" in overview.answer_text
    assert "CLX-205" in overview.answer_text


def test_prefers_exact_match_and_labels_unverified():
    verification = _sample_verification()
    verification[1]["status"] = "not_verified"

    answer = answer_patient_query(
        "Who can join this study?",
        _sample_analysis(),
        verification,
    )

    assert answer.confidence == "partial"
    statements = [c.statement for c in answer.citations]
    assert "HbA1c greater than or equal to 7.0%." not in statements
    assert any("Unverified" in item for item in answer.limitations)


def test_missing_category_does_not_imply_protocol_absence():
    analysis = ProtocolAnalysis(
        trial_title="No Safety Listed Study",
        protocol_id="NSL-1",
    )
    answer = answer_patient_query(
        "What safety information should I know?",
        analysis,
        [],
    )

    assert answer.confidence == "unavailable"
    assert "not available in the extracted study summary" in answer.answer_text
    assert "does not necessarily mean the full protocol lacks it" in (
        answer.answer_text
    )
    assert answer.citations == []


def test_refuses_personal_eligibility_decision():
    answer = answer_patient_query(
        "Am I eligible for this study?",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "eligibility_decision_refused"
    assert "cannot tell you whether you personally are eligible" in (
        answer.answer_text.casefold()
    )
    assert "7.0%" in answer.answer_text
    assert answer.audience == "patient"
    assert any("eligib" in d.casefold() for d in answer.disclaimers)


def test_refuses_medical_advice():
    answer = answer_patient_query(
        "Please give me medical advice about my diabetes medicine.",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "medical_advice_refused"
    assert answer.citations == []
    assert "cannot provide medical advice" in answer.answer_text.casefold()


def test_unsupported_and_empty_queries():
    unsupported = answer_patient_query(
        "What is the weather today?",
        _sample_analysis(),
        _sample_verification(),
    )
    assert unsupported.intent == "unsupported"
    assert unsupported.confidence == "unavailable"
    assert unsupported.citations == []

    empty = answer_patient_query(
        "   ",
        _sample_analysis(),
        _sample_verification(),
    )
    assert empty.intent == "unsupported"
    assert "Empty query." in empty.limitations


def test_filters_informational_accuracy_warnings():
    answer = answer_patient_query(
        "Who can join this study?",
        _sample_analysis(),
        _sample_verification(),
        accuracy_warnings=[
            "Eligibility rescue skipped: criteria appear complete.",
            (
                "Completeness warning: source eligibility text mentions HbA1c, "
                "but no extracted inclusion/exclusion criterion contains it."
            ),
        ],
    )

    assert not any("rescue skipped" in item.casefold() for item in answer.limitations)
    assert any("HbA1c" in item for item in answer.limitations)


def test_pipeline_wrapper_and_no_invented_facts():
    pipeline_result = {
        "analysis": _sample_analysis(),
        "verification": _sample_verification(),
        "extraction_warnings": [],
        "accuracy_warnings": [
            "Eligibility rescue skipped: criteria appear complete.",
        ],
    }

    answer = answer_patient_from_pipeline_result(
        "What visits will I have?",
        pipeline_result,
    )

    assert answer.intent == "visits"
    assert answer.confidence == "high"
    assert "Day 0" in answer.answer_text
    assert "Day 30" in answer.answer_text

    analysis_text = " ".join(
        [
            "Complex Layout Diabetes Study",
            "CLX-205",
            "Participants must be between 18 and 65 years old.",
            "HbA1c greater than or equal to 7.0%.",
            "eGFR at least 60 mL/min/1.73m2.",
            "eGFR below 30 mL/min/1.73m2.",
            "Blood glucose testing.",
            "Screening Visit",
            "Day 0",
            "Follow-up Visit",
            "Day 30",
            "Participants will be monitored for adverse events.",
        ]
    )
    for citation in answer.citations:
        assert citation.statement.split(":")[0] in analysis_text
