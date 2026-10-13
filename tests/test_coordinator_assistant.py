
from agents.coordinator_assistant_agent import (
    answer_coordinator_query,
    classify_coordinator_intent,
)
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from services.coordinator_query import answer_from_pipeline_result


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


def test_classify_supported_intents():
    assert classify_coordinator_intent("What are the inclusion criteria?") == (
        "eligibility_inclusion"
    )
    assert classify_coordinator_intent("List exclusion criteria") == (
        "eligibility_exclusion"
    )
    assert classify_coordinator_intent("When is the follow-up visit?") == "visits"
    assert classify_coordinator_intent("What procedures are required?") == (
        "procedures"
    )
    assert classify_coordinator_intent("Any safety monitoring notes?") == "safety"
    assert classify_coordinator_intent("What is the protocol ID?") == "metadata"


def test_answer_inclusion_criteria_with_citations():
    answer = answer_coordinator_query(
        "What are the inclusion criteria?",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "eligibility_inclusion"
    assert answer.confidence == "high"
    assert len(answer.citations) == 2
    assert all(c.verification_status == "exact_match" for c in answer.citations)
    assert all(c.source_page == 1 for c in answer.citations)
    assert "HbA1c greater than or equal to 7.0%" in answer.answer_text
    assert "page 1" in answer.answer_text
    assert answer.disclaimers


def test_answer_visits_include_timings():
    answer = answer_coordinator_query(
        "Show the visit schedule and timings",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "visits"
    assert answer.confidence == "high"
    assert "Day 0" in answer.answer_text
    assert "Day 30" in answer.answer_text
    assert {c.statement.split(":")[0] for c in answer.citations} == {
        "Screening Visit",
        "Follow-up Visit",
    }
    # General schedule questions should not force a count lead-in.
    assert "2 distinct scheduled visits" not in answer.answer_text


def test_visit_count_questions_return_confirmed_count_and_schedule():
    from agents.coordinator_assistant_agent import is_visit_count_query

    assert is_visit_count_query("How many visits does the patient have?")
    assert is_visit_count_query("How many visits?")
    assert is_visit_count_query("Number of scheduled visits?")
    assert is_visit_count_query("How many study visits are there?")
    assert not is_visit_count_query("Show the visit schedule and timings")

    for query in (
        "How many visits does the patient have?",
        "How many visits?",
        "Number of scheduled visits?",
        "How many study visits are there?",
    ):
        answer = answer_coordinator_query(
            query,
            _sample_analysis(),
            _sample_verification(),
        )
        assert answer.intent == "visits"
        assert answer.confidence == "high"
        assert "2 distinct scheduled visits" in answer.answer_text
        assert "Day 0" in answer.answer_text
        assert "Day 30" in answer.answer_text
        assert all(c.verification_status == "exact_match" for c in answer.citations)
        assert all(c.source_page == 2 for c in answer.citations)


def test_visit_count_uses_distinct_names_for_duplicates():
    analysis = _sample_analysis()
    analysis.visits.append(
        TrialVisit(
            visit_name="Screening Visit",
            timing="Day 0",
            source_page=2,
        )
    )
    verification = _sample_verification()

    answer = answer_coordinator_query(
        "How many visits?",
        analysis,
        verification,
    )

    assert "2 distinct scheduled visits" in answer.answer_text
    assert "3 distinct" not in answer.answer_text


def test_visit_count_missing_visits_not_confirmed():
    analysis = ProtocolAnalysis(
        trial_title="No Visits Study",
        protocol_id="NV-1",
    )
    answer = answer_coordinator_query(
        "How many visits?",
        analysis,
        [],
    )

    assert answer.confidence == "unavailable"
    assert "confirmed visit count is not available" in answer.answer_text.casefold()
    assert "2 distinct" not in answer.answer_text
    assert answer.citations == []


def test_visit_count_unverified_visits_not_confirmed():
    verification = _sample_verification()
    for row in verification:
        if row["category"] == "visits":
            row["status"] = "not_verified"

    answer = answer_coordinator_query(
        "How many visits does the patient have?",
        _sample_analysis(),
        verification,
    )

    assert "confirmed visit count is not available" in answer.answer_text.casefold()
    assert "2 distinct scheduled visits" not in answer.answer_text
    # Schedule details may still appear as unverified/partial, but not as a
    # confirmed count.
    assert answer.confidence == "partial"


def test_prefers_verified_and_labels_unverified():
    analysis = _sample_analysis()
    verification = _sample_verification()
    # Mark one inclusion as unverified.
    verification[1]["status"] = "not_verified"

    answer = answer_coordinator_query(
        "What are the inclusion criteria?",
        analysis,
        verification,
    )

    assert answer.confidence == "partial"
    assert len(answer.citations) == 1
    assert answer.citations[0].verification_status == "exact_match"
    assert any("Unverified" in item for item in answer.limitations)
    assert "HbA1c" not in " ".join(c.statement for c in answer.citations)


def test_empty_category_returns_unavailable():
    analysis = ProtocolAnalysis(
        trial_title="Empty Safety Study",
        protocol_id="ESS-1",
    )
    answer = answer_coordinator_query(
        "Any safety information?",
        analysis,
        [],
    )

    assert answer.intent == "safety"
    assert answer.confidence == "unavailable"
    assert answer.citations == []
    assert "No safety statements were found" in answer.answer_text


def test_refuses_patient_eligibility_decision():
    answer = answer_coordinator_query(
        "Is this patient eligible for the trial?",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "eligibility_decision_refused"
    assert "cannot determine whether a specific patient is eligible" in (
        answer.answer_text.casefold()
    )
    assert any("eligibility determination" in item.casefold() for item in answer.limitations)
    # May still cite criteria for review, but does not decide.
    assert "HbA1c" in answer.answer_text or answer.citations


def test_refuses_medical_advice():
    answer = answer_coordinator_query(
        "Please give medical advice about dosing for my patient.",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "medical_advice_refused"
    assert "cannot provide medical advice" in answer.answer_text.casefold()
    assert answer.citations == []
    assert any("medical advice" in d.casefold() for d in answer.disclaimers)


def test_does_not_invent_missing_thresholds():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be 18 years or older.",
                source_page=1,
            )
        ]
    )
    verification = [
        {
            "category": "inclusion_criteria",
            "statement": "Participants must be 18 years or older.",
            "source_page": 1,
            "status": "exact_match",
        }
    ]
    warnings = [
        "Completeness warning: source eligibility text mentions HbA1c, "
        "but no extracted inclusion/exclusion criterion contains it."
    ]

    answer = answer_coordinator_query(
        "What are the inclusion criteria?",
        analysis,
        verification,
        accuracy_warnings=warnings,
    )

    assert "HbA1c" not in answer.answer_text
    assert any("HbA1c" in item for item in answer.limitations)
    assert len(answer.citations) == 1


def test_metadata_lookup():
    answer = answer_coordinator_query(
        "What is the trial title and protocol ID?",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "metadata"
    assert answer.confidence == "high"
    assert "Complex Layout Diabetes Study" in answer.answer_text
    assert "CLX-205" in answer.answer_text


def test_unsupported_query():
    answer = answer_coordinator_query(
        "What is the weather today?",
        _sample_analysis(),
        _sample_verification(),
    )

    assert answer.intent == "unsupported"
    assert answer.confidence == "unavailable"
    assert answer.citations == []


def test_pipeline_wrapper_uses_phase3_result_shape():
    pipeline_result = {
        "analysis": _sample_analysis(),
        "verification": _sample_verification(),
        "extraction_warnings": [],
        "accuracy_warnings": [],
    }

    answer = answer_from_pipeline_result(
        "List the procedures",
        pipeline_result,
    )

    assert answer.intent == "procedures"
    assert answer.confidence == "high"
    assert answer.citations[0].statement.startswith("Blood glucose testing")
