
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    TrialVisit,
)
from services.protocol_accuracy import (
    apply_protocol_accuracy_checks,
    check_eligibility_completeness,
    ground_protocol_metadata,
    is_grounded_protocol_id,
    is_grounded_trial_title,
    recover_labeled_metadata,
    repair_source_pages,
)


PROTOCOL_TEXT = """
[SOURCE PAGE 1]
PROTOCOL TITLE: Complex Layout Diabetes Study
PROTOCOL ID: CLX-205

INCLUSION CRITERIA:
1. Participants must be between 18 and 65 years old.
2. HbA1c greater than or equal to 7.0%.
3. eGFR at least 60 mL/min/1.73m2.

[SOURCE PAGE 2]
VISIT SCHEDULE:
Screening Visit: Day 0.
Follow-up Visit: Day 30.
"""


PDF_RESULT = {
    "pages": [
        {
            "page_number": 1,
            "text": (
                "PROTOCOL TITLE: Complex Layout Diabetes Study\n"
                "PROTOCOL ID: CLX-205\n"
                "INCLUSION CRITERIA:\n"
                "1. Participants must be between 18 and 65 years old.\n"
                "2. HbA1c greater than or equal to 7.0%.\n"
                "3. eGFR at least 60 mL/min/1.73m2."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "VISIT SCHEDULE:\n"
                "Screening Visit: Day 0.\n"
                "Follow-up Visit: Day 30."
            ),
        },
    ]
}


def test_grounded_title_and_id_accepted():
    assert is_grounded_trial_title(
        "Complex Layout Diabetes Study",
        PROTOCOL_TEXT,
    )
    assert is_grounded_protocol_id("CLX-205", PROTOCOL_TEXT)

    analysis = ProtocolAnalysis(
        trial_title="Complex Layout Diabetes Study",
        protocol_id="CLX-205",
    )
    warnings = ground_protocol_metadata(analysis, PROTOCOL_TEXT)

    assert warnings == []
    assert analysis.trial_title == "Complex Layout Diabetes Study"
    assert analysis.protocol_id == "CLX-205"


def test_ungrounded_title_and_id_nulled():
    analysis = ProtocolAnalysis(
        trial_title="Example Trial",
        protocol_id="example-protocol-123",
    )
    warnings = ground_protocol_metadata(analysis, PROTOCOL_TEXT)

    assert analysis.trial_title is None
    assert analysis.protocol_id is None
    assert any("trial_title" in warning for warning in warnings)
    assert any("protocol_id" in warning for warning in warnings)


def test_generic_short_title_rejected_even_if_substring_exists():
    # "study" appears in the source but is too generic to accept.
    assert not is_grounded_trial_title("Study", PROTOCOL_TEXT)
    assert not is_grounded_trial_title("Clinical Trial", PROTOCOL_TEXT)


def test_repair_missing_source_page_unique_match():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description=(
                    "Participants must be between 18 and 65 years old."
                ),
                source_page=None,
            )
        ]
    )

    warnings = repair_source_pages(analysis, PDF_RESULT)

    assert analysis.inclusion_criteria[0].source_page == 1
    assert any("repaired missing source_page" in w for w in warnings)


def test_repair_incorrect_source_page_unique_match():
    analysis = ProtocolAnalysis(
        visits=[
            TrialVisit(
                visit_name="Screening Visit",
                timing="Day 0",
                source_page=1,
            )
        ]
    )

    warnings = repair_source_pages(analysis, PDF_RESULT)

    assert analysis.visits[0].source_page == 2
    assert any("corrected source_page" in w for w in warnings)


def test_ambiguous_source_page_not_invented():
    ambiguous_pdf = {
        "pages": [
            {
                "page_number": 1,
                "text": "Screening Visit appears here.",
            },
            {
                "page_number": 2,
                "text": "Screening Visit also appears here.",
            },
        ]
    }
    analysis = ProtocolAnalysis(
        visits=[
            TrialVisit(
                visit_name="Screening Visit",
                source_page=None,
            )
        ]
    )

    warnings = repair_source_pages(analysis, ambiguous_pdf)

    assert analysis.visits[0].source_page is None
    assert any("ambiguous source pages" in w for w in warnings)


def test_unsupported_statement_flagged_without_fabricating_page():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="Invented criterion not in the protocol.",
                source_page=None,
            )
        ]
    )

    warnings = repair_source_pages(analysis, PDF_RESULT)

    assert analysis.inclusion_criteria[0].source_page is None
    assert any("unsupported statement" in w for w in warnings)


def test_completeness_warns_when_hba1c_and_egfr_missing():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description=(
                    "Participants must be between 18 and 65 years old."
                ),
                source_page=1,
            )
        ]
    )

    warnings = check_eligibility_completeness(
        analysis,
        PDF_RESULT,
        PROTOCOL_TEXT,
    )

    assert any("HbA1c" in warning for warning in warnings)
    assert any("eGFR" in warning for warning in warnings)
    assert len(analysis.inclusion_criteria) == 1


def test_completeness_silent_when_markers_extracted():
    analysis = ProtocolAnalysis(
        inclusion_criteria=[
            EligibilityCriterion(
                description="HbA1c greater than or equal to 7.0%.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="eGFR at least 60 mL/min/1.73m2.",
                source_page=1,
            ),
        ]
    )

    warnings = check_eligibility_completeness(
        analysis,
        PDF_RESULT,
        PROTOCOL_TEXT,
    )

    assert warnings == []


def test_recover_labeled_metadata_when_null():
    analysis = ProtocolAnalysis(trial_title=None, protocol_id=None)

    warnings = recover_labeled_metadata(analysis, PROTOCOL_TEXT)

    assert analysis.trial_title == "Complex Layout Diabetes Study"
    assert analysis.protocol_id == "CLX-205"
    assert any("trial_title recovered" in warning for warning in warnings)
    assert any("protocol_id recovered" in warning for warning in warnings)


def test_recover_labeled_metadata_does_not_invent():
    analysis = ProtocolAnalysis(trial_title=None, protocol_id=None)
    source = "INCLUSION CRITERIA:\n1. Age 18 or older.\n"

    warnings = recover_labeled_metadata(analysis, source)

    assert analysis.trial_title is None
    assert analysis.protocol_id is None
    assert warnings == []


def test_apply_protocol_accuracy_checks_combined():
    analysis = ProtocolAnalysis(
        trial_title="Example Trial",
        protocol_id="CLX-205",
        inclusion_criteria=[
            EligibilityCriterion(
                description=(
                    "Participants must be between 18 and 65 years old."
                ),
                source_page=None,
            )
        ],
        visits=[
            TrialVisit(
                visit_name="Follow-up Visit",
                timing="Day 30",
                source_page=1,
            )
        ],
    )

    warnings = apply_protocol_accuracy_checks(
        analysis,
        PDF_RESULT,
        PROTOCOL_TEXT,
    )

    # Invented title nulled; labeled title recovered afterward would require
    # null first — here invented title is grounded-null then recovery fills it.
    assert analysis.trial_title == "Complex Layout Diabetes Study"
    assert analysis.protocol_id == "CLX-205"
    assert analysis.inclusion_criteria[0].source_page == 1
    assert analysis.visits[0].source_page == 2
    assert any("HbA1c" in warning for warning in warnings)
    assert any("eGFR" in warning for warning in warnings)


def test_apply_recovers_null_metadata_then_grounds():
    analysis = ProtocolAnalysis(trial_title=None, protocol_id=None)

    warnings = apply_protocol_accuracy_checks(
        analysis,
        PDF_RESULT,
        PROTOCOL_TEXT,
    )

    assert analysis.trial_title == "Complex Layout Diabetes Study"
    assert analysis.protocol_id == "CLX-205"
    assert any("recovered from labeled source field" in w for w in warnings)
