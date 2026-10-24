"""
Source grounding of patient-facing wording, built on the real 2-page benchmark
(Type 2 Diabetes Treatment Study). All model output is mocked; Ollama is never
called. The protocol text below is the PDF's actual extracted text.
"""

import pytest

from agents.plain_language_agent import (
    approve_item,
    explain_statement,
    generate_item,
    statement_specs,
    verification_warnings,
)
from models.plain_language_models import GuideItem, GuideSection, PatientGuide
from models.protocol_models import (
    EligibilityCriterion as Criterion, ProtocolAnalysis,
    TrialProcedure as ProtocolProcedure, SafetyInformation, TrialVisit as Visit,
)
from services import grounding
from services.summary_review import (
    _generate_patient_summary,
    extractive_summary,
    validate_summary,
)

PAGE_1 = (
    "PROTOCOL TITLE: Type 2 Diabetes Treatment Study\nPROTOCOL ID: T2D-001\n\n"
    "INCLUSION CRITERIA:\n1. Participants must be between 18 and 65 years old.\n"
    "2. Participants must have Type 2 Diabetes.\n\nEXCLUSION CRITERIA:\n"
    "1. Participants with severe kidney disease are excluded."
)
PAGE_2 = (
    "STUDY PROCEDURES:\n1. Blood glucose testing.\n2. Blood pressure measurement.\n\n"
    "VISIT SCHEDULE:\nScreening Visit: Day 0.\nFollow-up Visit: Day 30.\n\n"
    "SAFETY INFORMATION:\nParticipants will be monitored for adverse events."
)
PAGES = {"1": PAGE_1, "2": PAGE_2}

# What qwen3:4b actually produced in the benchmark (statements and summary).
BAD_GLUCOSE = (
    "Blood glucose testing is a simple test that checks how much sugar is in your "
    "blood. We will measure your blood glucose level at specific times."
)
BAD_PRESSURE = (
    "We measure your blood pressure. This is done using a device called a "
    "sphygmomanometer. The measurement is taken in millimeters of mercury."
)
BAD_SUMMARY = """## About the Study

This clinical trial is designed to study a new treatment for Type 2 Diabetes. Participants will be closely monitored to ensure their safety and to collect important data about the treatment's effects.

## Who Can Participate

People in this trial must be between 18 and 65 years old and must have Type 2 Diabetes. Participants with severe kidney disease are excluded from the study.

## What Will Happen During the Study

You will receive a new treatment for Type . This treatment is intended to help manage your diabetes.

## Study Visits and Procedures

This is your first visit to the trial. It happens on Day 0. This trial has a follow-up visit on Day 30.

## Possible Risks and Discomforts

We will watch for any side effects that happen during the trial. These side effects may include temporary discomfort or changes in blood glucose levels.

## Your Rights as a Participant

You have the right to withdraw from the study at any time without penalty. Your decision will not affect your medical care.

## Contact Information

For more information about this study, contact the research team directly. They will provide details about eligibility and next steps.
"""


def _analysis():
    return ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
        inclusion_criteria=[
            Criterion(description="Participants must be between 18 and 65 years old.", source_page=1),
            Criterion(description="Participants must have Type 2 Diabetes.", source_page=1),
        ],
        exclusion_criteria=[
            Criterion(description="Participants with severe kidney disease are excluded.", source_page=1),
        ],
        procedures=[
            ProtocolProcedure(name="Blood glucose testing", source_page=2),
            ProtocolProcedure(name="Blood pressure measurement", source_page=2),
        ],
        visits=[
            Visit(visit_name="Screening Visit", timing="Day 0", source_page=2),
            Visit(visit_name="Follow-up Visit", timing="Day 30", source_page=2),
        ],
        safety_information=[
            SafetyInformation(description="Participants will be monitored for adverse events.", source_page=2),
        ],
    )


def _item(text, plain, page=2, section="procedures"):
    return GuideItem(
        section=section, original_text=text, plain_text=plain, source_page=page,
        source_status="verified", verification="verified",
    )


def _guide():
    """A guide of verified statements that match the 2-page protocol."""
    guide = PatientGuide(
        trial_title="Type 2 Diabetes Treatment Study",
        disclaimer="Test disclaimer.",
        who_can_join=GuideSection(title="Who may be able to join", introduction="", items=[
            _item("Participants must be between 18 and 65 years old.",
                  "People in this trial must be between 18 and 65 years old.", 1, "inclusion_criteria"),
            _item("Participants must have Type 2 Diabetes.",
                  "People in this trial must have Type 2 Diabetes.", 1, "inclusion_criteria"),
        ]),
        who_cannot_join=GuideSection(title="Who may not be able to join", introduction="", items=[
            _item("Participants with severe kidney disease are excluded.",
                  "People with severe kidney disease cannot join.", 1, "exclusion_criteria"),
        ]),
        visits=GuideSection(title="Trial visits", introduction="", items=[
            _item("Screening Visit. Timing: Day 0", "A screening visit on Day 0.", 2, "visits"),
            _item("Follow-up Visit. Timing: Day 30", "A follow-up visit on Day 30.", 2, "visits"),
        ]),
        procedures=GuideSection(title="Tests and procedures", introduction="", items=[
            _item("Blood glucose testing", "Blood glucose testing."),
            _item("Blood pressure measurement", "Blood pressure measurement."),
        ]),
        safety=GuideSection(title="Safety information", introduction="", items=[
            _item("Participants will be monitored for adverse events.",
                  "Participants will be monitored for side effects.", 2, "safety_information"),
        ]),
    )
    return guide


def _explain(original, rewrite, section="procedures", page=2, attempts=None):
    seen = []

    def rewriter(orig, sec, feedback=None):
        seen.append(feedback)
        return rewrite if not callable(rewrite) else rewrite(len(seen))

    item = explain_statement(
        original, section, page, rewriter, page_text=PAGES[str(page)]
    )
    item.source_status = "verified"  # the fact itself is on its page
    return item, seen


# --- Statement grounding ----------------------------------------------------


@pytest.mark.parametrize("original,rewrite", [
    ("Blood pressure measurement", BAD_PRESSURE),
    ("Blood glucose testing", BAD_GLUCOSE),
    ("Participants will be monitored for adverse events.",
     "We will watch for headaches, dizziness and bruising during the trial."),
])
def test_unsupported_rewrites_retry_once_then_fall_back_with_warning(original, rewrite):
    item, feedback = _explain(original, rewrite)
    assert len(feedback) == 2 and feedback[0] is None and feedback[1]  # one retry, with feedback
    assert item.generated_by == "fallback"
    assert item.plain_text == original  # the protocol's own wording
    assert item.needs_review
    warnings = verification_warnings(item)
    assert warnings and all(w.startswith(grounding.UNSUPPORTED_PREFIX) for w in warnings), warnings
    assert item.source_page == 2  # citation kept


def test_retry_feedback_names_the_unsupported_wording():
    _item_, feedback = _explain("Blood pressure measurement", BAD_PRESSURE)
    assert "sphygmomanometer" in feedback[1]


def test_retry_can_succeed_with_a_supported_rewrite():
    item, feedback = _explain(
        "Blood pressure measurement",
        lambda n: BAD_PRESSURE if n == 1 else "Your blood pressure will be measured.",
    )
    assert len(feedback) == 2
    assert item.generated_by == "llm" and not item.needs_review
    assert verification_warnings(item) == []


@pytest.mark.parametrize("original,rewrite,section", [
    ("Participants will be monitored for adverse events.",
     "We will watch for any side effects that happen during the trial.", "safety_information"),
    ("Participants must have Type 2 Diabetes.",
     "People in this trial must have Type 2 Diabetes.", "inclusion_criteria"),
    ("Blood glucose testing", "Your blood sugar will be tested.", "procedures"),
    ("Participants with severe kidney disease are excluded.",
     "People with severe kidney disease cannot join.", "exclusion_criteria"),
])
def test_legitimate_paraphrases_pass(original, rewrite, section):
    item, _ = _explain(original, rewrite, section=section, page=1 if "Type 2" in original or "kidney" in original else 2)
    assert item.generated_by == "llm", item.review_reasons
    assert verification_warnings(item) == []


def test_glossary_explanation_counts_as_support():
    # "adverse event" is explained by the approved glossary ("unwanted ... health problem").
    item, _ = _explain(
        "Participants will be monitored for adverse events.",
        "We will watch for any unwanted health problem during the trial.",
        section="safety_information",
    )
    assert item.generated_by == "llm", item.review_reasons


def test_other_line_of_the_page_does_not_support_a_rewrite():
    # "blood pressure" is on page 2 but not in the glucose statement's passage.
    item, _ = _explain("Blood glucose testing", "Your blood pressure will be checked.")
    assert item.generated_by == "fallback"


def test_truncated_diagnosis_in_a_statement_is_rejected():
    item, _ = _explain(
        "Participants must have Type 2 Diabetes.",
        "People in this trial must have Type .",
        section="inclusion_criteria", page=1,
    )
    assert item.generated_by == "fallback"
    assert any("cut off" in w for w in verification_warnings(item))


def test_fallback_warning_needs_a_recorded_override_to_approve():
    item, _ = _explain("Blood pressure measurement", BAD_PRESSURE)
    with pytest.raises(ValueError):
        approve_item(item)
    approve_item(item, "Checked page 2 myself; the protocol wording is right.")
    assert item.approved and item.override and item.override_reason
    assert item.override_warnings and item.needs_review  # audit keeps the warning


def test_generate_item_grounds_in_the_cited_page():
    spec = next(s for s in statement_specs(_analysis()) if "pressure" in s.text.lower())
    calls = []

    def rewriter(orig, sec, feedback=None):
        calls.append(feedback)
        return BAD_PRESSURE

    item = generate_item(spec, rewriter, None, PAGES)
    assert item.generated_by == "fallback" and len(calls) == 2
    assert any(w.startswith(grounding.UNSUPPORTED_PREFIX) for w in verification_warnings(item))
    assert item.source_page == 2


# --- Summary grounding ------------------------------------------------------


def test_the_benchmark_summary_is_rejected_for_each_invented_claim():
    problems = " ".join(validate_summary(BAD_SUMMARY, _guide()))
    assert "treatment" in problems  # unsupported treatment description
    assert "side effects" in problems or "discomfort" in problems  # invented risks
    assert "withdraw" in problems and "penalty" in problems  # invented rights
    assert "contact" in problems  # invented contact details
    assert "cut off" in problems and "Type" in problems  # truncated diagnosis


def test_headings_without_statements_are_rejected():
    problems = validate_summary(BAD_SUMMARY, _guide())
    assert any('heading "contact information"' in p for p in problems)
    assert any('heading "your rights as a participant"' in p for p in problems)
    assert not any('heading "who can participate"' in p for p in problems)


def test_unsupported_medical_terms_are_rejected():
    text = "## Study Visits and Procedures\n\nA nurse uses a sphygmomanometer on Day 0."
    assert any("sphygmomanometer" in p for p in validate_summary(text, _guide()))


def test_grounded_summary_with_paraphrases_passes():
    text = (
        "## Who Can Participate\n\nPeople in this trial must be between 18 and 65 "
        "years old and must have Type 2 Diabetes. People with severe kidney disease "
        "cannot join.\n\n## Study Visits and Procedures\n\nThere is a screening visit "
        "on Day 0 and a follow-up visit on Day 30. Your blood glucose and blood "
        "pressure are checked.\n\n## Possible Risks and Discomforts\n\n"
        "Participants will be monitored for side effects."
    )
    assert validate_summary(text, _guide()) == []


def test_sections_are_not_required_for_absent_categories():
    guide = _guide()
    guide.safety.items.clear()
    text = (
        "## Who Can Participate\n\nPeople in this trial must be between 18 and 65 years old "
        "and must have Type 2 Diabetes. People with severe kidney disease cannot join.\n\n"
        "## Study Visits and Procedures\n\nA screening visit on Day 0 and a follow-up visit "
        "on Day 30. Blood glucose testing and blood pressure measurement."
    )
    assert validate_summary(text, guide) == []
    risky = text + "\n\n## Possible Risks and Discomforts\n\nYou may feel pain."
    assert validate_summary(risky, guide)


def test_summary_retries_once_then_uses_grounded_fallback():
    guide = _guide()
    prompts = []

    def writer(prompt):
        prompts.append(prompt)
        return BAD_SUMMARY

    text, by = _generate_patient_summary(guide, writer)
    assert len(prompts) == 2 and "Fix them" in prompts[1]  # one retry, with feedback
    assert by == "fallback"
    assert text == extractive_summary(guide)
    assert validate_summary(text, guide) == []  # fallback is itself grounded
    for invented in ("new treatment", "penalty", "contact the research team", "Type ."):
        assert invented not in text


def test_good_summary_is_accepted_without_retry():
    guide = _guide()
    good = extractive_summary(guide)
    calls = []
    text, by = _generate_patient_summary(guide, lambda p: calls.append(p) or good)
    assert by == "llm" and len(calls) == 1


def test_summary_prompt_forbids_uncovered_topics():
    from services.summary_review import build_summary_prompt
    prompt = build_summary_prompt({"Trial visits": ["A visit on Day 0."]})
    assert "omit any topic that has no statements" in prompt
    assert "full" in prompt and "Type 2 Diabetes" in prompt


def test_editing_a_summary_to_add_unsupported_claims_is_flagged():
    edited = extractive_summary(_guide()) + "\n\nYou may withdraw at any time."
    assert any("withdraw" in p for p in validate_summary(edited, _guide()))


def test_rejected_statement_cannot_support_a_published_summary():
    guide = _guide()
    text = extractive_summary(guide)
    for item in guide.all_items():
        item.approved = True
    assert validate_summary(text, guide, approved_only=True) == []
    guide.safety.items[0].approved = False
    assert validate_summary(text, guide, approved_only=True)
