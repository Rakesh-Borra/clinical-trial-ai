"""
Tests for the Plain-Language Agent (Patient Guide).

These tests use fake rewriters instead of Ollama, so they run
quickly and do not need a model server.
"""

from agents.plain_language_agent import (
    DISCLAIMER,
    approve_all_unflagged,
    approve_item,
    check_rewrite,
    count_statements,
    display_term,
    edit_item,
    explain_term_query,
    explain_statement,
    find_medical_terms,
    find_missing_numbers,
    find_unsafe_phrases,
    generate_patient_guide,
    guide_to_markdown,
)
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)


def sample_analysis() -> ProtocolAnalysis:
    return ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be between 18 and 65 years old.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="HbA1c between 7.0% and 10.5% at screening.",
                source_page=1,
            ),
        ],
        exclusion_criteria=[
            EligibilityCriterion(
                description="eGFR below 45 mL/min/1.73 m2.",
                source_page=2,
            ),
        ],
        procedures=[
            TrialProcedure(name="Blood glucose testing.", source_page=2),
        ],
        visits=[
            TrialVisit(
                visit_name="Follow-up Visit",
                timing="Day 30",
                procedures=["Vital signs", "ECG"],
                source_page=3,
            ),
        ],
        safety_information=[
            SafetyInformation(
                description="Participants will be monitored for adverse events.",
                source_page=3,
            ),
        ],
    )


def echo_rewriter(original, section, feedback=None):
    """A well-behaved fake model: simple wording, keeps every number."""
    return f"In simple words: {original}"


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

def test_finds_medical_terms_case_insensitive():
    terms = find_medical_terms("HbA1c and eGFR will be checked with an ECG.")
    names = [t.term for t in terms]

    assert "hba1c" in names
    assert "egfr" in names
    assert "ecg" in names


def test_longer_term_wins_over_shorter_overlap():
    terms = find_medical_terms("Report any serious adverse event immediately.")
    names = [t.term for t in terms]

    assert "serious adverse event" in names
    assert "adverse event" not in names


def test_does_not_match_inside_other_words():
    # "renal" must not match inside "adrenal"
    terms = find_medical_terms("History of adrenal disease.")
    assert "renal" not in [t.term for t in terms]


# ---------------------------------------------------------------------------
# Number preservation
# ---------------------------------------------------------------------------

def test_detects_missing_numbers():
    missing = find_missing_numbers(
        "Age 18 to 65 with HbA1c of 7.5%",
        "Adults with HbA1c of 7.5%",
    )
    assert missing == ["18", "65"]


def test_all_numbers_preserved():
    assert find_missing_numbers(
        "Visit on Day 30",
        "You will have a visit on Day 30.",
    ) == []


def test_decimal_must_match_exactly():
    # 7.0 rounded to 7 is a change in a clinical value
    assert find_missing_numbers("HbA1c of 7.0%", "HbA1c of 7%") == ["7.0"]


# ---------------------------------------------------------------------------
# Medical advice and eligibility guardrails
# ---------------------------------------------------------------------------

def test_flags_eligibility_decisions():
    assert find_unsafe_phrases("Good news, you are eligible for this trial.")
    assert find_unsafe_phrases("You qualify for the study.")
    assert find_unsafe_phrases("You are not eligible.")


def test_flags_medical_advice():
    assert find_unsafe_phrases("You should stop taking metformin.")
    assert find_unsafe_phrases("Stop taking your insulin before the visit.")
    assert find_unsafe_phrases("I recommend you join.")
    assert find_unsafe_phrases("This treatment is guaranteed to work.")


def test_neutral_requirement_wording_is_allowed():
    assert find_unsafe_phrases(
        "People in this trial must be between 18 and 65 years old."
    ) == []


def test_check_rewrite_rejects_empty_output():
    assert check_rewrite("Age 18 or older", "   ") == ["The rewrite was empty."]


# ---------------------------------------------------------------------------
# Agent loop: retry and fallback
# ---------------------------------------------------------------------------

def test_good_rewrite_is_accepted():
    item = explain_statement(
        "Participants must be 18 years or older.",
        "inclusion_criteria",
        source_page=4,
        rewriter=echo_rewriter,
    )

    assert item.generated_by == "llm"
    assert item.needs_review is False
    assert item.source_page == 4
    assert "18" in item.plain_text


def test_retries_with_feedback_then_succeeds():
    calls = []

    def flaky_rewriter(original, section, feedback=None):
        calls.append(feedback)
        if feedback is None:
            return "Adults can join."  # drops 18 and 65
        return "Adults aged 18 to 65 can be considered."

    item = explain_statement(
        "Age 18 to 65 years.",
        "inclusion_criteria",
        rewriter=flaky_rewriter,
    )

    assert len(calls) == 2
    assert calls[0] is None
    assert "18" in calls[1] and "65" in calls[1]
    assert item.generated_by == "llm"
    assert item.needs_review is False


def test_falls_back_to_original_when_numbers_keep_missing():
    item = explain_statement(
        "eGFR below 45 mL/min/1.73 m2.",
        "exclusion_criteria",
        rewriter=lambda o, s, f=None: "People with weak kidneys cannot join.",
    )

    assert item.generated_by == "fallback"
    assert item.plain_text == "eGFR below 45 mL/min/1.73 m2."
    assert item.needs_review is True
    assert any("45" in reason for reason in item.review_reasons)
    # Glossary help is still attached in fallback mode
    assert "egfr" in [t.term for t in item.terms_explained]


def test_falls_back_when_model_gives_advice():
    item = explain_statement(
        "Participants must be 18 years or older.",
        "inclusion_criteria",
        rewriter=lambda o, s, f=None: "You are eligible if you are 18 years or older.",
    )

    assert item.generated_by == "fallback"
    assert item.needs_review is True
    assert "you are eligible" not in item.plain_text.lower()


def test_falls_back_when_model_is_unavailable():
    def broken_rewriter(original, section, feedback=None):
        raise ConnectionError("Ollama is not running")

    item = explain_statement(
        "Blood pressure measurement.",
        "procedures",
        rewriter=broken_rewriter,
    )

    assert item.generated_by == "fallback"
    assert item.plain_text == "Blood pressure measurement."
    assert item.needs_review is True


# ---------------------------------------------------------------------------
# Full guide
# ---------------------------------------------------------------------------

def test_generate_full_guide():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)

    assert guide.trial_title == "Type 2 Diabetes Treatment Study"
    assert guide.protocol_id == "T2D-001"
    assert len(guide.who_can_join.items) == 2
    assert len(guide.who_cannot_join.items) == 1
    assert len(guide.visits.items) == 1
    assert len(guide.procedures.items) == 1
    assert len(guide.safety.items) == 1
    assert guide.disclaimer == DISCLAIMER
    assert guide.flagged_items() == []


def test_guide_preserves_numbers_and_source_pages():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)

    hba1c_item = guide.who_can_join.items[1]
    assert "7.0" in hba1c_item.plain_text
    assert "10.5" in hba1c_item.plain_text
    assert hba1c_item.source_page == 1

    visit_item = guide.visits.items[0]
    assert "Day 30" in visit_item.plain_text
    assert visit_item.source_page == 3


def test_visit_text_includes_timing_and_procedures():
    seen = []

    def recording_rewriter(original, section, feedback=None):
        seen.append((section, original))
        return original

    generate_patient_guide(sample_analysis(), rewriter=recording_rewriter)

    visit_inputs = [text for section, text in seen if section == "visits"]
    assert visit_inputs == [
        "Follow-up Visit. Timing: Day 30. Procedures: Vital signs; ECG"
    ]


def test_glossary_collects_terms_without_duplicates():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)
    names = [t.term for t in guide.glossary]

    for expected in ["hba1c", "screening", "egfr", "ecg", "vital signs", "adverse event"]:
        assert expected in names
    assert len(names) == len(set(names))


def test_empty_sections_say_information_is_unavailable():
    guide = generate_patient_guide(
        ProtocolAnalysis(trial_title="Empty Study"),
        rewriter=echo_rewriter,
    )

    for section in guide.sections():
        assert section.items == []
        assert "did not include this information" in section.unavailable_message


def test_eligibility_sections_never_decide_for_the_patient():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)

    assert "Only the study team can decide" in guide.who_can_join.introduction
    assert "does not decide whether you can join" in guide.disclaimer
    for section in guide.sections():
        assert find_unsafe_phrases(section.introduction) == []
        for item in section.items:
            assert find_unsafe_phrases(item.plain_text) == []


def test_flagged_items_are_reported_for_review():
    def drop_numbers(original, section, feedback=None):
        return "Some simple words."

    guide = generate_patient_guide(sample_analysis(), rewriter=drop_numbers)
    flagged = guide.flagged_items()

    # Every item except the two without numbers should be flagged
    flagged_texts = [item.original_text for item in flagged]
    assert "Participants must be between 18 and 65 years old." in flagged_texts
    assert "Blood glucose testing." not in flagged_texts
    assert "Participants will be monitored for adverse events." not in flagged_texts


def test_markdown_output():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)
    markdown = guide_to_markdown(guide)

    assert markdown.startswith("# Type 2 Diabetes Treatment Study")
    assert "## Who may be able to join" in markdown
    assert "## Words to know" in markdown
    assert "**HbA1c**" in markdown
    assert "(protocol page 1)" in markdown
    assert DISCLAIMER in markdown


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------

def test_new_guide_starts_unapproved():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)

    assert guide.approved_count() == 0
    assert guide.is_fully_approved() is False


def test_patient_view_hides_unapproved_items():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)
    approve_item(guide.who_can_join.items[0])

    patient_view = guide_to_markdown(guide, approved_only=True)

    assert "18 and 65" in patient_view
    assert "7.0%" not in patient_view
    assert "still being reviewed" in patient_view
    # Glossary only lists terms from approved items
    assert "HbA1c" not in patient_view


def test_approve_all_unflagged_skips_flagged_items():
    def drop_numbers(original, section, feedback=None):
        return "Some simple words."

    guide = generate_patient_guide(sample_analysis(), rewriter=drop_numbers)
    newly_approved = approve_all_unflagged(guide)

    assert newly_approved == 2  # the two items that contain no numbers
    for item in guide.flagged_items():
        assert item.approved is False


def test_fully_approved_guide():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)
    approve_all_unflagged(guide)

    assert guide.is_fully_approved() is True
    assert guide.approved_count() == count_statements(sample_analysis())


def test_coordinator_edit_is_checked_and_needs_reapproval():
    guide = generate_patient_guide(sample_analysis(), rewriter=echo_rewriter)
    item = guide.who_can_join.items[0]
    approve_item(item)

    warnings = edit_item(item, "Adults can take part.")

    assert item.generated_by == "coordinator"
    assert item.approved is False
    assert item.needs_review is True
    assert any("18" in w for w in warnings)


def test_good_coordinator_edit_clears_flag():
    item = explain_statement(
        "Age 18 to 65 years.",
        "inclusion_criteria",
        rewriter=lambda o, s, f=None: "Adults.",
    )
    assert item.needs_review is True

    warnings = edit_item(item, "People aged 18 to 65 years.")

    assert warnings == []
    assert item.needs_review is False
    assert item.plain_text == "People aged 18 to 65 years."


def test_empty_edit_is_rejected():
    item = explain_statement("Age 18.", "inclusion_criteria", rewriter=echo_rewriter)
    before = item.plain_text

    assert edit_item(item, "   ") == ["The edited text is empty."]
    assert item.plain_text == before


def test_progress_callback_called_for_every_item():
    finished = []
    generate_patient_guide(
        sample_analysis(),
        rewriter=echo_rewriter,
        on_item=finished.append,
    )
    assert len(finished) == count_statements(sample_analysis()) == 6


# ---------------------------------------------------------------------------
# "What does X mean?"
# ---------------------------------------------------------------------------

def test_explains_term_questions():
    terms = explain_term_query("What does HbA1c mean?")
    assert [t.term for t in terms] == ["hba1c"]

    terms = explain_term_query("What's an ECG?")
    assert [t.term for t in terms] == ["ecg"]

    terms = explain_term_query("what is a placebo")
    assert [t.term for t in terms] == ["placebo"]


def test_term_query_ignores_other_questions():
    assert explain_term_query("Am I eligible for this trial?") is None
    assert explain_term_query("When is the follow-up visit?") is None
    assert explain_term_query("What does flibbertigibbet mean?") is None


def test_display_term_capitalisation():
    assert display_term("hba1c") == "HbA1c"
    assert display_term("egfr") == "eGFR"
    assert display_term("placebo") == "Placebo"
