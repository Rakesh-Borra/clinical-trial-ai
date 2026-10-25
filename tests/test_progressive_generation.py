"""Progressive generation, bulk approval, and the rewrite/verification checks."""

import threading
import time

import pytest

from agents.plain_language_agent import (
    approval_block_reason,
    approve_all_unflagged,
    approve_item,
    check_rewrite,
    edit_item,
    ensure_item_ids,
    generate_item,
    generate_patient_guide,
    merge_generated,
    statement_specs,
    verification_index,
    verification_warnings,
)
from models.protocol_models import ProtocolAnalysis, TrialVisit
from services import clinical_text, review_store
from services.guide_review import (
    FILTERS,
    PENDING,
    decision_block_reason,
    review_state,
    state_counts,
)
from services.protocol_accuracy import dedupe_visits, link_visit_activities
from services.source_verifier import _on_page
from tests.publish_helpers import exact_rows, gk
from tests.test_patient_guide_ui import (
    _click,
    _eight_statement_result,
    _generated,
    _prepare_review_app,
)
from ui.patient_guide_tab import analysis_fingerprint


def _echo(original, section, feedback=None):
    return f"In simple words: {original}"


# --- numbers, conditions, negation -----------------------------------------


def test_five_ml_never_becomes_half_a_millilitre():
    problems = check_rewrite(
        "Collect approximately 5 mL of blood at Week 2.",
        "Collect about .5 mL of blood at Week 2.",
    )
    assert any("missing: 5" in p for p in problems)
    assert any("0.5" in p for p in problems)


@pytest.mark.parametrize(
    "original,rewrite",
    [
        ("Take 5 mg.", "Take 5 mg."),
        ("Collect 0.5 mL.", "Collect .5 mL."),
        ("Up to 1,000 steps.", "Up to 1000 steps."),
        ("Collect approximately 5 mL.", "Collect about 5 mL."),
        ("Visit every two weeks.", "Visit every 2 weeks."),
    ],
)
def test_equivalent_numbers_are_accepted(original, rewrite):
    assert not clinical_text.missing_numbers(original, rewrite)
    assert not clinical_text.added_numbers(original, rewrite)


def test_unit_change_and_dropped_precision_are_caught():
    assert clinical_text.missing_numbers("Take 5 mg.", "Take 5 mL.")
    assert clinical_text.missing_numbers("HbA1c 7.0%", "HbA1c 7%")


def test_unsupported_added_details_are_caught():
    problems = check_rewrite(
        "Return for Visit 2 at Week 4.", "Return within 18 days for Visit 2 at Week 4."
    )
    assert any("18" in p for p in problems)
    assert any("within" in p for p in problems)


def test_applicable_condition_is_preserved_not_rewritten():
    original = "Pregnancy test when applicable."
    assert check_rewrite(original, "A pregnancy test if you are pregnant.")
    assert not check_rewrite(
        original, "A pregnancy test is done only when it applies (when applicable)."
    )


def test_semantic_negation_flip_is_caught():
    original = "This guide is not individualized advice."
    assert check_rewrite(original, "This guide gives individualized advice.")
    assert not check_rewrite(original, "This guide is not individualized advice for you.")


def test_negated_safety_wording_is_not_mistaken_for_advice():
    original = "Do not stop taking your current medicine without asking the study doctor."
    assert not check_rewrite(
        original, "Do not stop taking your current medicine without asking the study doctor."
    )


def test_real_medical_advice_is_still_blocked():
    assert check_rewrite(
        "Do not stop taking your medicine.", "You should stop taking your medicine."
    )
    # A negation in the rewrite does not excuse advice the original never gave.
    assert check_rewrite(
        "The study drug is taken daily.", "Do not worry, stop taking your pills if tired."
    )
    assert check_rewrite("The study is safe.", "There are no risks.")


# --- source matching and visit linking --------------------------------------


def test_plural_and_singular_forms_match_but_negation_does_not():
    assert _on_page("Hemoglobin blood tests", "hemoglobin blood test at each visit")
    assert not _on_page("No blood test", "blood test at each visit")
    assert not _on_page("Visit 5", "visit 15")


def _visit_analysis():
    return ProtocolAnalysis(
        visits=[
            TrialVisit(visit_name="Baseline", timing="Day 1", source_page=4),
            TrialVisit(visit_name="Baseline", timing="Day 1", source_page=4),
            TrialVisit(visit_name="Day 1 venous blood draw", source_page=5),
            TrialVisit(visit_name="Follow-up 1", timing="Week 2", source_page=4),
        ]
    )


def test_duplicate_visits_merge_and_blood_draw_links_to_its_visit():
    analysis = _visit_analysis()
    pdf = {"pages": [{"page_number": 5, "text": "Day 1 venous blood draw, 10 mL"}]}
    dedupe_visits(analysis)
    link_visit_activities(analysis, pdf)
    names = [v.visit_name for v in analysis.visits]
    assert names == ["Baseline", "Follow-up 1"]
    assert "Day 1 venous blood draw" in analysis.visits[0].procedures


def test_unsupported_activity_is_not_linked():
    analysis = _visit_analysis()
    pdf = {"pages": [{"page_number": 5, "text": "Unrelated text"}]}
    link_visit_activities(analysis, pdf)
    assert "Day 1 venous blood draw" in [v.visit_name for v in analysis.visits]


# --- progressive generation --------------------------------------------------


def _specs_and_index(result):
    analysis = result["analysis"]
    return statement_specs(analysis), verification_index(exact_rows(analysis))


def test_statement_ids_are_stable_and_not_positional():
    result = _eight_statement_result()
    analysis = result["analysis"]
    first = {s.item_id: s.text for s in statement_specs(analysis)}
    analysis.inclusion_criteria.reverse()
    second = {s.item_id: s.text for s in statement_specs(analysis)}
    assert first == second
    assert len(first) == len(statement_specs(analysis))


def test_identical_statements_are_generated_once():
    result = _eight_statement_result()
    analysis = result["analysis"]
    analysis.inclusion_criteria.append(analysis.inclusion_criteria[0].model_copy())
    ids = [s.item_id for s in statement_specs(analysis)]
    assert len(ids) == len(set(ids)) == 8


def test_partial_results_merge_without_overwriting_decisions():
    result = _eight_statement_result()
    analysis = result["analysis"]
    specs, index = _specs_and_index(result)
    items = {s.item_id: generate_item(s, _echo, index) for s in specs}
    ids = list(items)

    guide = generate_patient_guide(analysis, _echo)  # shell source of sections
    partial = guide.model_copy(deep=True)
    for section in partial.sections():
        section.items = []
    partial.generation_complete = False

    assert merge_generated(partial, analysis, {k: items[k] for k in ids[:3]})
    assert len(partial.all_items()) == 3 and not partial.generation_complete
    first = partial.all_items()[0]
    first.rejected = True
    assert merge_generated(partial, analysis, items)
    assert partial.generation_complete and len(partial.all_items()) == 8
    assert partial.all_items()[0] is first and first.rejected
    assert len({i.item_id for i in partial.all_items()}) == 8
    # Re-merging changes nothing and never duplicates.
    assert not merge_generated(partial, analysis, items)
    assert len(partial.all_items()) == 8


def test_publication_is_blocked_until_generation_finishes():
    result = _eight_statement_result()
    guide = generate_patient_guide(result["analysis"], _echo, verification=exact_rows(result["analysis"]))
    for item in guide.all_items():
        item.approved = True
    assert decision_block_reason(guide) is None
    guide.generation_complete = False
    assert "still being generated" in decision_block_reason(guide)


def test_failed_statement_is_retried_alone_and_blocks_until_fixed():
    result = _eight_statement_result()
    analysis = result["analysis"]
    specs, index = _specs_and_index(result)

    def broken(original, section, feedback=None):
        raise RuntimeError("model offline")

    failed = generate_item(specs[0], broken, index)
    assert failed.generation_failed and failed.needs_review
    good = {s.item_id: generate_item(s, _echo, index) for s in specs[1:]}
    stored = {specs[0].item_id: failed, **good}

    guide = generate_patient_guide(analysis, _echo, reuse=stored)
    guide.generation_complete = False
    merge_generated(guide, analysis, stored)
    assert not guide.generation_complete  # a failed statement keeps it open

    retried = {**good, specs[0].item_id: generate_item(specs[0], _echo, index)}
    calls = []

    def counting(original, section, feedback=None):
        calls.append(original)
        return _echo(original, section)

    reuse = {k: v for k, v in stored.items() if not v.generation_failed}
    generate_patient_guide(analysis, counting, reuse=reuse)
    assert calls == [specs[0].text]  # only the failed one is redone
    merge_generated(guide, analysis, retried)
    assert guide.generation_complete
    assert not any(i.generation_failed for i in guide.all_items())


def test_interrupted_job_keeps_finished_statements_in_sqlite():
    result = _eight_statement_result()
    specs, index = _specs_and_index(result)
    fingerprint = analysis_fingerprint(result)
    for spec in specs[:3]:
        review_store.save_generated_item("p-int", fingerprint, generate_item(spec, _echo, index))
    stored = review_store.load_generated_items("p-int", fingerprint)
    assert len(stored) == 3
    calls = []

    def counting(original, section, feedback=None):
        calls.append(original)
        return _echo(original, section)

    guide = generate_patient_guide(result["analysis"], counting, reuse=stored)
    assert len(calls) == len(specs) - 3
    assert len(guide.all_items()) == len(specs)
    # Saving the same statement again updates it instead of duplicating it.
    review_store.save_generated_item("p-int", fingerprint, stored[specs[0].item_id])
    assert len(review_store.load_generated_items("p-int", fingerprint)) == 3


def test_ids_are_assigned_to_legacy_statements_and_duplicates_repaired():
    result = _eight_statement_result()
    guide = generate_patient_guide(result["analysis"], _echo)
    for item in guide.all_items():
        item.item_id = ""
    guide.who_can_join.items.append(guide.who_can_join.items[0].model_copy())
    assert ensure_item_ids(guide)
    ids = [i.item_id for i in guide.all_items()]
    assert all(ids) and len(ids) == len(set(ids))


# --- approval -----------------------------------------------------------------


def test_filter_tab_is_removed_but_unverified_statements_stay_blocked():
    assert "needs_reverification" not in dict(FILTERS)
    result = _eight_statement_result()
    guide = generate_patient_guide(result["analysis"], _echo)  # no source check
    assert all(review_state(i) == PENDING for i in guide.all_items())
    assert "needs_reverification" not in state_counts(guide)
    assert all(verification_warnings(i) for i in guide.all_items())


def test_approve_all_approves_every_eligible_statement_only():
    result = _eight_statement_result()
    analysis = result["analysis"]
    guide = generate_patient_guide(analysis, _echo, verification=exact_rows(analysis))
    items = guide.all_items()
    items[0].source_status = "unverified"
    items[0].source_issue = "Not on the page."
    items[0].needs_review = True
    items[1].needs_review = True  # a failed automatic check
    items[2].rejected = True
    assert approve_all_unflagged(guide) == 5
    assert not items[0].approved and not items[1].approved and not items[2].approved
    assert state_counts(guide)[PENDING] == 2
    # Running it again (for newly completed statements) never touches blocked ones.
    assert approve_all_unflagged(guide) == 0
    with pytest.raises(ValueError):
        approve_item(items[0])


def test_edited_unsupported_statement_cannot_be_approved():
    result = _eight_statement_result()
    analysis = result["analysis"]
    guide = generate_patient_guide(analysis, _echo, verification=exact_rows(analysis))
    item = guide.all_items()[0]
    edit_item(item, "Something the protocol never said.")
    assert verification_warnings(item)
    with pytest.raises(ValueError):
        approve_item(item)  # needs a recorded override reason


# --- end to end through the review page ---------------------------------------


def _wait_for_stored(protocol_id, fingerprint, count):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if len(review_store.load_generated_items(protocol_id, fingerprint)) >= count:
            return
        time.sleep(0.05)
    raise AssertionError("statements were not stored in time")


def test_statements_can_be_approved_while_generation_continues(monkeypatch):
    from unittest.mock import patch

    gate = threading.Event()
    calls = []

    def rewriter(original, section, feedback=None):
        calls.append(original)
        if len(calls) > 2:
            gate.wait(10)
        return _echo(original, section)

    result = _eight_statement_result()
    fingerprint = analysis_fingerprint(result)
    with patch("agents.plain_language_agent.ollama_rewriter", rewriter):
        at = _prepare_review_app(result)
        try:
            _wait_for_stored("protocol-eight", fingerprint, 2)
            at = at.run()
            guide = at.session_state["patient_guide"]
            assert not guide.generation_complete
            assert 2 <= len(guide.all_items()) < 8

            first = guide.all_items()[0]
            at = _click(at, f"guide_{first.item_id}_approve")
            guide = at.session_state["patient_guide"]
            assert guide.all_items()[0].approved
            assert "still being generated" in decision_block_reason(guide)

            # A second browser session (refresh) joins the job and sees progress.
            at2 = _prepare_review_app(result)
            assert len(at2.session_state["patient_guide"].all_items()) >= 2
        finally:
            gate.set()
        at = _generated(at)

    guide = at.session_state["patient_guide"]
    assert guide.generation_complete and len(guide.all_items()) == 8
    assert len({i.item_id for i in guide.all_items()}) == 8
    assert sum(1 for i in guide.all_items() if i.approved) == 1  # decision kept
    assert len(calls) == 8  # no statement was rewritten twice
    from services import guide_jobs

    assert guide_jobs.jobs.started == 1
    saved = review_store.load_latest_draft()["guide"]
    assert saved.generation_complete and len(saved.all_items()) == 8


def test_legacy_unchecked_guide_can_be_rechecked_without_losing_decisions():
    result = _eight_statement_result()
    guide = generate_patient_guide(result["analysis"], _echo)  # unchecked sources
    guide.all_items()[0].rejected = True
    at = _prepare_review_app(result, guide)
    assert not at.exception
    at = _click(at, "guide_recheck_btn")
    saved = at.session_state["patient_guide"]
    assert saved.all_items()[0].rejected  # saved decision kept
    assert sum(1 for i in saved.all_items() if i.source_status == "verified") >= 1
    assert approval_block_reason(saved.all_items()[1]) is None
