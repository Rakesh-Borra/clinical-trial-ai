"""Patients see only the current published version of each study."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from agents.guide_patient_agent import answer_patient_from_guide
from services import review_store, version_store
from services.guide_review import patient_visible_guide
from tests.auth_helpers import sign_in_as
from tests.test_patient_guide_ui import APP_PATH
from tests.version_helpers import make_guide

NAME = "T2 Diabetes"
OTHER = "Asthma Study"

V1_CONTENT = {
    "inclusion": [("You must be 18 to 65 years old.", 1, "Age 18–65 years inclusive.")],
    "visits": [("A follow-up visit on Day 30.", 2, "Follow-up Visit. Timing: Day 30")],
    "procedures": [("Blood is drawn at each visit.", 5, "Blood collection occurs.")],
}
V2_CONTENT = {
    "inclusion": [("You must be 18 to 70 years old.", 1, "Age 18–70 years inclusive.")],
    "visits": [
        ("A follow-up visit on Day 30.", 2, "Follow-up Visit. Timing: Day 30"),
        (
            "There will be an extra check-up visit on Day 60.",
            2,
            "Additional Follow-up Visit. Timing: Day 60",
        ),
    ],
    "procedures": [
        ("Blood is drawn at each visit.", 5, "Blood collection occurs."),
        ("An HbA1c blood test is added.", 5, "HbA1c blood test at screening."),
    ],
}


def _save(pid, content, *, name=NAME, study_id=None, label=None):
    review_store.save_draft(
        pid,
        make_guide(content),
        name=name,
        uploaded_name="p.pdf",
        upload_fingerprint=f"{pid}.pdf:1:x",
        study_id=study_id,
        version_label=label,
    )
    return version_store.get_version(pid)["study_id"]


def _publish(pid, content, name=NAME):
    review_store.publish(pid, name, make_guide(content))


def _patient_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    return at.run()


def _selector_options(at):
    return list(at.selectbox(key="patient_protocol_select").options)


def _guide_blob(guide):
    return " ".join(f"{i.plain_text} {i.original_text}" for i in guide.all_items())


def test_v1_published_patient_sees_v1():
    _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    at = _patient_app()
    assert set(at.session_state["published_protocols"]) == {"v1"}
    assert _selector_options(at) == [NAME]


def test_v2_uploaded_but_pending_patient_still_sees_v1():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    assert review_store.publication_status("v2") is None
    at = _patient_app()
    assert set(at.session_state["published_protocols"]) == {"v1"}
    assert at.session_state["published_protocols"]["v1"]["version_label"] == "1.0"


def test_v2_approved_but_unpublished_patient_still_sees_v1():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    # Approving the guide draft does not publish.
    draft = review_store.load_draft("v2")
    assert draft is not None
    at = _patient_app()
    assert set(at.session_state["published_protocols"]) == {"v1"}
    assert "v2" not in at.session_state["published_protocols"]


def test_v2_published_patient_sees_only_v2():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)
    assert review_store.publication_status("v1") == review_store.SUPERSEDED
    assert review_store.publication_status("v2") == review_store.PUBLISHED
    visible = review_store.patient_visible_publications()
    assert set(visible) == {"v2"}
    at = _patient_app()
    assert set(at.session_state["published_protocols"]) == {"v2"}
    assert _selector_options(at) == [NAME]


def test_v2_patient_guide_contains_amendments():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)
    guide = review_store.patient_visible_publications()["v2"]["guide"]
    blob = _guide_blob(guide).casefold()
    assert "70" in blob and "18" in blob
    assert "hba1c" in blob
    assert "day 60" in blob or "60" in blob
    assert "65" not in make_guide(V2_CONTENT).who_can_join.items[0].plain_text


def test_patient_chat_uses_only_v2_evidence():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)
    # Chat and guide share the same active published snapshot.
    active = review_store.patient_visible_publications()
    assert set(active) == {"v2"}
    guide = patient_visible_guide(active["v2"]["guide"])
    approved = [i for i in guide.all_items() if i.approved and not i.rejected]
    blob = _guide_blob(guide).casefold()
    assert any("70" in i.plain_text for i in approved)
    assert "hba1c" in blob and ("day 60" in blob or "60" in blob)
    # V1 upper age is not in the patient-visible corpus.
    assert not any("65" in i.plain_text for i in approved)
    answer = answer_patient_from_guide("Who can join this study?", guide)
    text = answer.answer_text.casefold()
    assert "70" in text
    assert "65" not in text


def test_v1_only_facts_do_not_override_v2():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)
    at = _patient_app()
    record = at.session_state["published_protocols"]["v2"]
    blob = _guide_blob(record["guide"]).casefold()
    assert "18–70" in blob or "18-70" in blob or "70" in blob
    assert "day 60" in blob or "extra check-up" in blob
    assert "hba1c" in blob
    # Superseded V1 snapshot remains stored for coordinators, not patients.
    assert "v1" not in at.session_state["published_protocols"]
    assert review_store.load_publications()["v1"]["status"] == review_store.SUPERSEDED


def test_two_different_studies_remain_separately_selectable():
    _save("a", V1_CONTENT, name=NAME)
    _publish("a", V1_CONTENT, name=NAME)
    _save("b", V1_CONTENT, name=OTHER)
    _publish("b", V1_CONTENT, name=OTHER)
    visible = review_store.patient_visible_publications()
    assert set(visible) == {"a", "b"}
    at = _patient_app()
    assert set(at.session_state["published_protocols"]) == {"a", "b"}
    names = {rec["name"] for rec in at.session_state["published_protocols"].values()}
    assert names == {NAME, OTHER}


def test_sync_collapses_stale_v1_and_v2_session_to_v2_only(monkeypatch):
    """Stale session with V1+V2 collapses to V2 after sync (no auto V1 restore)."""
    import app

    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)

    class _State(dict):
        pass

    state = _State()
    state.update(
        {
            "published_protocols": {
                "v1": {
                    "name": NAME,
                    "guide": make_guide(V1_CONTENT),
                    "version_label": "1.0",
                    "study_id": study,
                },
                "v2": {
                    "name": NAME,
                    "guide": make_guide(V2_CONTENT),
                    "version_label": "2.0",
                    "study_id": study,
                },
            },
            "patient_active_protocol_id": "v1",
            "patient_protocol_select": "v1",
        }
    )
    monkeypatch.setattr(app.st, "session_state", state)
    app.sync_published_from_store()
    assert set(state["published_protocols"]) == {"v2"}
    assert state["patient_active_protocol_id"] is None


def test_restart_loads_only_v2_after_publish():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)
    at = _patient_app()
    assert set(at.session_state["published_protocols"]) == {"v2"}
    assert _selector_options(at) == [NAME]


def test_withdraw_v2_does_not_republish_v1():
    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)
    assert review_store.unpublish("v2") is True
    assert review_store.publication_status("v2") == review_store.WITHDRAWN
    assert review_store.publication_status("v1") == review_store.SUPERSEDED
    assert review_store.patient_visible_publications() == {}
    assert version_store.get_version("v1")["is_current"] == 0
    at = _patient_app()
    assert at.session_state["published_protocols"] == {}


def test_stale_session_duplicate_hidden_by_patient_selector(monkeypatch):
    """Patient portal hides superseded siblings even before sync rewrites session."""
    from ui import patient_view

    study = _save("v1", V1_CONTENT)
    _publish("v1", V1_CONTENT)
    _save("v2", V2_CONTENT, study_id=study, label="2.0")
    _publish("v2", V2_CONTENT)

    class _State(dict):
        pass

    state = _State()
    state["published_protocols"] = {
        "v1": {"name": NAME, "guide": make_guide(V1_CONTENT), "study_id": study},
        "v2": {"name": NAME, "guide": make_guide(V2_CONTENT), "study_id": study},
    }
    monkeypatch.setattr(patient_view.st, "session_state", state)
    visible = patient_view._published_protocols()
    assert set(visible) == {"v2"}
