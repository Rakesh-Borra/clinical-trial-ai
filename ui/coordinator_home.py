"""
Coordinator home dashboard and the saved-protocol list.

Pure presentation: every action is returned to the caller (app.py), which owns
workflow state. Only data that is really stored is shown; nothing is invented.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import Any, Mapping, Optional

import streamlit as st

from services import protocol_deletion, review_store, version_store
from services.guide_review import ALL, APPROVED, EDITED, REJECTED, state_counts
from ui import theme

NO_PROTOCOLS_MESSAGE = "No protocols have been added yet."

ACTION_ADD = "add"
ACTION_VIEW = "view"
ACTION_VERSION = "version"
ACTION_RESUME = "resume"


def format_saved_time(value: Optional[str]) -> Optional[str]:
    """'2026-10-24T20:01:05+00:00' -> '2026-10-24 20:01 UTC'; None if unparseable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return None


def analysis_status(entry: Mapping[str, Any]) -> tuple[str, str]:
    if entry["has_analysis"]:
        return "Analysis saved", "teal"
    return "Analysis not saved", "sun"


def review_status(entry: Mapping[str, Any]) -> tuple[str, str]:
    """A one-line review/approval status built only from stored decisions."""
    guide = entry["guide"]
    if guide is None:
        return "Saved review unreadable", "coral"
    publication = entry["publication"]
    if publication == review_store.PUBLISHED:
        return "Published to patients", "teal"
    counts = state_counts(guide)
    total = counts[ALL]
    approved = counts[APPROVED] + counts[EDITED]
    if total == 0:
        text = "No statements generated yet"
    else:
        text = f"{approved} of {total} statements approved"
        if counts[REJECTED]:
            text += f", {counts[REJECTED]} rejected"
    text += " · summary approved" if entry["summary_approved"] else " · summary not approved"
    if publication == review_store.SUPERSEDED:
        return "Superseded by a newer published version", "grey"
    if publication == review_store.WITHDRAWN:
        text = "Withdrawn from patients · " + text
    return text, "grey"


def render_home(
    name: str, resumable_name: Optional[str] = None, resumable_detail: str = ""
) -> Optional[str]:
    """Welcome message and the three option cards; returns the chosen action."""
    theme.page_head(
        f"Welcome, {name}" if name else "Welcome, coordinator",
        "Add a new clinical trial protocol, or return to one you have already saved.",
    )
    action: Optional[str] = None
    if resumable_name is not None:
        with st.container(key="home_resume"):
            st.info(
                f"You have work in progress for **{resumable_name}**: {resumable_detail}."
            )
            if st.button("Return to your work in progress", key="home_resume_btn"):
                action = ACTION_RESUME
    left, middle, right = st.columns(3, gap="large")
    with left, st.container(key="home_card_add"):
        st.html(
            '<div class="ctai-label">Option 1</div>'
            '<h3 style="margin:0 0 .4rem">Add New Protocol</h3>'
            "<p>Upload and analyze a new clinical trial protocol.</p>"
        )
        if st.button("Add New Protocol", type="primary", key="home_add_btn", width="stretch"):
            action = ACTION_ADD
    with middle, st.container(key="home_card_version"):
        st.html(
            '<div class="ctai-label">Option 2</div>'
            '<h3 style="margin:0 0 .4rem">Upload New Version</h3>'
            "<p>Add an amended protocol as a new version of a saved clinical trial.</p>"
        )
        if st.button("Upload New Version", type="primary", key="home_version_btn", width="stretch"):
            action = ACTION_VERSION
    with right, st.container(key="home_card_view"):
        st.html(
            '<div class="ctai-label">Option 3</div>'
            '<h3 style="margin:0 0 .4rem">View Existing Protocols</h3>'
            "<p>Open a saved clinical trial protocol and continue.</p>"
        )
        if st.button("View Existing Protocols", type="primary", key="home_view_btn", width="stretch"):
            action = ACTION_VIEW
    return action


def render_protocol_list() -> Optional[tuple[str, str]]:
    """
    The saved protocols. Returns ("open", protocol_id) or ("add", "") when the
    coordinator picks one of the actions, otherwise None.
    """
    theme.page_head("Existing protocols", "Protocols you have saved, newest first.")
    try:
        entries = review_store.list_protocols()
    except (sqlite3.Error, OSError) as exc:
        st.error(f"Saved protocols could not be read from the database: {exc}")
        return None
    chosen: Optional[tuple[str, str]] = None
    notice = st.session_state.pop("delete_notice", None)
    if notice:
        st.success(notice)
    if not entries:
        st.info(NO_PROTOCOLS_MESSAGE)
        if st.button("Add New Protocol", type="primary", key="list_add_btn"):
            chosen = ("add", "")
        return chosen
    groups: dict[str, list] = {}
    for entry in entries:
        groups.setdefault(entry["study_id"] or entry["protocol_id"], []).append(entry)
    for study_entries in groups.values():
        first = study_entries[0]
        title = first["study_title"] or first["name"] or "Unnamed protocol"
        current = next((e for e in study_entries if e["is_current"]), None)
        st.html(f'<h3 style="margin:1.2rem 0 .2rem">{theme.e(title)}</h3>')
        st.caption(
            f"Current published version: {current['version_label']}"
            if current
            else "No version is currently published to patients."
        )
        for entry in study_entries:
            result = _render_version_row(entry)
            chosen = chosen or result
    return chosen


def version_status(publication: Optional[str]) -> tuple[str, str]:
    """Chip text for a version's publication state."""
    if publication == review_store.PUBLISHED:
        return "Current published version", "teal"
    if publication == review_store.SUPERSEDED:
        return "Superseded", "grey"
    if publication == review_store.WITHDRAWN:
        return "Withdrawn", "coral"
    return "Not published", "sun"


def _render_version_row(entry: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    protocol_id = entry["protocol_id"]
    chosen: Optional[tuple[str, str]] = None
    with st.container(key=f"protocol_row_{protocol_id}"):
        label = entry["version_label"] or "—"
        st.html(f'<h4 style="margin:0 0 .3rem">Version {theme.e(label)}</h4>')
        if entry["uploaded_name"]:
            st.caption(f"PDF: {entry['uploaded_name']}")
        uploaded = format_saved_time(entry["uploaded_at"])
        saved = format_saved_time(entry["updated_at"])
        published = format_saved_time(entry["published_at"])
        if uploaded:
            st.caption(f"Uploaded: {uploaded}")
        if saved:
            st.caption(f"Last saved: {saved}")
        if published:
            st.caption(f"Published: {published}")
        analysis_text, analysis_tone = analysis_status(entry)
        review_text, review_tone = review_status(entry)
        pub_text, pub_tone = version_status(entry["publication"])
        st.html(
            '<div class="ctai-chip-row" style="justify-content:flex-start">'
            f"{theme.chip(pub_text, pub_tone)}"
            f"{theme.chip(analysis_text, analysis_tone)}"
            f"{theme.chip(review_text, review_tone)}</div>"
        )
        if entry["pdf_status"] == version_store.PDF_UNAVAILABLE:
            st.caption("Original PDF not stored for this version.")
            _render_reattach(protocol_id)
        if st.button(
            "Open Protocol",
            key=f"open_protocol_{protocol_id}",
            disabled=entry["guide"] is None,
        ):
            chosen = ("open", protocol_id)
        if protocol_id == st.session_state.get("coordinator_protocol_id"):
            st.caption("Close your work in progress before deleting this protocol.")
        elif st.button("Delete Protocol", key=f"delete_protocol_{protocol_id}"):
            _confirm_delete(entry)
    return chosen


@st.dialog("Delete protocol")
def _confirm_delete(entry: Mapping[str, Any]) -> None:
    protocol_id = entry["protocol_id"]
    name = protocol_deletion.study_name(protocol_id) or entry["name"] or ""
    label = entry["version_label"] or "—"
    st.markdown(f"Protocol: `{name}`  \nVersion: **{label}**")
    siblings = [e for e in review_store.list_protocols() if e["study_id"] == entry["study_id"]]
    options = {protocol_deletion.SCOPE_VERSION: f"Only version {label} (recommended)"}
    if len(siblings) > 1:
        options[protocol_deletion.SCOPE_STUDY] = f"The entire study ({len(siblings)} versions)"
    scope = st.radio(
        "What should be deleted?", list(options), format_func=options.get,
        key=f"delete_scope_{protocol_id}",
    )
    if scope == protocol_deletion.SCOPE_STUDY:
        st.warning("This permanently deletes every version of this study and its stored PDFs.")
    else:
        st.warning(
            f"This permanently deletes version {label}, its analysis, summary, approvals "
            "and stored PDF. Other versions are not changed."
        )
    if entry["publication"] == review_store.PUBLISHED or scope == protocol_deletion.SCOPE_STUDY and any(
        e["publication"] == review_store.PUBLISHED for e in siblings
    ):
        st.error(
            "A published version will be deleted: patients will lose access to this "
            "protocol. No older version is republished automatically."
        )
    typed = st.text_input("Type the protocol name exactly to confirm", key=f"delete_name_{protocol_id}")
    password = st.text_input(
        "Your account password", type="password", key=f"delete_password_{protocol_id}"
    )
    ready = typed == name and bool(password)
    if st.button(
        "Delete Protocol", type="primary", disabled=not ready, key=f"delete_confirm_{protocol_id}"
    ):
        try:
            result = protocol_deletion.delete_protocol(
                protocol_id, scope=scope, typed_name=typed, password=password
            )
        except (protocol_deletion.DeletionError, PermissionError) as exc:
            st.error(str(exc))
            return
        # Drop deleted ids from session caches so name reuse is not blocked by
        # a stale published_protocols entry after rerun.
        protocol_deletion.clear_session_after_delete(
            st.session_state, result.get("deleted_protocol_ids") or []
        )
        # Delayed import avoids a circular import with app.py at module load.
        from app import sync_published_from_store

        sync_published_from_store()
        if result["pdf_cleanup_incomplete"]:
            st.session_state["delete_notice"] = (
                "Deleted, but a stored PDF file could not be removed and needs manual cleanup."
            )
        else:
            st.session_state["delete_notice"] = "Protocol deleted."
        st.rerun()


def _render_reattach(protocol_id: str) -> None:
    """Reattach the original PDF; it must match the fingerprint saved with the record."""
    with st.expander("Reattach original PDF"):
        file = st.file_uploader(
            "Original PDF", type=["pdf"], key=f"reattach_upload_{protocol_id}",
            label_visibility="collapsed",
        )
        if file is not None and st.button("Attach PDF", key=f"reattach_btn_{protocol_id}"):
            try:
                version_store.reattach_pdf(protocol_id, file.getvalue(), file.name)
            except version_store.VersionError as exc:
                st.error(str(exc))
            except (sqlite3.Error, OSError) as exc:
                st.error(f"The PDF could not be stored: {exc}")
            else:
                st.rerun()


def suggest_next_label(labels: list[str]) -> str:
    """'2.0' after '1.0' (highest major + 1); empty when labels are not numeric."""
    majors = [int(m.group(1)) for l in labels if (m := re.match(r"^(\d+)\.\d+$", l))]
    return f"{max(majors) + 1}.0" if majors else ""


def render_version_picker() -> Optional[tuple[str, str]]:
    """
    Pick a saved clinical trial and a new version identifier.

    Returns ("version", "<study_id>|<label>") to continue into the upload
    workflow, ("add", "") to start a new trial, or None.
    """
    theme.page_head(
        "Upload a new version",
        "Choose a saved clinical trial, then upload its amended protocol as a new version.",
    )
    try:
        studies = version_store.list_studies()
    except (sqlite3.Error, OSError) as exc:
        st.error(f"Saved clinical trials could not be read from the database: {exc}")
        return None
    if not studies:
        st.info("No clinical trials have been saved yet.")
        if st.button("Add New Protocol", type="primary", key="version_add_btn"):
            return ("add", "")
        return None
    by_id = {s["study_id"]: s for s in studies}
    study_id = st.selectbox(
        "Clinical trial",
        options=list(by_id),
        key="version_study_select",
        format_func=lambda sid: by_id[sid]["title"],
        index=None,
        placeholder="Choose a clinical trial",
    )
    if study_id is None:
        return None
    study = by_id[study_id]
    current = study["current"]
    if current:
        when = format_saved_time(current["published_at"])
        st.success(
            f"Currently published: **Version {current['version_label']}**"
            + (f" · {when}" if when else "")
        )
    else:
        st.info("No version of this clinical trial is currently published.")
    with st.container(key="version_history"):
        st.markdown("**Version history**")
        for version in study["versions"]:
            text, tone = version_status(version["publication"])
            when = format_saved_time(version["uploaded_at"] or version["saved_at"])
            st.html(
                f'<div class="ctai-chip-row" style="justify-content:flex-start">'
                f"<b>Version {theme.e(version['version_label'])}</b>{theme.chip(text, tone)}"
                + (f"<span>{theme.e(when)}</span>" if when else "")
                + "</div>"
            )
    labels = [v["version_label"] for v in study["versions"]]
    label = st.text_input(
        "New version identifier",
        key=f"version_label_{study_id}",
        value=suggest_next_label(labels),
        max_chars=20,
        help=version_store.LABEL_RULES,
    ).strip()
    problem = None
    if not label:
        problem = "Enter a version identifier."
    else:
        try:
            version_store.validate_label(label)
        except version_store.VersionError as exc:
            problem = str(exc)
        else:
            if version_store.label_taken(study_id, label):
                problem = f"Version {label} already exists for this clinical trial. Choose a different identifier."
    if problem and label:
        st.warning(problem)
    if st.button(
        "Continue to upload",
        type="primary",
        key="version_continue_btn",
        disabled=problem is not None,
    ):
        return ("version", f"{study_id}|{label}")
    return None
