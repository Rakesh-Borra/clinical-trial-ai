"""Coordinator panel: compare an amended version with the one before it."""

from __future__ import annotations

import sqlite3

import streamlit as st

from models.plain_language_models import PatientGuide
from services import review_store, version_compare, version_store
from ui import theme

_EVIDENCE = {
    version_compare.VERIFIED: ("Verified in both PDFs", "teal"),
    version_compare.UNVERIFIED: ("Not found on the cited page", "coral"),
    version_compare.UNAVAILABLE: ("Not verified: original PDF unavailable", "sun"),
}
_KIND = {"added": "Added", "removed": "Removed", "modified": "Changed"}


def render_version_panel(guide: PatientGuide, protocol_id: str) -> None:
    try:
        previous = version_store.previous_version(protocol_id)
        saved = version_store.load_comparison(protocol_id) if previous else None
    except (sqlite3.Error, OSError) as exc:
        st.error(f"Version data could not be read: {exc}")
        return
    if previous is None:
        return
    label = previous["version_label"]
    with st.container(key="version_panel"):
        st.html(
            f'<div class="ctai-label">Version changes</div>'
            f'<h3 style="margin:0 0 .3rem">What changed since Version {theme.e(label)}</h3>'
        )
        st.caption(
            "Every addition, removal and change is checked against the original PDFs "
            "of both versions. Publishing this version requires an approved comparison."
        )
        if st.button(f"Compare with Version {label}", key="version_compare_btn"):
            try:
                with st.spinner("Comparing the two versions…"):
                    version_compare.compare_versions(protocol_id)
                review_store.log_action(protocol_id, "version_compared", f"vs {label}")
            except version_store.VersionError as exc:
                st.error(str(exc))
            except (sqlite3.Error, OSError) as exc:
                st.error(f"The comparison could not be saved: {exc}")
            st.rerun()
        if saved is None or saved["previous_protocol_id"] != previous["protocol_id"]:
            st.info("This version has not been compared yet.")
            return
        _render_saved(saved, guide, protocol_id)


def _render_saved(saved, guide: PatientGuide, protocol_id: str) -> None:
    diff = saved["diff"]
    changes = diff["changes"]
    if not diff["source_backed"]:
        st.warning(diff["notice"])
    st.caption(
        f"{len(changes)} change(s) between Version {diff['previous_label']} and "
        f"Version {diff['new_label']}."
        if changes
        else f"No differences found between Version {diff['previous_label']} and Version {diff['new_label']}."
    )
    for index, change in enumerate(changes):
        text, tone = _EVIDENCE[change["evidence"]]
        pages = [
            f"Version {diff[key]} page {side['page']}"
            for key, side in (("previous_label", change["old"]), ("new_label", change["new"]))
            if side and side.get("page")
        ]
        with st.expander(f"{_KIND[change['change']]} · {change['section']}", expanded=False):
            st.html(theme.chip(text, tone))
            st.markdown(change["explanation"])
            if pages:
                st.caption("Source: " + "; ".join(pages))

    stale = saved["diff"].get("guide_fingerprint") != version_compare.guide_fingerprint(guide)
    if stale:
        st.warning(
            "Statements changed after this comparison. Compare again before approving."
        )
        return
    if saved["approved_at"]:
        st.success(f"Version changes approved ({saved['approved_by'] or 'coordinator'}).")
        return
    blocker = version_compare.approval_block_reason(saved)
    if blocker:
        st.error(blocker)
    if st.button(
        "Approve version changes",
        type="primary",
        key="version_approve_btn",
        disabled=blocker is not None,
    ):
        try:
            version_store.approve_comparison(protocol_id)
            review_store.log_action(
                protocol_id,
                "version_changes_approved",
                "source-backed" if diff["source_backed"] else "without PDF verification",
            )
        except (sqlite3.Error, OSError) as exc:
            st.error(f"The approval could not be saved: {exc}")
        st.rerun()
