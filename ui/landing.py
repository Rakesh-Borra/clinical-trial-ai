"""
Welcome page: pick a role and sign in.

This is a demo sign-in for the hackathon prototype. There are no passwords or
accounts; choosing a role decides which workspace opens. The optional name is
only used to greet the person.
"""

from __future__ import annotations

import streamlit as st

from ui import theme

ROLE_KEY = "role"
NAME_KEY = "user_name"

COORDINATOR = "coordinator"
PATIENT = "patient"


def current_role() -> str | None:
    return st.session_state.get(ROLE_KEY)


def sign_in(role: str, name: str = "") -> None:
    st.session_state[ROLE_KEY] = role
    st.session_state[NAME_KEY] = (name or "").strip()
    st.rerun()


def sign_out() -> None:
    st.session_state[ROLE_KEY] = None
    st.session_state[NAME_KEY] = ""
    st.rerun()


def render_top_bar(role: str | None = None) -> None:
    """Brand on the left; role badge and 'Switch role' on the right."""
    with st.container(key="topbar"):
        left, mid, right = st.columns([5, 4, 1.7], vertical_alignment="center")
        left.html(theme.brand_html())
        if role:
            name = st.session_state.get(NAME_KEY) or ""
            label = "Coordinator" if role == COORDINATOR else "Patient"
            tone = "teal" if role == COORDINATOR else "coral"
            who = f" · {theme.e(name)}" if name else ""
            mid.html(f'<div class="ctai-chip-row">{theme.chip(label + " workspace", tone)}'
                     f'<span style="color:#48626A;font-weight:700">{who}</span></div>')
            if right.button("Switch role", key="switch_role_btn"):
                sign_out()
        else:
            mid.html(f'<div class="ctai-chip-row">{theme.chip("Research prototype", "sun")}</div>')


def render_landing() -> None:
    render_top_bar(None)

    st.html(
        '<div class="ctai-hero">'
        '<div class="ctai-blob" style="left:-30px;top:30px;width:110px;height:110px;background:#F2C14E"></div>'
        '<div class="ctai-blob" style="left:46%;top:-6px;width:64px;height:64px;background:#E27D60;animation-duration:7s"></div>'
        '<div class="ctai-hero-text">'
        '<div class="ctai-eyebrow">PEAAII AI Hackathon 2026 · Agentic AI</div>'
        '<h1 class="ctai-h1">Welcome to <em>Clinical Trial AI</em></h1>'
        '<p class="ctai-sub">Trial protocols that patients can understand, '
        '<b>checked by AI</b> and <b>approved by people</b>.</p>'
        '</div>'
        f'<div class="ctai-hero-art">{theme.hero_art()}</div>'
        '</div>'
    )

    st.html('<div class="ctai-section-title"><p>Get started</p><h2>Who is using Clinical Trial AI today?</h2></div>')
    st.info(
        "Demo role navigation only — choosing a coordinator or patient role "
        "does not authenticate users or restrict access."
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(key="card_coordinator"):
            st.html(
                '<div class="ctai-role-card coordinator">' + theme.icon_coordinator() +
                '<h3>Research coordinator</h3>'
                '<p>Turn a protocol into a patient guide, and decide what patients see.</p>'
                '<ul><li>Upload a document and choose its type</li>'
                '<li>Review the protocol summary, with page numbers</li>'
                '<li>Mark each AI statement accurate or not, then approve</li></ul></div>'
            )
            name = st.text_input("Your name (optional)", key="login_name_coordinator",
                                 placeholder="e.g. Dr. Rivera")
            if st.button("Sign in as coordinator", key="login_coordinator_btn", type="primary"):
                sign_in(COORDINATOR, name)
    with right:
        with st.container(key="card_patient"):
            st.html(
                '<div class="ctai-role-card patient">' + theme.icon_patient() +
                '<h3>Patient or caregiver</h3>'
                '<p>Read your study guide in everyday words and ask questions.</p>'
                '<ul><li>Study summary on the left, always easy to read</li>'
                '<li>A chat assistant on the right, sized how you like</li>'
                '<li>Only information your study team has approved</li></ul></div>'
            )
            name = st.text_input("Your name (optional)", key="login_name_patient",
                                 placeholder="e.g. Lakshmi")
            if st.button("Sign in as patient", key="login_patient_btn", type="primary"):
                sign_in(PATIENT, name)

    st.html('<div class="ctai-section-title"><p>Why you can trust it</p><h2>Built to be right, not just fast</h2></div>')
    f1, f2, f3 = st.columns(3, gap="medium")
    features = [
        (f1, theme.TEAL, theme.GLYPH_CHECK, "Source-checked",
         "Every statement is checked against, and cites, the protocol page it came from."),
        (f2, theme.CORAL, theme.GLYPH_LOCK, "Numbers protected",
         "A rewrite that drops or changes an age, dose or lab value is never published."),
        (f3, theme.LAVENDER, theme.GLYPH_USERS, "Human-approved",
         "A coordinator approves every item. Patients only ever see what was approved."),
    ]
    for col, color, glyph, title, text in features:
        col.html(f'<div class="ctai-feature">{theme.feature_dot(color, glyph)}<h4>{title}</h4><p>{text}</p></div>')

    st.html(
        '<div class="ctai-stats">'
        '<div class="ctai-stat"><b>25 / 25</b><span>risky questions answered safely</span></div>'
        '<div class="ctai-stat"><b>13 / 13</b><span>unsafe rewrites caught</span></div>'
        '<div class="ctai-stat"><b>91%</b><span>of extracted statements correct</span></div>'
        '<div class="ctai-stat"><b>3</b><span>real public protocols tested</span></div>'
        '</div>'
    )
    st.html(
        '<div class="ctai-footer">Research prototype for educational demonstration only. It does not give medical advice, '
        'diagnose conditions, or decide whether anyone is eligible for a trial. Demo sign-in: no password or account is used.'
        '<br>Clinical Trial AI · UMass Boston · Paul English Applied AI Institute</div>'
    )
