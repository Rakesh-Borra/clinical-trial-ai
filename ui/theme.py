"""
Look and feel for the Clinical Trial AI web app.

One place for the colours, fonts, CSS and small HTML pieces (logo, top bar,
badges) so every screen looks the same. Nothing here touches protocol data.
"""

from __future__ import annotations

import base64
import html
from typing import Optional

import streamlit as st

INK = "#183A43"
BODY = "#48626A"
TEAL = "#2A7D74"
TEAL_TEXT = "#23706A"
CORAL = "#E27D60"
CORAL_TEXT = "#B4523A"
SUN = "#F2C14E"
LAVENDER = "#A9A3D9"
MINT = "#E9F4F1"
CREAM = "#FBF7F0"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Nunito+Sans:wght@300..800&display=swap');

:root {
  --ink:#183A43; --body:#48626A; --teal:#2A7D74; --teal-text:#23706A;
  --coral:#E27D60; --coral-text:#B4523A; --sun:#F2C14E; --lav:#A9A3D9;
  --mint:#E9F4F1; --cream:#FBF7F0; --line:#DCE8E4;
  --shadow:0 14px 40px rgba(24,58,67,.10);
}

html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {
  font-family:'Nunito Sans', -apple-system, 'Segoe UI', Arial, sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 520px at 88% -8%, rgba(169,163,217,.28), transparent 60%),
    radial-gradient(900px 480px at -6% 4%, rgba(242,193,78,.20), transparent 60%),
    linear-gradient(180deg, #EAF5F2 0%, #F6FAF8 38%, #FBF7F0 100%);
  color:var(--ink);
}
.block-container { max-width:1240px; padding-top:1.4rem; padding-bottom:4rem; }

/* hide default Streamlit chrome so it feels like a real product */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; height:0; }

h1, h2, h3, .ctai-serif {
  font-family:'DM Serif Display', Georgia, serif !important;
  font-weight:400 !important; color:var(--ink); letter-spacing:-.4px;
}
h2 { font-size:2.1rem !important; } h3 { font-size:1.5rem !important; }
p, li { color:var(--body); }
a { color:var(--teal-text); }

/* ---------- top bar ---------- */
.st-key-topbar {
  background:rgba(255,255,255,.82); backdrop-filter:blur(10px);
  border:1px solid var(--line); border-radius:22px; padding:.55rem 1.1rem;
  box-shadow:0 8px 28px rgba(24,58,67,.07); margin-bottom:1.4rem;
}
.ctai-brand { display:flex; align-items:center; gap:.7rem; }
.ctai-brand b { font-family:'DM Serif Display',Georgia,serif; font-weight:400; font-size:1.45rem; color:var(--ink); letter-spacing:-.3px; }
.ctai-brand small { display:block; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:var(--teal-text); font-weight:800; margin-top:-2px; }
.ctai-chip-row { display:flex; justify-content:flex-end; align-items:center; gap:.6rem; flex-wrap:wrap; }
.ctai-chip { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .8rem; border-radius:999px; font-size:.82rem; font-weight:800; }
.ctai-chip.teal { background:var(--mint); color:var(--teal-text); }
.ctai-chip.coral { background:#FBE3DA; color:var(--coral-text); }
.ctai-chip.sun { background:#FBF0CF; color:#7A5A00; }
.ctai-chip.lav { background:#E7E5F6; color:#4B4685; }
.ctai-chip.ink { background:var(--ink); color:#fff; }
.ctai-chip.grey { background:#EEF2F1; color:var(--body); }

/* ---------- landing hero ---------- */
.ctai-hero { position:relative; display:flex; align-items:center; justify-content:space-between; gap:2rem; padding:2.2rem 1rem 1.2rem; animation:ctaiUp .8s ease both; }
.ctai-hero-text { max-width:640px; position:relative; z-index:2; }
.ctai-eyebrow { font-size:.9rem; font-weight:800; letter-spacing:.2em; text-transform:uppercase; color:var(--teal-text); margin-bottom:.9rem; }
.ctai-h1 { font-family:'DM Serif Display',Georgia,serif; font-size:4.3rem; line-height:1.02; letter-spacing:-1.5px; color:var(--ink); margin:0 0 1.1rem; }
.ctai-h1 em { font-style:normal; color:var(--teal); }
.ctai-sub { font-size:1.3rem; line-height:1.5; color:var(--body); font-weight:300; margin:0; }
.ctai-sub b { color:var(--ink); font-weight:700; }
.ctai-hero-art { flex:none; animation:ctaiFloat 6s ease-in-out infinite; }
.ctai-blob { z-index:0; position:absolute; border-radius:50%; opacity:.32; animation:ctaiFloat 9s ease-in-out infinite; pointer-events:none; }

.ctai-section-title { text-align:center; margin:2.2rem 0 1.2rem; }
.ctai-section-title p { font-size:.85rem; font-weight:800; letter-spacing:.2em; text-transform:uppercase; color:var(--teal-text); margin:0 0 .3rem; }
.ctai-section-title h2 { margin:0; }

/* role cards */
.st-key-card_coordinator, .st-key-card_patient {
  background:#fff; border:1px solid var(--line); border-radius:30px; padding:2rem 2rem 1.6rem;
  box-shadow:var(--shadow); transition:transform .25s ease, box-shadow .25s ease; animation:ctaiUp .9s ease both;
}
.st-key-card_coordinator:hover, .st-key-card_patient:hover { transform:translateY(-5px); box-shadow:0 22px 54px rgba(24,58,67,.16); }
.ctai-role-icon { width:68px; height:68px; border-radius:22px; display:flex; align-items:center; justify-content:center; margin-bottom:1rem; }
.ctai-role-card h3 { margin:.2rem 0 .4rem; font-size:1.9rem !important; }
.ctai-role-card p { margin:0 0 .9rem; font-size:1.05rem; line-height:1.5; }
.ctai-role-card ul { list-style:none; padding:0; margin:0 0 1.1rem; }
.ctai-role-card li { padding:.28rem 0 .28rem 1.6rem; position:relative; font-size:1rem; color:var(--body); }
.ctai-role-card li::before { content:''; position:absolute; left:0; top:.62rem; width:.62rem; height:.62rem; border-radius:50%; background:var(--teal); }
.ctai-role-card.patient li::before { background:var(--coral); }

.ctai-feature { background:rgba(255,255,255,.7); border:1px solid var(--line); border-radius:24px; padding:1.4rem 1.5rem; height:100%; }
.ctai-feature h4 { margin:.7rem 0 .3rem; font-size:1.15rem; font-weight:800; color:var(--ink); }
.ctai-feature p { margin:0; font-size:.98rem; line-height:1.5; }
.ctai-dot { width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center; }

.ctai-stats { display:flex; gap:1rem; flex-wrap:wrap; justify-content:center; margin:1.2rem 0 .4rem; }
.ctai-stat { background:#fff; border:1px solid var(--line); border-radius:20px; padding:.9rem 1.5rem; text-align:center; min-width:170px; }
.ctai-stat b { display:block; font-family:'DM Serif Display',Georgia,serif; font-weight:400; font-size:2rem; color:var(--teal); line-height:1.1; }
.ctai-stat span { font-size:.88rem; color:var(--body); font-weight:600; }

.ctai-footer { text-align:center; color:var(--body); font-size:.9rem; margin-top:2.4rem; padding-top:1.2rem; border-top:1px solid var(--line); }

/* ---------- workspace ---------- */
.ctai-page-head { margin:.2rem 0 1rem; }
.ctai-page-head p { margin:0; font-size:1.05rem; }
.ctai-page-head h2 { margin:0 0 .25rem; }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius:22px; }
.st-key-upload_card, .st-key-doctype_card, .st-key-guide_panel, .st-key-chat_panel {
  background:#fff; border:1px solid var(--line); border-radius:26px; padding:1.5rem 1.6rem; box-shadow:var(--shadow);
}

/* segmented control used as the step bar */
[data-testid="stButtonGroup"] button, [data-testid="stSegmentedControl"] button { font-weight:800; }

.ctai-metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:.4rem 0 1.2rem; }
.ctai-metric { background:#fff; border:1px solid var(--line); border-radius:20px; padding:.9rem 1.2rem; }
.ctai-metric span { font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--teal-text); }
.ctai-metric b { display:block; font-family:'DM Serif Display',Georgia,serif; font-weight:400; font-size:2.1rem; line-height:1.15; color:var(--ink); }
.ctai-metric.warn b { color:var(--coral-text); }
.ctai-progress { height:10px; border-radius:99px; background:#DDE9E5; overflow:hidden; margin:.2rem 0 1.1rem; }
.ctai-progress > div { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--teal),#59B3A6); transition:width .5s ease; }

.ctai-quote { border-left:5px solid var(--lav); background:#F5F4FB; border-radius:0 16px 16px 0; padding:.8rem 1.1rem; color:var(--ink); font-size:1.02rem; margin:.3rem 0 .8rem; }
.ctai-plain { border-left:5px solid var(--teal); background:var(--mint); border-radius:0 16px 16px 0; padding:.8rem 1.1rem; color:var(--ink); font-size:1.08rem; font-weight:600; margin:.3rem 0 .8rem; }
.ctai-label { font-size:.74rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; color:var(--body); margin:.2rem 0 .1rem; }
.ctai-check { display:flex; gap:.5rem; align-items:flex-start; padding:.45rem .8rem; border-radius:12px; font-size:.95rem; margin:.15rem 0; }
.ctai-check.ok { background:#E6F4EE; color:#1E6B50; }
.ctai-check.bad { background:#FFF3D6; color:#7A5A00; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius:999px; font-weight:800; padding:.55rem 1.4rem; border:1.5px solid var(--line); background:#fff; color:var(--ink);
  transition:transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 8px 20px rgba(24,58,67,.12); border-color:var(--teal); color:var(--teal-text); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg,#2A7D74,#1F6A62); color:#fff; border:none; box-shadow:0 10px 24px rgba(42,125,116,.32);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { color:#fff; box-shadow:0 14px 30px rgba(42,125,116,.42); }
.st-key-card_coordinator .stButton > button, .st-key-card_patient .stButton > button { width:100%; padding:.75rem 1.4rem; font-size:1.05rem; }
.st-key-card_patient .stButton > button[data-testid="stBaseButton-primary"] { background:linear-gradient(135deg,#E27D60,#CF6446); box-shadow:0 10px 24px rgba(226,125,96,.35); }
.stButton > button[data-testid="stBaseButton-primary"] p, .stButton > button[data-testid="stBaseButton-primary"] span { color:#fff !important; }
.st-key-chat_panel .stButton > button { white-space:normal; text-align:left; height:auto; min-height:2.4rem; line-height:1.25; padding:.5rem .9rem; }
.st-key-chat_panel .stButton > button p { white-space:normal; overflow:visible; text-overflow:clip; }
.st-key-reject_style button { color:var(--coral-text); border-color:#F1C5B8; }

/* inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div { border-radius:14px; }
[data-testid="stFileUploaderDropzone"] { border-radius:20px; border:2px dashed #9FCBC3; background:#F4FAF8; }
[data-testid="stChatInput"] { border-radius:18px; }
[data-testid="stChatMessage"] { background:transparent; }
[data-testid="stExpander"] { border-radius:18px; border:1px solid var(--line); background:rgba(255,255,255,.7); }
[data-testid="stAlert"] { border-radius:16px; }

/* patient guide cards */
.ctai-guide-head { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin-bottom:.4rem; }
.ctai-guide-head h2 { margin:0; font-size:1.9rem !important; }
.ctai-guide-sec { margin:1.1rem 0 .4rem; display:flex; align-items:center; gap:.6rem; }
.ctai-guide-sec h3 { margin:0; font-size:1.35rem !important; }
.ctai-item { background:#fff; border:1px solid var(--line); border-radius:18px; padding:.85rem 1.1rem; margin:.45rem 0; color:var(--ink); font-size:1.04rem; line-height:1.5; box-shadow:0 4px 14px rgba(24,58,67,.05); }
.ctai-item small { display:block; color:var(--body); margin-top:.25rem; font-size:.82rem; font-weight:700; }
.ctai-pending { background:#FFF8E5; border:1px dashed #E8C766; color:#7A5A00; border-radius:16px; padding:.7rem 1rem; font-size:.95rem; margin:.5rem 0; }
.ctai-empty { text-align:center; padding:2.4rem 1rem; }
.ctai-empty h3 { margin:.8rem 0 .3rem; }
.ctai-terms { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.4rem; }
.ctai-term { background:var(--mint); border-radius:14px; padding:.5rem .8rem; font-size:.92rem; color:var(--ink); }
.ctai-term b { color:var(--teal-text); }
.ctai-chat-hello { background:linear-gradient(135deg,#E9F4F1,#F6F4FD); border-radius:20px; padding:1rem 1.2rem; color:var(--ink); margin-bottom:.6rem; }

@keyframes ctaiUp { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:none; } }
@keyframes ctaiFloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-14px); } }
@media (max-width: 900px) {
  .ctai-hero { flex-direction:column; text-align:left; } .ctai-h1 { font-size:2.9rem; } .ctai-hero-art { display:none; }
  .ctai-metrics { grid-template-columns:repeat(2,1fr); }
}
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
"""

DARK_WORKSPACE_CSS = """
:root {
  color-scheme: dark;
}
.stApp:not(:has(.st-key-landing_shell)) {
  --ink:#E8EDF2;
  --body:#A6B0BC;
  --teal:#A9C9D6;
  --teal-text:#B9D5DF;
  --coral:#E9A18C;
  --coral-text:#F1B19E;
  --sun:#D9BD7A;
  --lav:#B7B8D8;
  --mint:rgba(112,153,167,.13);
  --cream:#090B0F;
  --line:rgba(205,215,230,.12);
  --shadow:0 22px 60px rgba(0,0,0,.2);
  min-height:100svh;
  overflow-x:hidden;
  color:var(--ink);
  background:
    radial-gradient(ellipse at 13% 22%, rgba(87,111,143,.14), transparent 38%),
    radial-gradient(ellipse at 75% 83%, rgba(64,113,126,.10), transparent 38%),
    #090B0F;
}
.stApp:not(:has(.st-key-landing_shell)),
.stApp:not(:has(.st-key-landing_shell)) p,
.stApp:not(:has(.st-key-landing_shell)) label,
.stApp:not(:has(.st-key-landing_shell)) input,
.stApp:not(:has(.st-key-landing_shell)) textarea,
.stApp:not(:has(.st-key-landing_shell)) button {
  font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
.stApp:not(:has(.st-key-landing_shell)) .block-container {
  max-width:1440px;
  padding:1.2rem clamp(1.25rem,5vw,5.5rem) 3rem;
}
.stApp:not(:has(.st-key-landing_shell)) h1,
.stApp:not(:has(.st-key-landing_shell)) h2,
.stApp:not(:has(.st-key-landing_shell)) h3,
.stApp:not(:has(.st-key-landing_shell)) h4 {
  font-family:'Manrope',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
  font-weight:600 !important;
  letter-spacing:-.045em;
}
.stApp:not(:has(.st-key-landing_shell)) h1,
.stApp:not(:has(.st-key-landing_shell)) h2,
.stApp:not(:has(.st-key-landing_shell)) h3,
.stApp:not(:has(.st-key-landing_shell)) h4,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stMarkdownContainer"] strong {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) p,
.stApp:not(:has(.st-key-landing_shell)) li,
.stApp:not(:has(.st-key-landing_shell)) label,
.stApp:not(:has(.st-key-landing_shell)) small,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stCaptionContainer"] {
  color:var(--body);
}
.stApp:not(:has(.st-key-landing_shell)) a {
  color:var(--teal-text);
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-topbar {
  padding:0 .6rem;
  min-height:48px;
  max-height:50px;
  display:flex;
  align-items:center;
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(19,22,28,.78);
  box-shadow:0 12px 50px rgba(0,0,0,.18);
  backdrop-filter:blur(18px);
  margin-bottom:.9rem;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-topbar .stButton > button {
  width:auto;
  min-height:2.1rem;
  padding:.2rem .7rem;
  border:1px solid transparent;
  border-radius:12px;
  background:transparent;
  color:#F1F3F6;
  font-size:.92rem;
  font-weight:700;
  box-shadow:none;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-topbar .stButton > button:hover {
  border-color:var(--line);
  background:rgba(255,255,255,.035);
  color:#F1F3F6;
  transform:none;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-nav_previous button,
.stApp:not(:has(.st-key-landing_shell)) .st-key-nav_next button {
  min-width:2.4rem;
  padding:.35rem .4rem;
  font-size:1.1rem;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-topbar [data-testid="stPopoverButton"] {
  width:100%;
  min-height:2.1rem;
  border:1px solid var(--line);
  border-radius:12px;
  background:rgba(255,255,255,.035);
  color:var(--ink);
  font-weight:650;
  box-shadow:none;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-topbar [data-testid="stPopoverButton"]:hover {
  border-color:rgba(169,201,214,.42);
  background:rgba(169,201,214,.08);
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-brand,
.stApp:not(:has(.st-key-landing_shell)) .ctai-brand b {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-brand small,
.stApp:not(:has(.st-key-landing_shell)) .ctai-label,
.stApp:not(:has(.st-key-landing_shell)) .ctai-metric span {
  color:var(--teal-text);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-chip.teal,
.stApp:not(:has(.st-key-landing_shell)) .ctai-chip.coral,
.stApp:not(:has(.st-key-landing_shell)) .ctai-chip.sun,
.stApp:not(:has(.st-key-landing_shell)) .ctai-chip.lav,
.stApp:not(:has(.st-key-landing_shell)) .ctai-chip.ink,
.stApp:not(:has(.st-key-landing_shell)) .ctai-chip.grey {
  border:1px solid var(--line);
  background:rgba(255,255,255,.035);
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-page-head {
  margin:.4rem 0 1.25rem;
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-page-head h2 {
  color:var(--ink);
  font-size:clamp(1.8rem,3vw,2.4rem) !important;
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-page-head p {
  color:var(--body);
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-upload_card,
.stApp:not(:has(.st-key-landing_shell)) .st-key-doctype_card,
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_panel,
.stApp:not(:has(.st-key-landing_shell)) .st-key-chat_panel,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stVerticalBlockBorderWrapper"] {
  border:1px solid var(--line);
  background:linear-gradient(145deg,rgba(255,255,255,.035),transparent 55%),rgba(18,21,27,.76);
  box-shadow:var(--shadow);
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-upload_card,
.stApp:not(:has(.st-key-landing_shell)) .st-key-doctype_card,
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_panel,
.stApp:not(:has(.st-key-landing_shell)) .st-key-chat_panel {
  border-radius:22px;
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-metric,
.stApp:not(:has(.st-key-landing_shell)) .ctai-item,
.stApp:not(:has(.st-key-landing_shell)) .ctai-term,
.stApp:not(:has(.st-key-landing_shell)) .ctai-chat-hello,
.stApp:not(:has(.st-key-landing_shell)) .ctai-quote,
.stApp:not(:has(.st-key-landing_shell)) .ctai-plain,
.stApp:not(:has(.st-key-landing_shell)) .ctai-pending,
.stApp:not(:has(.st-key-landing_shell)) .ctai-check.ok,
.stApp:not(:has(.st-key-landing_shell)) .ctai-check.bad {
  border:1px solid var(--line);
  background:rgba(255,255,255,.035);
  color:var(--ink);
  box-shadow:none;
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-quote {
  border-left:3px solid var(--lav);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-plain {
  border-left:3px solid var(--teal);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-check.ok,
.stApp:not(:has(.st-key-landing_shell)) .ctai-check.bad,
.stApp:not(:has(.st-key-landing_shell)) .ctai-pending {
  color:#E8EDF2;
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-metric b {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-metric.warn b {
  color:var(--coral);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-progress {
  background:rgba(255,255,255,.08);
}
.stApp:not(:has(.st-key-landing_shell)) .ctai-progress > div {
  background:linear-gradient(90deg,#7899A7,#A9C9D6);
}
.stApp:not(:has(.st-key-landing_shell)) .stButton > button,
.stApp:not(:has(.st-key-landing_shell)) .stDownloadButton > button,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stPopover"] > button {
  border:1px solid var(--line);
  border-radius:12px;
  background:rgba(255,255,255,.035);
  color:var(--ink);
  box-shadow:none;
  transition:border-color .2s ease,background .2s ease,transform .2s ease;
}
.stApp:not(:has(.st-key-landing_shell)) .stButton > button:hover,
.stApp:not(:has(.st-key-landing_shell)) .stDownloadButton > button:hover {
  border-color:rgba(169,201,214,.42);
  background:rgba(169,201,214,.08);
  color:var(--ink);
  box-shadow:0 8px 24px rgba(0,0,0,.16);
  transform:translateY(-1px);
}
.stApp:not(:has(.st-key-landing_shell)) .stButton > button[data-testid="stBaseButton-primary"],
.stApp:not(:has(.st-key-landing_shell)) .stDownloadButton > button[data-testid="stBaseButton-primary"] {
  border-color:rgba(169,201,214,.32);
  background:linear-gradient(135deg,rgba(169,201,214,.2),rgba(169,201,214,.08));
  color:#F1F3F6;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-publish_protocol_btn button[data-testid="stBaseButton-primary"] {
  border-color:rgba(169,201,214,.48);
  background:linear-gradient(135deg,#587D8C,#344F5C);
  box-shadow:0 8px 24px rgba(60,95,110,.28);
  color:#F7FAFC;
}
/* statement filters: a wrapping row of pills; the selected one is teal */
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_filters { flex-wrap:wrap; row-gap:.5rem; }
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_filters > div { flex:0 0 auto; width:auto; }
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_filters button {
  min-height:2.25rem; padding:.2rem .95rem; border-radius:999px; white-space:nowrap;
  background:rgba(255,255,255,.03); border:1px solid rgba(205,215,230,.14); color:#B8C2CF;
}
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_filters button:hover { color:var(--ink); border-color:rgba(110,190,185,.45); }
.stApp:not(:has(.st-key-landing_shell)) .st-key-guide_filters button[data-testid="stBaseButton-primary"] {
  background:rgba(42,143,138,.24); border-color:rgba(94,201,194,.6); color:#E6F7F5; box-shadow:0 0 0 1px rgba(94,201,194,.18) inset;
}
@media (max-width: 640px) {
  .stApp:not(:has(.st-key-landing_shell)) .st-key-guide_filters button { padding:.2rem .7rem; font-size:.85rem; }
}
.stApp:not(:has(.st-key-landing_shell)) .stButton > button:disabled,
.stApp:not(:has(.st-key-landing_shell)) .stDownloadButton > button:disabled {
  border-color:rgba(205,215,230,.07);
  background:rgba(255,255,255,.018);
  color:#737E8B;
  opacity:1;
  box-shadow:none;
}
.stApp:not(:has(.st-key-landing_shell)) .stTextInput input,
.stApp:not(:has(.st-key-landing_shell)) .stTextArea textarea,
.stApp:not(:has(.st-key-landing_shell)) .stSelectbox [data-baseweb="select"] > div,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stChatInput"] textarea {
  border:1px solid var(--line);
  border-radius:12px;
  background:rgba(6,8,11,.42);
  color:var(--ink);
  box-shadow:none;
}
.stApp:not(:has(.st-key-landing_shell)) input::placeholder,
.stApp:not(:has(.st-key-landing_shell)) textarea::placeholder {
  color:#7D8793;
}
.stApp:not(:has(.st-key-landing_shell)) .stTextInput input:focus,
.stApp:not(:has(.st-key-landing_shell)) .stTextArea textarea:focus,
.stApp:not(:has(.st-key-landing_shell)) .stSelectbox [data-baseweb="select"] > div:focus-within {
  border-color:rgba(169,201,214,.55);
  box-shadow:0 0 0 1px rgba(169,201,214,.2);
}
.stApp:not(:has(.st-key-landing_shell)) .stSelectbox [role="group"] {
  border:1px solid var(--line) !important;
  border-radius:12px;
  background:rgba(6,8,11,.42) !important;
  color:var(--ink) !important;
}
.stApp:not(:has(.st-key-landing_shell)) .stSelectbox input[role="combobox"] {
  background:transparent !important;
  color:var(--ink) !important;
}
[role="listbox"] {
  border:1px solid rgba(205,215,230,.12) !important;
  border-radius:12px;
  background:#14171D !important;
  color:#E8EDF2 !important;
}
[role="option"] {
  background:#14171D !important;
  color:#E8EDF2 !important;
}
[role="option"][aria-selected="true"],
[role="option"]:hover {
  background:rgba(169,201,214,.13) !important;
  color:#F1F3F6 !important;
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stFileUploaderDropzone"] {
  border:1px dashed rgba(169,201,214,.34);
  border-radius:16px;
  background:rgba(255,255,255,.025);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stFileUploaderDropzone"] button {
  border:1px solid var(--line);
  background:rgba(255,255,255,.045);
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [role="radiogroup"] {
  display:flex;
  gap:.18rem;
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(18,21,27,.76);
  padding:.22rem;
}
.stApp:not(:has(.st-key-landing_shell)) [role="radiogroup"] button[role="radio"] {
  border-radius:10px;
  border:1px solid transparent;
  background:transparent;
  color:var(--body);
  box-shadow:none;
}
.stApp:not(:has(.st-key-landing_shell)) [role="radiogroup"] button[role="radio"]:hover {
  border-color:var(--line);
  background:rgba(255,255,255,.035);
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [role="radiogroup"] button[role="radio"][aria-checked="true"] {
  border-color:rgba(169,201,214,.24);
  background:rgba(169,201,214,.14);
  color:var(--ink);
}
[data-testid="stPopoverBody"],
[role="dialog"] {
  border:1px solid rgba(205,215,230,.12) !important;
  border-radius:14px;
  background:#14171D !important;
  color:#E8EDF2 !important;
}
[role="dialog"] p {
  color:#A6B0BC !important;
}
[role="dialog"] .stButton > button {
  width:100%;
  border:1px solid rgba(205,215,230,.16);
  border-radius:10px;
  background:rgba(255,255,255,.045);
  color:#E8EDF2;
}
[role="dialog"] .stButton > button:hover {
  border-color:rgba(169,201,214,.42);
  background:rgba(169,201,214,.1);
  color:#F1F3F6;
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stAlert"] {
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(255,255,255,.035);
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stAlert"] p,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stAlert"] li {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stExpander"] {
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(18,21,27,.72);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stExpander"] summary,
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stExpander"] summary:hover {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stChatMessage"] {
  border:1px solid rgba(205,215,230,.07);
  border-radius:16px;
  background:rgba(255,255,255,.025);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stChatInput"] {
  border:1px solid var(--line);
  border-radius:16px;
  background:rgba(18,21,27,.78);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stDataFrame"],
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stTable"] {
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(18,21,27,.76);
}
.stApp:not(:has(.st-key-landing_shell)) table {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) th {
  background:rgba(255,255,255,.055);
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) td {
  background:rgba(255,255,255,.018);
  color:var(--body);
}
.stApp:not(:has(.st-key-landing_shell)) [data-baseweb="popover"],
.stApp:not(:has(.st-key-landing_shell)) [data-baseweb="menu"],
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stPopoverBody"] {
  border:1px solid var(--line);
  border-radius:14px;
  background:#14171D;
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [role="option"],
.stApp:not(:has(.st-key-landing_shell)) [role="menuitem"] {
  color:var(--ink);
}
.stApp:not(:has(.st-key-landing_shell)) [role="option"]:hover,
.stApp:not(:has(.st-key-landing_shell)) [role="menuitem"]:hover {
  background:rgba(169,201,214,.1);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stProgress"] > div {
  background-color:rgba(255,255,255,.08);
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stProgress"] [role="progressbar"] {
  background-color:#A9C9D6;
}
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stSlider"] [data-testid="stThumbValue"],
.stApp:not(:has(.st-key-landing_shell)) [data-testid="stSlider"] [role="slider"] {
  color:var(--ink);
}
@media (max-width: 640px) {
  .stApp:not(:has(.st-key-landing_shell)) .block-container {
    padding:.75rem 1rem 1.5rem;
  }
  .stApp:not(:has(.st-key-landing_shell)) .st-key-topbar [data-testid="stHorizontalBlock"] {
    flex-direction:row !important;
    flex-wrap:nowrap !important;
    gap:.4rem;
  }
  .stApp:not(:has(.st-key-landing_shell)) .st-key-topbar [data-testid="stColumn"]:first-child {
    flex:1 1 auto !important;
    width:auto !important;
    min-width:0;
  }
  .stApp:not(:has(.st-key-landing_shell)) .st-key-topbar [data-testid="stColumn"]:last-child {
    flex:0 0 108px !important;
    width:108px !important;
    min-width:0 !important;
  }
  .stApp:not(:has(.st-key-landing_shell)) [role="radiogroup"] {
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
  }
  .stApp:not(:has(.st-key-landing_shell)) [role="radiogroup"] button[role="radio"] {
    min-width:0;
  }
  .stApp:not(:has(.st-key-landing_shell)) .st-key-topbar {
    margin-bottom:1.2rem;
  }
  .stApp:not(:has(.st-key-landing_shell)) .ctai-metrics {
    grid-template-columns:repeat(2,minmax(0,1fr));
  }
}

/* ---------- patient portal: fixed, resizable split pane ---------- */
html { --ctai-chat-w:34vw; --ctai-top:5.5rem; --ctai-vh:100dvh; --ctai-vt:0px; }
html.ctai-dragging, html.ctai-dragging * { user-select:none !important; cursor:col-resize !important; }
html:not(:has(.st-key-chat_panel)) .ctai-divider,
html:not(:has(.st-key-chat_panel)) .ctai-reopen { display:none !important; }

/* header stays reachable while the guide scrolls */
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) { position:sticky; top:.6rem; z-index:200; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-topbar { background:#0E1218; }

/* guide fills the space left of the assistant */
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stMainBlockContainer"] {
  max-width:none;
  padding-left:clamp(1rem,2.2vw,2rem);
  padding-right:calc(var(--ctai-chat-w) + clamp(1rem,2.2vw,2rem));
}
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) { width:calc(100% + var(--ctai-chat-w)); max-width:none; }
html[data-ctai-chat="min"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stMainBlockContainer"],
html[data-ctai-chat="max"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stMainBlockContainer"] { padding-right:clamp(1rem,2.2vw,2rem); }
html[data-ctai-chat="min"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
html[data-ctai-chat="max"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) { width:100%; }

.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-guide_panel { background:transparent; border:0; box-shadow:none; padding:0; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-guide-sub { margin:.35rem 0 0; font-size:1.02rem; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-guide_panel [data-testid="stExpander"] {
  border:1px solid var(--line); border-radius:18px; background:rgba(18,21,27,.76); margin:.7rem 0;
}
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-guide_panel [data-testid="stExpander"] summary { padding:.85rem 1.1rem; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-guide_panel [data-testid="stExpander"] summary p { font-size:1.3rem; font-weight:600; color:var(--ink); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-guide_panel [data-testid="stExpander"] summary svg { color:var(--teal); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-guide_panel [data-testid="stExpanderDetails"] { padding:0 1.1rem 1rem; }

/* the assistant: out of flow, pinned to the viewport */
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel {
  position:fixed; top:var(--ctai-top); right:0; bottom:0; width:var(--ctai-chat-w);
  z-index:150; box-sizing:border-box; margin:0;
  display:flex; flex-direction:column; gap:.55rem; overflow:hidden;
  padding:.8rem 1.25rem 1rem; border-radius:0; border-width:0 0 0 1px;
  background:#0B0E13; box-shadow:none;
}
/* chat bubbles: patient right, assistant left */
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_user_"], .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_assistant_"] { width:100%; min-width:0; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_"] [data-testid="stChatMessage"] { background:transparent; border:0; padding:.15rem 0; gap:.55rem; align-items:flex-start; min-width:0; max-width:100%; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_user_"] [data-testid="stChatMessage"] { flex-direction:row-reverse; justify-content:flex-start; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_"] [data-testid="stChatMessageContent"] { flex:0 1 auto; width:auto; max-width:calc(100% - 3rem); min-width:0; margin:0; padding:.6rem .85rem; border-radius:16px; overflow-wrap:anywhere; word-break:break-word; line-height:1.5; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_user_"] [data-testid="stChatMessageContent"] { background:rgba(63,127,191,.22); border:1px solid rgba(110,170,230,.28); border-bottom-right-radius:5px; text-align:left; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_assistant_"] [data-testid="stChatMessageContent"] { background:rgba(255,255,255,.05); border:1px solid rgba(205,215,230,.10); border-bottom-left-radius:5px; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_"] [data-testid="stChatMessageContent"] * { max-width:100%; overflow-wrap:anywhere; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_"] [data-testid="stChatMessageAvatarCustom"] { width:2rem; height:2rem; min-width:2rem; border-radius:50%; flex:none; color:#E8F4F2; border:1px solid rgba(255,255,255,.12); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_user_"] [data-testid="stChatMessageAvatarCustom"] { background:#2F6E9E; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [class*="st-key-msg_assistant_"] [data-testid="stChatMessageAvatarCustom"] { background:#2A8F8A; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-chat_messages) { flex:1 1 0; min-height:0; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_messages { height:100%; min-height:0; overflow-y:auto; overscroll-behavior:contain; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-chat-title { margin:0 0 .3rem !important; font-size:1.55rem !important; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-chat-controls { display:flex; justify-content:flex-end; gap:.25rem; margin:0 -.3rem; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-chat-controls button {
  width:30px; height:30px; padding:0; display:grid; place-items:center; cursor:pointer;
  background:transparent; border:0; border-radius:8px; color:var(--body);
}
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-chat-controls button:hover, .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-chat-controls button:focus-visible { background:var(--mint); color:var(--ink); outline:none; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-chat-controls button::before { content:''; width:16px; height:16px; background:currentColor; -webkit-mask-position:center; -webkit-mask-repeat:no-repeat; mask-position:center; mask-repeat:no-repeat; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-min::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 8h10'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 8h10'/%3E%3C/svg%3E"); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-max::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='10' height='10' rx='1.5'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='10' height='10' rx='1.5'/%3E%3C/svg%3E"); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-restore::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='5.5' width='7.5' height='7.5' rx='1.5'/%3E%3Cpath d='M6 5.5V4a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-1.5'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='5.5' width='7.5' height='7.5' rx='1.5'/%3E%3Cpath d='M6 5.5V4a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-1.5'/%3E%3C/svg%3E"); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-restore, html[data-ctai-chat="max"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-max { display:none !important; }
html[data-ctai-chat="max"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-restore { display:grid !important; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel [data-testid="stHorizontalBlock"] { gap:.5rem; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel .stButton > button { border-radius:999px; font-size:.9rem; padding:.4rem .9rem; border:1px solid var(--line); background:rgba(255,255,255,.03); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel .stButton > button:hover { border-color:var(--teal); }

/* new chat / history icons, placed beside the minimize and maximize controls */
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-chat_toolbar) { position:absolute; top:.8rem; right:calc(1.25rem + 62px); width:auto; z-index:3; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar { width:auto; flex-wrap:nowrap; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button {
  width:30px; height:30px; min-height:30px; padding:0; border:0; border-radius:8px;
  background:transparent; color:var(--body); display:grid; place-items:center; font-size:0;
}
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button:hover, .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button:focus-visible { background:var(--mint); color:var(--ink); }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button [data-testid="stMarkdownContainer"],
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button [data-testid="stIconMaterial"]:not(:first-child) { display:none; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button [data-testid="stIconMaterial"] { font-size:1.1rem; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar [data-testid="stPopover"] { width:auto; }
.stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_toolbar button > div > div:last-child:not(:first-child) { display:none; }

html[data-ctai-chat="max"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel { left:.75rem; right:.75rem; bottom:.75rem; width:auto; border-width:1px; border-radius:18px; }
html[data-ctai-chat="min"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel { display:none; }

/* divider with a small drag handle */
.ctai-divider {
  position:fixed; top:var(--ctai-top); bottom:0; right:calc(var(--ctai-chat-w) - 7px); width:14px;
  z-index:160; cursor:col-resize; touch-action:none; outline:none;
}
.ctai-divider::before { content:''; position:absolute; left:6px; top:0; bottom:0; width:2px; background:rgba(110,198,214,.7); transition:background .15s; }
.ctai-divider:hover::before, .ctai-divider:focus-visible::before, html.ctai-dragging .ctai-divider::before { background:#7FE0F0; }
.ctai-grip {
  position:absolute; left:0; top:50%; width:14px; height:32px; transform:translateY(-50%);
  background:#0E1218; border-radius:7px;
  background-image:radial-gradient(circle,#C9D6DF 1.3px,transparent 1.6px); background-size:7px 7px; background-position:center;
}
html[data-ctai-chat="max"] .ctai-divider, html[data-ctai-chat="min"] .ctai-divider { display:none; }

/* compact reopen button */
.ctai-reopen {
  display:none; position:fixed; right:1rem; bottom:1rem; z-index:300; cursor:pointer;
  align-items:center; gap:.5rem; padding:.6rem 1rem; border-radius:999px;
  background:#12303A; color:#D5EEF4; border:1px solid rgba(110,198,214,.5);
  font:600 .92rem 'DM Sans',sans-serif; box-shadow:0 8px 24px rgba(0,0,0,.45);
}
.ctai-reopen::before {
  content:''; width:18px; height:18px; background:currentColor;
  -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linejoin='round'%3E%3Cpath d='M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z'/%3E%3C/svg%3E") center/contain no-repeat;
  mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.5' stroke-linejoin='round'%3E%3Cpath d='M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z'/%3E%3C/svg%3E") center/contain no-repeat;
}
.ctai-reopen:hover { background:#17404D; }
html[data-ctai-chat="min"] .ctai-reopen { display:inline-flex; }

/* small screens: single column, assistant opens as a full-screen sheet */
@media (max-width: 900px) {
  .ctai-divider { display:none !important; }
  .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-chat_toolbar) { right:calc(1.25rem + 32px); }
  .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
  .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) { width:100%; }
  .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel, html[data-ctai-chat="max"] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .st-key-chat_panel {
    top:var(--ctai-vt); left:0; right:0; bottom:auto; width:auto; height:var(--ctai-vh);
    z-index:500; border-width:0; border-radius:0;
  }
  html .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-max, html[data-ctai-chat] .stApp:not(:has(.st-key-landing_shell)):has(.st-key-chat_panel) .ctai-ctl-restore { display:none !important; }
}
"""

LANDING_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp:has(.st-key-landing_shell) {
  --landing-bg: #090B0F;
  --landing-panel: rgba(18, 21, 27, .76);
  --landing-panel-soft: rgba(255, 255, 255, .025);
  --landing-text: #F1F3F6;
  --landing-muted: #9BA3AF;
  --landing-subtle: #747D89;
  --landing-line: rgba(205, 215, 230, .12);
  --landing-accent: #A9C9D6;
  --landing-accent-strong: #D5E7EB;
  --landing-glow: rgba(98, 145, 164, .12);
  background:
    radial-gradient(ellipse at 13% 22%, rgba(87, 111, 143, .14), transparent 38%),
    radial-gradient(ellipse at 75% 83%, rgba(64, 113, 126, .10), transparent 38%),
    var(--landing-bg) !important;
  color: var(--landing-text) !important;
  min-height: 100svh;
  overflow-x: hidden;
}
.stApp:has(.st-key-landing_shell) .block-container {
  box-sizing: border-box;
  max-width: 1440px;
  min-height: 100svh;
  padding: 1.2rem clamp(1.25rem, 5vw, 5.5rem) 1.4rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell {
  position: relative;
  isolation: isolate;
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: calc(100svh - 2.6rem);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell::before,
.stApp:has(.st-key-landing_shell) .st-key-landing_shell::after {
  content: "";
  position: fixed;
  z-index: -1;
  pointer-events: none;
  border: 1px solid rgba(191, 211, 226, .08);
  border-radius: 32px;
  transform: rotate(32deg);
  animation: landingDrift 24s ease-in-out infinite alternate;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell::before {
  width: 34vw;
  height: 34vw;
  min-width: 310px;
  min-height: 310px;
  right: -9vw;
  top: 12vh;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell::after {
  width: 21vw;
  height: 21vw;
  min-width: 210px;
  min-height: 210px;
  left: -11vw;
  bottom: -13vh;
  border-color: rgba(191, 211, 226, .055);
  animation-delay: -9s;
}
.stApp:has(.st-key-landing_shell),
.stApp:has(.st-key-landing_shell) p,
.stApp:has(.st-key-landing_shell) label,
.stApp:has(.st-key-landing_shell) input,
.stApp:has(.st-key-landing_shell) button {
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell p {
  color: var(--landing-muted);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav {
  position: relative;
  z-index: 2;
  margin: 0 auto clamp(2.2rem, 6vh, 4.4rem);
  width: min(100%, 920px);
  padding: .48rem .65rem .48rem 1.15rem;
  border: 1px solid var(--landing-line);
  border-radius: 999px;
  background: rgba(19, 22, 28, .78);
  backdrop-filter: blur(18px);
  box-shadow: 0 12px 50px rgba(0, 0, 0, .18);
}
.stApp:has(.st-key-landing_shell) .ctai-landing-brand {
  display: flex;
  align-items: center;
  gap: .7rem;
  color: var(--landing-text);
  white-space: nowrap;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-brand-mark {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(169, 201, 214, .32);
  border-radius: 10px;
  color: var(--landing-accent-strong);
  font-size: .87rem;
  font-weight: 700;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-brand-name {
  font-family: 'Manrope', sans-serif;
  font-size: .92rem;
  font-weight: 700;
  letter-spacing: -.025em;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav .stButton > button {
  min-height: 2.3rem;
  width: auto;
  padding: .35rem .55rem;
  border: 1px solid transparent;
  background: transparent;
  color: var(--landing-text);
  font-size: .92rem;
  font-weight: 700;
  box-shadow: none;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav .stButton > button:hover {
  border-color: var(--landing-line);
  background: var(--landing-panel-soft);
  color: var(--landing-text);
  transform: none;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav .stButton > button [data-testid="stIconMaterial"] {
  color: var(--landing-accent-strong);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav [data-testid="stHorizontalBlock"] {
  align-items: center;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav [data-testid="stPopoverButton"] {
  min-height: 2.3rem;
  border: 1px solid var(--landing-line);
  border-radius: 999px;
  background: rgba(255,255,255,.035);
  color: var(--landing-text);
  font-size: .84rem;
  font-weight: 600;
  box-shadow: none;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_nav [data-testid="stPopoverButton"]:hover {
  border-color: rgba(169,201,214,.34);
  background: rgba(255,255,255,.06);
  color: var(--landing-text);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_role_panel {
  position: relative;
  overflow: hidden;
  padding: clamp(1.4rem, 3vw, 2.1rem);
  border: 1px solid var(--landing-line);
  border-radius: 26px;
  background:
    radial-gradient(ellipse at 100% 0%, rgba(169,201,214,.07), transparent 46%),
    linear-gradient(145deg, rgba(255,255,255,.035), transparent 55%),
    var(--landing-panel);
  box-shadow: 0 30px 90px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.035);
  backdrop-filter: blur(20px);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_role_panel::before {
  content: "";
  position: absolute;
  inset: 0 18% auto;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(190,216,225,.45), transparent);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_hero {
  align-items: center;
  gap: clamp(2rem, 6vw, 6rem);
  animation: landingEnter .8s cubic-bezier(.2,.7,.2,1) both;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-copy {
  max-width: 660px;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-kicker {
  display: flex;
  align-items: center;
  gap: .65rem;
  color: var(--landing-accent);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .18em;
  text-transform: uppercase;
  margin-bottom: 1.3rem;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-kicker::before {
  content: "";
  width: 24px;
  height: 1px;
  background: var(--landing-accent);
}
.stApp:has(.st-key-landing_shell) .ctai-landing-title {
  max-width: 670px;
  margin: 0 0 1.2rem;
  color: var(--landing-text);
  font-family: 'Manrope', sans-serif;
  font-size: clamp(3.3rem, 6vw, 5.8rem);
  font-weight: 600;
  letter-spacing: -.075em;
  line-height: .99;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-title span {
  display: block;
  color: var(--landing-accent);
}
.stApp:has(.st-key-landing_shell) .ctai-landing-description {
  max-width: 530px;
  margin: 0;
  color: var(--landing-muted);
  font-size: clamp(1rem, 1.3vw, 1.15rem);
  line-height: 1.75;
}
.stApp:has(.st-key-landing_shell) .ctai-trust-list {
  display: grid;
  gap: .9rem;
  margin-top: clamp(1.7rem, 4vh, 2.9rem);
}
.stApp:has(.st-key-landing_shell) .ctai-trust-item {
  display: flex;
  align-items: center;
  gap: .8rem;
  color: #C3CAD3;
  font-size: .91rem;
}
.stApp:has(.st-key-landing_shell) .ctai-trust-dot {
  width: 7px;
  height: 7px;
  flex: 0 0 7px;
  border-radius: 50%;
  background: var(--landing-accent);
  box-shadow: 0 0 15px var(--landing-glow);
}
.stApp:has(.st-key-landing_shell) .ctai-research-art {
  width: min(100%, 490px);
  margin-top: clamp(1.5rem, 4vh, 3rem);
  opacity: .85;
}
.stApp:has(.st-key-landing_shell) .ctai-role-panel {
  position: relative;
  overflow: hidden;
  padding: clamp(1.35rem, 3vw, 2.15rem);
  border: 1px solid var(--landing-line);
  border-radius: 26px;
  background:
    linear-gradient(145deg, rgba(255,255,255,.045), transparent 48%),
    var(--landing-panel);
  box-shadow: 0 30px 90px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.035);
  backdrop-filter: blur(20px);
}
.stApp:has(.st-key-landing_shell) .ctai-role-panel::before {
  content: "";
  position: absolute;
  inset: 0 18% auto;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(190,216,225,.45), transparent);
}
.stApp:has(.st-key-landing_shell) .ctai-role-heading {
  margin: 0 0 .45rem;
  color: var(--landing-text);
  font-family: 'Manrope', sans-serif;
  font-size: 1.55rem;
  font-weight: 600;
  letter-spacing: -.045em;
}
.stApp:has(.st-key-landing_shell) .ctai-role-subtitle {
  margin: 0 0 1.45rem;
  color: var(--landing-muted);
  font-size: .9rem;
  line-height: 1.55;
}
.stApp:has(.st-key-landing_shell) .ctai-role-option {
  margin: .8rem 0;
  padding: 1rem 1.05rem;
  border: 1px solid var(--landing-line);
  border-radius: 17px;
  background: var(--landing-panel-soft);
  transition: border-color .2s ease, background .2s ease, transform .2s ease;
}
.stApp:has(.st-key-landing_shell) .ctai-role-option:hover {
  border-color: rgba(169,201,214,.34);
  background: rgba(255,255,255,.045);
  transform: translateY(-2px);
}
.stApp:has(.st-key-landing_shell) .ctai-role-option-title {
  margin: 0 0 .22rem;
  color: var(--landing-text);
  font-size: .98rem;
  font-weight: 650;
}
.stApp:has(.st-key-landing_shell) .ctai-role-option-copy {
  margin: 0;
  color: var(--landing-muted);
  font-size: .82rem;
  line-height: 1.5;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell .stTextInput input {
  min-height: 2.7rem;
  border: 1px solid var(--landing-line);
  border-radius: 12px;
  background: rgba(6,8,11,.35);
  color: var(--landing-text);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_shell .stTextInput label {
  color: var(--landing-subtle);
  font-size: .79rem;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_hero .stButton > button {
  min-height: 2.7rem;
  width: 100%;
  border: 1px solid rgba(169,201,214,.29);
  border-radius: 13px;
  background: linear-gradient(135deg, rgba(169,201,214,.16), rgba(169,201,214,.06));
  color: var(--landing-text);
  font-size: .87rem;
  font-weight: 650;
  box-shadow: none;
  transition: border-color .2s ease, background .2s ease, transform .2s ease;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_hero .stButton > button:hover {
  border-color: rgba(195,224,234,.55);
  background: linear-gradient(135deg, rgba(169,201,214,.23), rgba(169,201,214,.1));
  box-shadow: 0 8px 24px rgba(0,0,0,.16);
  color: var(--landing-text);
  transform: translateY(-1px);
}
.stApp:has(.st-key-landing_shell) .st-key-landing_role_panel [data-testid="stRadio"] {
  margin: .45rem 0 .3rem;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_role_panel [data-testid="stRadioOption"] {
  min-height: 2.8rem;
  padding: .55rem .75rem;
  border: 1px solid var(--landing-line);
  border-radius: 12px;
  background: rgba(255,255,255,.025);
  color: var(--landing-muted);
  font-size: .82rem;
  font-weight: 600;
}
.stApp:has(.st-key-landing_shell) .st-key-landing_role_panel [data-testid="stRadioOption"][data-selected="true"] {
  border-color: rgba(169,201,214,.4);
  background: rgba(169,201,214,.1);
  color: var(--landing-text);
}
.stApp:has(.st-key-landing_shell) .ctai-demo-notice {
  margin-top: 1rem;
  padding: .75rem .85rem;
  border: 1px solid var(--landing-line);
  border-radius: 12px;
  color: var(--landing-muted);
  font-size: .75rem;
  line-height: 1.5;
}
.stApp:has(.st-key-landing_shell) .ctai-demo-notice strong {
  color: var(--landing-accent-strong);
  font-weight: 650;
}
.stApp:has(.st-key-landing_shell) .ctai-landing-footer {
  margin-top: auto;
  padding-top: 1.5rem;
  color: var(--landing-subtle);
  font-size: .72rem;
  text-align: center;
}
@keyframes landingDrift {
  from { transform: rotate(32deg) translate3d(0,0,0); }
  to { transform: rotate(42deg) translate3d(1.5rem,-1rem,0); }
}
@keyframes landingEnter {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 900px) {
  .stApp:has(.st-key-landing_shell) .block-container {
    display: block;
    min-height: 100svh;
  }
  .stApp:has(.st-key-landing_shell) .st-key-landing_shell {
    min-height: 100svh;
    justify-content: flex-start;
  }
  .stApp:has(.st-key-landing_shell) .st-key-landing_nav { margin-bottom: 2rem; }
  .stApp:has(.st-key-landing_shell) .st-key-landing_hero { gap: 1.5rem; }
  .stApp:has(.st-key-landing_shell) .ctai-landing-title { font-size: clamp(2.8rem, 10vw, 4.1rem); }
  .stApp:has(.st-key-landing_shell) .ctai-research-art { max-width: 340px; }
}
@media (max-width: 640px) {
  .stApp:has(.st-key-landing_shell) .block-container { padding: .75rem 1rem 1.3rem; }
  .stApp:has(.st-key-landing_shell) .st-key-landing_nav {
    padding: .45rem .55rem;
    border-radius: 22px;
  }
  .stApp:has(.st-key-landing_shell) .st-key-landing_nav [data-testid="stHorizontalBlock"] {
    gap: .15rem;
  }
  .stApp:has(.st-key-landing_shell) .ctai-landing-brand-name { font-size: .78rem; }
  .stApp:has(.st-key-landing_shell) .st-key-landing_nav .stButton > button {
    padding: .3rem .45rem;
    font-size: .74rem;
  }
  .stApp:has(.st-key-landing_shell) .ctai-landing-title { font-size: clamp(2.7rem, 12vw, 3.7rem); }
  .stApp:has(.st-key-landing_shell) .ctai-research-art { margin-top: 1.2rem; }
  .stApp:has(.st-key-landing_shell) .ctai-role-panel { padding: 1.25rem; }
}
@media (prefers-reduced-motion: reduce) {
  .stApp:has(.st-key-landing_shell) *,
  .stApp:has(.st-key-landing_shell) *::before,
  .stApp:has(.st-key-landing_shell) *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}
"""



_RB = ".stApp:not(:has(.st-key-landing_shell)):has(.st-key-review_bar)"
REVIEW_BAR_CSS = f"""
/* ---------- coordinator review: sticky nav + compact status bar ---------- */
{_RB} .block-container {{ padding-top:.5rem; }}
{_RB} [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{ position:sticky; top:0; z-index:300; }}
{_RB} .st-key-topbar {{ background:#0E1218; margin-bottom:0; height:50px; }}
{_RB} [data-testid="stLayoutWrapper"]:has(> .st-key-review_bar) {{ position:sticky; top:54px; z-index:290; }}
{_RB} .st-key-review_bar {{
  position:relative; box-sizing:border-box; min-height:58px; max-width:100%; overflow:hidden;
  padding:.4rem .8rem .55rem; gap:0; background:#0E1218;
  border:1px solid var(--line); border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.28);
}}
{_RB} .st-key-review_bar [data-testid="stHorizontalBlock"] {{ gap:.6rem; flex-wrap:nowrap; align-items:center; }}
{_RB} .st-key-review_bar [data-testid="stColumn"] {{ min-width:0; }}
{_RB} .ctai-rbar-stats {{ display:flex; flex-wrap:wrap; gap:.25rem 1.1rem; align-items:baseline; font-size:.78rem; color:var(--body); }}
{_RB} .ctai-rstat b {{ margin-left:.35rem; font-size:.95rem; color:var(--ink); font-weight:750; }}
{_RB} .ctai-rstat.ok b {{ color:#59B3A6; }}
{_RB} .ctai-rstat.warn b {{ color:#E3B15C; }}
{_RB} .ctai-rstat.bad b {{ color:#E58A8A; }}
{_RB} .st-key-review_bar .stButton > button,
{_RB} .st-key-review_bar [data-testid="stPopoverButton"] {{ min-height:2.1rem; padding:.2rem .7rem; font-size:.82rem; }}
{_RB} .st-key-review_bar [data-testid="stElementContainer"]:has(.ctai-rbar-progress) {{
  position:absolute; left:0; right:0; bottom:0; height:3px; margin:0; }}
{_RB} .ctai-rbar-progress {{ height:3px; background:rgba(255,255,255,.08); }}
{_RB} .ctai-rbar-progress > div {{ height:100%; background:linear-gradient(90deg,#2A7D74,#59B3A6); transition:width .4s ease; }}
@media (max-width: 900px) {{
  {_RB} .st-key-review_bar [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; flex-direction:row !important; }}
  {_RB} .st-key-review_bar [data-testid="stColumn"]:first-child {{ flex:1 1 100% !important; width:100% !important; }}
  {_RB} .st-key-review_bar [data-testid="stColumn"]:not(:first-child) {{ flex:1 1 calc(50% - .6rem) !important; width:auto !important; }}
}}
"""


def inject_css() -> None:
    """Apply the Clinical Trial AI theme to the page."""
    st.markdown(
        f"<style>{CSS}{DARK_WORKSPACE_CSS}{REVIEW_BAR_CSS}</style>",
        unsafe_allow_html=True,
    )



def inject_landing_css() -> None:
    """Apply the premium theme only while the landing screen is rendered."""
    st.markdown(
        f"<style>{LANDING_CSS}</style>",
        unsafe_allow_html=True,
    )


def svg_img(svg: str, alt: str, style: str = "") -> str:
    """Embed an SVG drawing as an <img>. (st.html removes raw inline <svg>.)"""
    if "xmlns=" not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img alt="{html.escape(alt)}" style="{style}" src="data:image/svg+xml;base64,{data}">'

LOGO_SVG = (
    '<svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" aria-label="Clinical Trial AI logo">'
    '<rect width="44" height="44" rx="14" fill="#2A7D74"/>'
    '<rect x="11" y="9" width="22" height="27" rx="5" fill="#fff"/>'
    '<rect x="16" y="6" width="12" height="7" rx="3" fill="#F2C14E"/>'
    '<path d="M15.5 21 l3 3 l5.5 -6" fill="none" stroke="#2A7D74" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<rect x="15" y="28" width="14" height="2.6" rx="1.3" fill="#B3D9D0"/></svg>'
)


def _hero_svg() -> str:
    return (
        '<svg width="420" height="400" viewBox="0 0 420 400" xmlns="http://www.w3.org/2000/svg" aria-label="Illustration of a checked trial protocol">'
        '<circle cx="210" cy="200" r="180" fill="#D7ECE7"/>'
        '<rect x="96" y="40" width="210" height="290" rx="26" fill="#fff" stroke="#183A43" stroke-width="7"/>'
        '<rect x="148" y="22" width="106" height="44" rx="14" fill="#2A7D74"/><circle cx="201" cy="44" r="7" fill="#E9F4F1"/>'
        '<circle cx="132" cy="108" r="11" fill="#2A7D74"/><rect x="156" y="100" width="118" height="16" rx="8" fill="#CFE7E1"/>'
        '<circle cx="132" cy="156" r="11" fill="#2A7D74"/><rect x="156" y="148" width="96" height="16" rx="8" fill="#CFE7E1"/>'
        '<circle cx="132" cy="204" r="11" fill="#E27D60"/><rect x="156" y="196" width="112" height="16" rx="8" fill="#F6D2C6"/>'
        '<circle cx="132" cy="252" r="11" fill="#2A7D74"/><rect x="156" y="244" width="84" height="16" rx="8" fill="#CFE7E1"/>'
        '<circle cx="272" cy="262" r="62" fill="#fff" fill-opacity=".9" stroke="#183A43" stroke-width="12"/>'
        '<path d="M246 264 l20 20 l38 -42" fill="none" stroke="#2A7D74" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>'
        '<line x1="318" y1="310" x2="372" y2="364" stroke="#183A43" stroke-width="20" stroke-linecap="round"/>'
        '<path d="M262 40 h112 a22 22 0 0 1 22 22 v58 a22 22 0 0 1 -22 22 h-70 l-30 28 v-28 h-12 a22 22 0 0 1 -22 -22 v-58 a22 22 0 0 1 22 -22 z" fill="#F2C14E"/>'
        '<path d="M318 112 C 288 92 292 68 308 68 C 315 68 318 74 318 78 C 318 74 321 68 328 68 C 344 68 348 92 318 112 Z" fill="#E27D60"/>'
        '</svg>'
    )


def _coord_svg() -> str:
    return ('<svg width="40" height="40" viewBox="0 0 40 40" aria-label="Coordinator icon">'
            '<rect x="9" y="6" width="22" height="29" rx="5" fill="#fff" stroke="#183A43" stroke-width="3"/>'
            '<rect x="14" y="3" width="12" height="7" rx="3" fill="#2A7D74"/>'
            '<path d="M14 20 l4 4 l8 -9" fill="none" stroke="#2A7D74" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>'
            '<rect x="14" y="28" width="12" height="2.6" rx="1.3" fill="#9FCBC3"/></svg>')


def _patient_svg() -> str:
    return ('<svg width="40" height="40" viewBox="0 0 40 40" aria-label="Patient icon">'
            '<circle cx="20" cy="14" r="7" fill="#fff" stroke="#183A43" stroke-width="3"/>'
            '<path d="M6 36 C6 26 12 22 20 22 C28 22 34 26 34 36 Z" fill="#E27D60"/>'
            '<path d="M20 33 C 14 29 15 24.5 18 24.5 C 19.2 24.5 20 25.6 20 26.4 C 20 25.6 20.8 24.5 22 24.5 C 25 24.5 26 29 20 33 Z" fill="#fff"/></svg>')


def hero_art() -> str:
    return svg_img(_hero_svg(), "Illustration of a checked trial protocol", "width:420px;height:400px")


def icon_coordinator() -> str:
    return ('<div class="ctai-role-icon" style="background:#D7ECE7">'
            + svg_img(_coord_svg(), "Coordinator icon", "width:40px;height:40px") + '</div>')


def icon_patient() -> str:
    return ('<div class="ctai-role-icon" style="background:#FBE3DA">'
            + svg_img(_patient_svg(), "Patient icon", "width:40px;height:40px") + '</div>')


def feature_dot(color: str, glyph: str) -> str:
    svg = ('<svg width="46" height="46" viewBox="0 0 46 46"><circle cx="23" cy="23" r="23" fill="' + color + '"/>'
           '<g transform="translate(11 11)">' + glyph + '</g></svg>')
    return svg_img(svg, "", "width:46px;height:46px")


GLYPH_CHECK = '<path d="M5 12.5 l4.5 4.5 L19 7" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
GLYPH_LOCK = '<rect x="5" y="11" width="14" height="10" rx="3" fill="#fff"/><path d="M8 11 V8 a4 4 0 0 1 8 0 v3" fill="none" stroke="#fff" stroke-width="2.6"/>'
GLYPH_USERS = '<circle cx="9" cy="9" r="3.5" fill="#fff"/><circle cx="17" cy="10" r="2.8" fill="#fff" fill-opacity=".8"/><path d="M2 20 c0-4 3-6 7-6 s7 2 7 6 z" fill="#fff"/>'


def e(text: object) -> str:
    """HTML-escape a value before putting it inside our own markup."""
    return html.escape(str(text if text is not None else ""))


def chip(text: str, tone: str = "teal") -> str:
    return f'<span class="ctai-chip {tone}">{e(text)}</span>'


def brand_html(tagline: str = "Research prototype") -> str:
    return (f'<div class="ctai-brand">{svg_img(LOGO_SVG, "Clinical Trial AI logo", "width:44px;height:44px")}<div><b>Clinical Trial AI</b>'
            f'<small>{e(tagline)}</small></div></div>')


def page_head(title: str, subtitle: Optional[str] = None) -> None:
    sub = f"<p>{e(subtitle)}</p>" if subtitle else ""
    st.html(f'<div class="ctai-page-head"><h2>{e(title)}</h2>{sub}</div>')


def metrics_row(items: list[tuple[str, object, bool]]) -> None:
    """Row of big-number tiles: (label, value, highlight_as_warning)."""
    cells = "".join(
        f'<div class="ctai-metric{" warn" if warn else ""}"><span>{e(label)}</span><b>{e(value)}</b></div>'
        for label, value, warn in items
    )
    st.html(f'<div class="ctai-metrics">{cells}</div>')


def progress_bar(fraction: float) -> None:
    pct = max(0, min(100, int(round(fraction * 100))))
    st.html(f'<div class="ctai-progress"><div style="width:{pct}%"></div></div>')
