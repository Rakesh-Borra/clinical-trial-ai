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


def inject_css() -> None:
    """Apply the Clinical Trial AI theme to the page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)



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
