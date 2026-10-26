"""Run JavaScript against Streamlit's main application document.

Streamlit 1.50+ sanitizes ``st.html`` and no longer accepts
``unsafe_allow_javascript``. Executable scripts must be mounted with
``components.v1.html`` (iframe) and reach the parent document where the
app chrome and scroll surface live.
"""

from __future__ import annotations

import streamlit.components.v1 as components


def run_in_app_document(script: str) -> None:
    """Execute ``script`` with ``window`` / ``document`` bound to the app page."""
    wrapped = (
        "<script>\n"
        "(function (window, document) {\n"
        f"{script}\n"
        "})(window.parent, window.parent.document);\n"
        "</script>"
    )
    components.html(wrapped, height=0, scrolling=False)
