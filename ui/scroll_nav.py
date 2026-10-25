"""
Top-bar Back to Top / Go to Bottom controls.

Rendered with ``st.html`` into the same Streamlit top-bar document as Account.
Scroll JS follows ``ui/split_pane.py`` (``unsafe_allow_javascript=True``).

Streamlit's scrollable surface is ``main[data-testid="stMain"]`` (overflow:auto),
not ``window`` / ``document.scrollingElement`` / ``stAppViewContainer`` (those
are overflow:hidden and do not move). Earlier handlers targeted the wrong node
and then disabled both buttons when window height equaled scroll height.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

_COORDINATOR = "coordinator"
_PATIENT = "patient"
_VIEW_LIST = "list"
_VIEW_VERSIONS = "versions"
_VIEW_WORKFLOW = "workflow"

# Markup must stay in the main document (visible next to Account).
SCROLL_ACTIONS_HTML = """
<div class="ctai-scroll-actions" role="group" aria-label="Page scroll">
  <button type="button" class="ctai-scroll-btn" data-ctai-scroll="top"
          title="Back to top" aria-label="Back to top">↑</button>
  <button type="button" class="ctai-scroll-btn" data-ctai-scroll="bottom"
          title="Go to bottom" aria-label="Go to bottom">↓</button>
</div>
"""

# Prefer Streamlit's real scroller; never rely on window.scrollTo alone.
# Built without HTML markup inside the script body (Streamlit sanitizer).
_SCRIPT = r"""
(function () {
  function scrollDelta(el) {
    if (!el) return 0;
    return (el.scrollHeight || 0) - (el.clientHeight || 0);
  }

  function canScroll(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const oy = style.overflowY;
    const ox = style.overflow;
    const allows = oy === "auto" || oy === "scroll" || ox === "auto" || ox === "scroll";
    return allows && scrollDelta(el) > 8;
  }

  function preferredCandidates() {
    return [
      document.querySelector('[data-testid="stMain"]'),
      document.querySelector("main.stMain"),
      document.querySelector("main"),
      document.querySelector("section.main"),
      document.querySelector(".main"),
      document.querySelector('[data-testid="stAppViewContainer"]'),
      document.scrollingElement,
      document.documentElement,
      document.body,
    ].filter(Boolean);
  }

  function findScrollContainer() {
    let best = null;
    let bestDelta = 0;
    for (const el of preferredCandidates()) {
      const delta = scrollDelta(el);
      if (canScroll(el) && delta > bestDelta) {
        best = el;
        bestDelta = delta;
      }
    }
    if (best) return best;
    const all = document.querySelectorAll("main, section, div");
    for (const el of all) {
      const delta = scrollDelta(el);
      if (canScroll(el) && delta > bestDelta) {
        best = el;
        bestDelta = delta;
      }
    }
    return best || document.querySelector('[data-testid="stMain"]') || document.scrollingElement || document.documentElement;
  }

  function metrics() {
    const el = findScrollContainer();
    return {
      el: el,
      top: el.scrollTop || 0,
      view: el.clientHeight || 0,
      height: el.scrollHeight || 0,
      delta: scrollDelta(el),
    };
  }

  function smoothTo(target) {
    const el = findScrollContainer();
    if (!el) return;
    const top = Math.max(0, target);
    if (typeof el.scrollTo === "function") {
      try {
        el.scrollTo({ top: top, behavior: "smooth" });
        return;
      } catch (err) {
        /* fall through */
      }
    }
    el.scrollTop = top;
  }

  function scrollTop() {
    smoothTo(0);
  }

  function scrollBottom() {
    const { el, height } = metrics();
    smoothTo(Math.max(0, height - (el.clientHeight || 0)));
  }

  function updateDisabled() {
    const { top, view, height, delta } = metrics();
    const scrollable = delta > 8;
    const nearTop = !scrollable || top <= 24;
    const nearBottom = !scrollable || top + view >= height - 24;
    document.querySelectorAll(".ctai-scroll-btn[data-ctai-scroll='top']").forEach(function (btn) {
      btn.disabled = nearTop;
    });
    document.querySelectorAll(".ctai-scroll-btn[data-ctai-scroll='bottom']").forEach(function (btn) {
      btn.disabled = nearBottom;
    });
  }

  function onScrollBtnClick(event) {
    const btn = event.currentTarget;
    if (!btn || btn.disabled) return;
    event.preventDefault();
    event.stopPropagation();
    if (btn.getAttribute("data-ctai-scroll") === "top") {
      scrollTop();
    } else {
      scrollBottom();
    }
    setTimeout(updateDisabled, 350);
  }

  function attachContainerScrollListener() {
    const el = findScrollContainer();
    if (!el || el.dataset.ctaiScrollListen === "1") return;
    el.dataset.ctaiScrollListen = "1";
    el.addEventListener("scroll", updateDisabled, { passive: true });
  }

  function bindButtons() {
    document.querySelectorAll(".ctai-scroll-btn[data-ctai-scroll]").forEach(function (btn) {
      if (btn.dataset.ctaiScrollBound === "1") return;
      btn.dataset.ctaiScrollBound = "1";
      btn.addEventListener("click", onScrollBtnClick);
    });
    attachContainerScrollListener();
    updateDisabled();
  }

  window.__ctaiScrollNav = {
    findScrollContainer: findScrollContainer,
    metrics: metrics,
    scrollTop: scrollTop,
    scrollBottom: scrollBottom,
    bindButtons: bindButtons,
  };

  bindButtons();
  if (!window.__ctaiScrollNavListening) {
    window.__ctaiScrollNavListening = true;
    window.addEventListener("resize", updateDisabled, { passive: true });
    new MutationObserver(function () { bindButtons(); })
      .observe(document.body, { childList: true, subtree: true });
  }
})();
"""


def scroll_target_candidates() -> tuple[str, ...]:
    """CSS selectors tried, in order, when locating Streamlit's scroll surface."""
    return (
        '[data-testid="stMain"]',
        "main.stMain",
        "main",
        "section.main",
        ".main",
        '[data-testid="stAppViewContainer"]',
    )


def pick_scroll_target(
    candidates: list[dict],
) -> Optional[dict]:
    """
    Pure helper mirroring the JS chooser for tests.

    Each candidate dict: {id, overflow_y, scroll_height, client_height}.
    Returns the best scrollable candidate or None.
    """
    best = None
    best_delta = 0
    for item in candidates:
        oy = (item.get("overflow_y") or "").lower()
        allows = oy in {"auto", "scroll"}
        delta = int(item.get("scroll_height") or 0) - int(item.get("client_height") or 0)
        if allows and delta > 8 and delta > best_delta:
            best = item
            best_delta = delta
    return best


def should_show_scroll_nav(
    *,
    role: Optional[str],
    view: Optional[str] = None,
    show_profile: bool = False,
    on_landing: bool = False,
) -> bool:
    """Whether ↑/↓ belong in the top bar for this screen."""
    if on_landing or role is None:
        return False
    if show_profile:
        return True
    if role == _PATIENT:
        return True
    if role == _COORDINATOR:
        return view in {_VIEW_LIST, _VIEW_VERSIONS, _VIEW_WORKFLOW}
    return False


def render_topbar_scroll_buttons() -> None:
    """Mount ↑/↓ in the current top-bar column (main document, no iframe)."""
    st.html(SCROLL_ACTIONS_HTML)
    st.html(f"<script>{_SCRIPT}</script>", unsafe_allow_javascript=True)
