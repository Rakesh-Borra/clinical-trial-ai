"""
Fixed, resizable split pane for the patient portal.

The study assistant is an ordinary keyed Streamlit container (``chat_panel``),
so its widgets, callbacks and session state are untouched. ``ui/theme.py``
takes it out of the page flow with ``position: fixed``; this module adds the
small amount of browser behaviour CSS cannot provide: the draggable divider,
minimize / maximize / restore / reopen, and keeping the width and mode on
``<html>`` where Streamlit reruns cannot reset them. Nothing here talks to
Python, so resizing never triggers a rerun or an AI request.
"""

from __future__ import annotations

import streamlit as st

from ui.app_js import run_in_app_document

# Icons are drawn with CSS masks in ui/theme.py: the sanitizer strips inline SVG.
CONTROLS_HTML = """
<div class="ctai-chat-controls">
  <button type="button" data-ctai-action="min" class="ctai-ctl-min" aria-label="Minimize study assistant" title="Minimize"></button>
  <button type="button" data-ctai-action="max" class="ctai-ctl-max" aria-label="Maximize study assistant" title="Maximize"></button>
  <button type="button" data-ctai-action="max" class="ctai-ctl-restore" aria-label="Restore study assistant" title="Restore"></button>
</div>
"""

# Idempotent: Streamlit may mount this element again after navigation.
_SCRIPT = """
(() => {
  if (window.__ctaiSplit) { window.__ctaiSplit.sync(); return; }
  const root = document.documentElement;
  const MIN_CHAT = 300, MIN_GUIDE = 340, DEFAULT_FRAC = 0.34, KEY = 'ctai.chat.frac';
  const isMobile = () => window.matchMedia('(max-width: 900px)').matches;
  const clampW = w => Math.max(MIN_CHAT, Math.min(w, window.innerWidth - MIN_GUIDE));
  let frac = parseFloat(localStorage.getItem(KEY));
  if (!(frac > 0 && frac < 1)) frac = DEFAULT_FRAC;
  let wasMobile = isMobile();
  let mode = wasMobile ? 'min' : 'split';

  // Built with DOM calls: Streamlit's sanitizer drops script bodies that contain markup.
  const el = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };
  const divider = el('div', 'ctai-divider');
  divider.setAttribute('role', 'separator');
  divider.setAttribute('aria-orientation', 'vertical');
  divider.setAttribute('aria-label', 'Resize study assistant');
  divider.tabIndex = 0;
  divider.append(el('span', 'ctai-grip'));
  const reopen = el('button', 'ctai-reopen');
  reopen.type = 'button';
  reopen.dataset.ctaiAction = 'open';
  reopen.setAttribute('aria-label', 'Open study assistant');
  const label = el('span');
  label.textContent = 'Study assistant';
  reopen.append(label);
  document.body.append(divider, reopen);

  function apply() {
    root.dataset.ctaiChat = mode;
    root.style.setProperty('--ctai-chat-w', clampW(frac * window.innerWidth) + 'px');
    const bar = document.querySelector('.st-key-topbar');
    if (bar) root.style.setProperty('--ctai-top', (bar.offsetHeight + 40) + 'px');
    const vv = window.visualViewport;
    root.style.setProperty('--ctai-vh', (vv ? vv.height : window.innerHeight) + 'px');
    root.style.setProperty('--ctai-vt', (vv ? vv.offsetTop : 0) + 'px');
  }
  function setMode(next) { mode = next; apply(); }

  document.addEventListener('click', e => {
    const btn = e.target.closest('[data-ctai-action]');
    if (!btn) return;
    const action = btn.dataset.ctaiAction;
    if (action === 'min') setMode('min');
    else if (action === 'open') setMode(isMobile() ? 'max' : 'split');
    else if (action === 'max') setMode(mode === 'max' ? 'split' : 'max');
  });

  let dragging = false;
  const move = clientX => {
    frac = clampW(window.innerWidth - clientX) / window.innerWidth;
    apply();
  };
  divider.addEventListener('pointerdown', e => {
    dragging = true;
    divider.setPointerCapture(e.pointerId);
    root.classList.add('ctai-dragging');
    e.preventDefault();
  });
  divider.addEventListener('pointermove', e => { if (dragging) move(e.clientX); });
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    root.classList.remove('ctai-dragging');
    localStorage.setItem(KEY, String(frac));
  };
  divider.addEventListener('pointerup', stop);
  divider.addEventListener('pointercancel', stop);
  divider.addEventListener('dblclick', () => { frac = DEFAULT_FRAC; localStorage.setItem(KEY, String(frac)); apply(); });
  divider.addEventListener('keydown', e => {
    const step = e.shiftKey ? 80 : 24;
    if (e.key === 'ArrowLeft') move(window.innerWidth - clampW(frac * window.innerWidth) - step);
    else if (e.key === 'ArrowRight') move(window.innerWidth - clampW(frac * window.innerWidth) + step);
    else return;
    localStorage.setItem(KEY, String(frac));
    e.preventDefault();
  });

  window.addEventListener('resize', () => {
    const mobile = isMobile();
    if (mobile !== wasMobile) { wasMobile = mobile; mode = mobile ? 'min' : 'split'; }
    apply();
  });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', apply);
    window.visualViewport.addEventListener('scroll', apply);
  }

  // Keep the newest chat message in view after each Streamlit rerun.
  let lastCount = -1, raf = 0;
  const tick = () => {
    raf = 0;
    const box = document.querySelector('.st-key-chat_messages');
    if (!box) { lastCount = -1; return; }
    apply();
    const count = box.querySelectorAll('[data-testid="stChatMessage"]').length;
    if (lastCount !== -1 && count > lastCount) box.scrollTop = box.scrollHeight;
    lastCount = count;
  };
  new MutationObserver(() => { if (!raf) raf = requestAnimationFrame(tick); })
    .observe(document.body, { childList: true, subtree: true });

  window.__ctaiSplit = { sync: apply };
  apply();
})();
"""


def render_script() -> None:
    run_in_app_document(_SCRIPT)
