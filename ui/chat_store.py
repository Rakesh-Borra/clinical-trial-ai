"""
In-session store for patient chat conversations.

``st.session_state.patient_history`` stays the *active* conversation (the list
the chat panel renders). Every conversation is also kept here, per protocol and
per published guide version, so New Chat never deletes anything and History can
reopen it. This lives in session state only: it does not survive a browser
refresh or a server restart.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Dict, List, Optional

import streamlit as st

CHATS_KEY = "patient_chats"
ACTIVE_KEY = "patient_active_chat_id"
_TITLE_LENGTH = 42
_counter = itertools.count(1)


def guide_signature(guide: Any) -> str:
    """Identify the exact approved guide a conversation was held against."""
    dump = guide.model_dump_json() if hasattr(guide, "model_dump_json") else repr(guide)
    return hashlib.sha1(dump.encode("utf-8")).hexdigest()


def _title(history: List[dict]) -> str:
    text = " ".join(str(history[0]["query"]).split()) if history else "New chat"
    return text if len(text) <= _TITLE_LENGTH else text[: _TITLE_LENGTH - 1].rstrip() + "…"


def _store() -> Dict[str, List[dict]]:
    return st.session_state.setdefault(CHATS_KEY, {})


def _continues(conversation: dict, history: List[dict]) -> bool:
    """True when history is the saved conversation, possibly with new turns."""
    saved = conversation["messages"]
    return len(history) >= len(saved) and all(a is b for a, b in zip(saved, history))


def sync_active(protocol_id: Optional[str], guide: Any) -> None:
    """Save the active conversation under its protocol."""
    history = st.session_state.get("patient_history") or []
    if not protocol_id or not history:
        return
    conversations = _store().setdefault(protocol_id, [])
    active_id = st.session_state.get(ACTIVE_KEY)
    conversation = next((c for c in conversations if c["id"] == active_id), None)
    if conversation is None or not _continues(conversation, history):
        conversation = {"id": f"chat-{next(_counter)}-{len(conversations)}"}
        conversations.append(conversation)
        st.session_state[ACTIVE_KEY] = conversation["id"]
    conversation.update(
        messages=list(history),
        title=_title(history),
        signature=guide_signature(guide),
        updated=next(_counter),
    )


def new_chat(protocol_id: Optional[str], guide: Any) -> None:
    """Keep the current conversation and start an empty one."""
    sync_active(protocol_id, guide)
    st.session_state.patient_history = []
    st.session_state[ACTIVE_KEY] = None
    st.session_state.pop("patient_chat_input", None)


def list_chats(protocol_id: Optional[str], guide: Any) -> List[dict]:
    """Saved conversations for this protocol only, newest first."""
    if not protocol_id:
        return []
    signature = guide_signature(guide)
    own = [c for c in _store().get(protocol_id, []) if c.get("signature") == signature]
    return sorted(own, key=lambda c: c["updated"], reverse=True)


def open_chat(protocol_id: Optional[str], guide: Any, chat_id: str) -> bool:
    """Make a saved conversation of this protocol the active one."""
    target = next((c for c in list_chats(protocol_id, guide) if c["id"] == chat_id), None)
    if target is None:
        return False
    sync_active(protocol_id, guide)
    st.session_state.patient_history = list(target["messages"])
    st.session_state[ACTIVE_KEY] = target["id"]
    st.session_state.pop("patient_chat_input", None)
    return True
