"""
Strict detection of purely social chat messages (greetings, thanks, goodbyes).

A message counts only when the WHOLE message is a social phrase. Anything with
extra content ("Hi, can I stop my medication?", "Thanks, but am I eligible?")
returns None and goes through the normal protocol and safety handling. No
model is called and no medical or protocol fact is produced here.
"""

from __future__ import annotations

import re
from typing import Optional

from models.patient_models import PatientAnswer

_GREETING = r"(?:hi|hello|hey|hiya|howdy|greetings)(?: there)?"
_TIME_OF_DAY = r"good (morning|afternoon|evening)"
_HOW_ARE_YOU = (
    r"how are you(?: doing| today)?|how (?:is|s) it going|how are things"
)
_THANKS = (
    r"thanks|thank you|thank u|thx|many thanks|thanks a lot|thanks so much"
    r"|thank you so much|thank you very much|thanks again"
)
_BYE = r"bye|bye bye|goodbye|good bye|see you|see you later|talk to you later"

_MAX_LENGTH = 40

_TIME_REPLY = {
    "morning": "Good morning! ☀️",
    "afternoon": "Good afternoon! 🌤️",
    "evening": "Good evening! 🌙",
}
_HOW_REPLY = "I'm doing well, thank you for asking! 😊 How can I help you with your study today?"


def _normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", " "))
    return re.sub(r"\s+", " ", text).strip()


def social_reply_text(query: str) -> Optional[str]:
    """Return a short friendly reply for a purely social message, else None."""
    if not query or len(query) > _MAX_LENGTH:
        return None
    text = _normalize(query)
    if not text:
        return None

    if re.fullmatch(_THANKS, text):
        return "You're welcome! Let me know if you have any other questions about your study."
    if re.fullmatch(_BYE, text):
        return "Goodbye! 👋 Take care, and come back any time you have questions about your study."

    time_match = re.fullmatch(rf"{_TIME_OF_DAY}(?: ({_HOW_ARE_YOU}))?", text)
    if time_match:
        if time_match.group(2):
            return f"{_TIME_REPLY[time_match.group(1)]} {_HOW_REPLY}"
        return (
            f"{_TIME_REPLY[time_match.group(1)]} I'm here to help you understand "
            "your clinical trial. What would you like to know?"
        )

    if re.fullmatch(_HOW_ARE_YOU, text):
        return _HOW_REPLY
    if re.fullmatch(rf"{_GREETING}(?: ({_HOW_ARE_YOU}))?", text):
        if re.search(_HOW_ARE_YOU, text):
            return f"Hi! 👋 {_HOW_REPLY}"
        return "Hi! 👋 Welcome to your study assistant. How can I help you today?"
    return None


def social_answer(query: str) -> Optional[PatientAnswer]:
    """Wrap a social reply as a PatientAnswer, or None for anything else."""
    reply = social_reply_text(query)
    if reply is None:
        return None
    return PatientAnswer(
        answer_text=reply,
        intent="conversational",
        confidence="not_applicable",
        audience="patient",
    )
