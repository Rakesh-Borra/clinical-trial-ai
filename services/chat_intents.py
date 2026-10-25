"""
Strict detection of purely social chat messages (greetings, thanks, goodbyes).

A message counts only when the WHOLE message is a social phrase. Anything with
extra content ("Hi, can I stop my medication?", "Thanks, but am I eligible?")
returns None and goes through the normal protocol and safety handling. No
model is called and no medical or protocol fact is produced here.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional

from models.patient_models import PatientAnswer

_GREETING = r"(?:hi|hello|hey|hiya|howdy|greetings)(?: there)?"
_TIME_OF_DAY = r"good (morning|afternoon|evening)"
_HOW_ARE_YOU = (
    r"how are you(?: doing| today)?|how (?:is|s) it going|how are things"
)
_THANKS = (
    r"thanks|thank you|thankyou|thanku|thank u|thx|many thanks|thanks a lot|thanks so much"
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


_SOCIAL_VOCAB = (
    "hello", "hiya", "howdy", "greetings", "thanks", "thank", "goodbye",
    "morning", "afternoon", "evening", "good", "there", "doing", "today",
    "later", "again", "much", "very", "many",
)
_ACK = (
    r"(?:ok|okay|k|alright|great|cool|nice|sure|perfect)"
    r"(?: (?:cool|great|thanks|thank you|thx|got it|good|nice))?"
    r"|got it|i see|understood|sounds good|makes sense|i understand"
)
_HELP = r"help|help me|what can you do|what can you help with|who are you"
_ACK_REPLY = "Great! Let me know if there's anything else you'd like to know about your study."
_HELP_REPLY = (
    "I can answer questions about your study using the information your study "
    "team has approved, such as eligibility, visits, tests, risks and what "
    "to expect. I can't give medical advice or decide whether you can join. "
    "What would you like to know?"
)


def _squeeze(word: str) -> str:
    """Collapse stretched letters: 'hellllo' -> 'hello', 'helllo' -> 'hello'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def _canonical_social(text: str) -> str:
    """Fix stretched or slightly misspelled words in a candidate social phrase."""
    fixed = []
    for word in text.split():
        word = _squeeze(word)
        collapsed = re.sub(r"(.)\1", r"\1", word)
        if collapsed in {"hi", "hey", "bye", "thx"}:
            word = collapsed
        elif len(word) >= 4 and word not in _SOCIAL_VOCAB:
            # Short words must match exactly, so "they" never becomes "hey".
            near = difflib.get_close_matches(word, _SOCIAL_VOCAB, 1, 0.85)
            if near:
                word = near[0]
        fixed.append(word)
    return " ".join(fixed)


def _match_social(text: str) -> Optional[str]:
    if re.fullmatch(_THANKS, text):
        return "You're welcome! Let me know if you have any other questions about your study."
    if re.fullmatch(_BYE, text):
        return "Goodbye! 👋 Take care, and come back any time you have questions about your study."
    if re.fullmatch(_ACK, text):
        return _ACK_REPLY
    if re.fullmatch(_HELP, text):
        return _HELP_REPLY

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


def social_reply_text(query: str) -> Optional[str]:
    """Return a short friendly reply for a purely social message, else None."""
    if not query or len(query) > _MAX_LENGTH:
        return None
    text = _normalize(query)
    if not text:
        return None
    return _match_social(text) or _match_social(_canonical_social(text))


_MISSPELLINGS = {
    "eligble": "eligible", "elegible": "eligible", "eligable": "eligible",
    "eligibile": "eligible", "elgible": "eligible", "eligibilty": "eligibility",
    "blud": "blood", "bloud": "blood", "bluud": "blood", "blod": "blood",
    "vist": "visit", "visite": "visit", "vists": "visits", "visists": "visits",
    "medcation": "medication", "medicin": "medicine", "meds": "medication",
    "sideeffects": "side effects", "sidefects": "side effects",
    "proceedure": "procedure", "proceedures": "procedures",
    "excersize": "exercise", "pregnent": "pregnant", "wiithdraw": "withdraw",
    "withdrawl": "withdrawal", "scedule": "schedule", "shedule": "schedule",
    "benifits": "benefits", "benifit": "benefit", "rsik": "risk",
    "rissk": "risk", "riks": "risks", "sefety": "safety", "saftey": "safety",
    "stuudy": "study", "stdy": "study", "tial": "trial", "triel": "trial",
}
_QUERY_VOCAB = (
    "eligible", "eligibility", "blood", "visit", "visits", "medication",
    "procedure", "procedures", "schedule", "benefits", "benefit", "risks",
    "risk", "safety", "study", "trial", "withdraw", "withdrawal", "consent",
    "participate", "participant", "exclusion", "inclusion", "criteria",
    "pregnant", "injection", "questionnaire", "appointment", "duration",
    "treatment", "placebo", "confidential", "compensation", "responsibilities",
)


def correct_query_spelling(query: str, extra_vocab: Iterable[str] = ()) -> str:
    """
    Fix obvious misspellings in a patient question. Known mistakes are mapped
    directly; other words of 5+ letters are corrected only to a close domain
    or guide word. Words that are already known are never changed.
    """
    vocab = {w for w in (*_QUERY_VOCAB, *extra_vocab) if len(w) >= 5}
    words = []
    for token in query.split():
        match = re.fullmatch(r"(\W*)(\w+)(\W*)", token)
        if not match:
            words.append(token)
            continue
        lead, core, trail = match.groups()
        low = _squeeze(core.casefold())
        if low in _MISSPELLINGS:
            core = _MISSPELLINGS[low]
        elif len(low) >= 5 and low not in vocab and low.isalpha():
            near = difflib.get_close_matches(low, vocab, 1, 0.86)
            if near:
                core = near[0]
        words.append(f"{lead}{core}{trail}")
    return " ".join(words)



# Arrival / check-in statements are not factual protocol questions.
# After _normalize(), "I'm" becomes "i m".
_ARRIVAL_STATEMENT = re.compile(
    r"^(?:i m |i am |im |i )?(?:just )?(?:came|coming|here|arrived|got here)"
    r"(?: (?:in|over))?(?: for| to)?"
    r"(?: (?:a|the|my|our))?"
    r" (?:test|tests|testing|study|trial|visit|appointment)s?"
    r"(?: today| now)?"
    r"$",
    re.IGNORECASE,
)


def situational_reply_text(query: str) -> Optional[str]:
    """
    Friendly clarification for arrival-style statements.

    Does not invent appointment reasons or clinical facts — only offers to
    explain approved study information if that is what the patient meant.
    """
    if not query or len(query) > 80:
        return None
    text = _normalize(query)
    if not text or not _ARRIVAL_STATEMENT.fullmatch(text):
        return None
    if re.search(r"\btests?\b|\btesting\b", text):
        return (
            "Welcome! Are you asking about the tests involved in this "
            "clinical trial? I can explain the tests described in the "
            "approved study information."
        )
    return (
        "Welcome! Are you asking about what happens in this clinical trial? "
        "I can share the visits, tests, and safety details from the approved "
        "study information."
    )


def social_answer(query: str) -> Optional[PatientAnswer]:
    """Wrap a social or situational reply as a PatientAnswer, or None."""
    reply = social_reply_text(query) or situational_reply_text(query)
    if reply is None:
        return None
    return PatientAnswer(
        answer_text=reply,
        intent="conversational",
        confidence="not_applicable",
        audience="patient",
    )


def display_unavailable_message(message: Optional[str]) -> str:
    """Patient-facing wording for empty sections (display only; data unchanged)."""
    text = (message or "").strip()
    if not text:
        return "Not specified in the provided protocol."
    if "did not include this information" in text.casefold():
        return "Not specified in the provided protocol."
    return text


_PHRASING = (
    (r"\bappointments?\b", lambda m: "visits" if m.group(0).endswith("s") else "visit"),
    (r"\bcome in\b|\bcome to the clinic\b|\bgo to the clinic\b", lambda m: "visit"),
    (r"\bquit\b|\bdrop out\b|\bopt out\b|\bleave the (?:study|trial)\b",
     lambda m: "withdraw from the study"),
    # Informal / misspelled patient language → guide vocabulary.
    (r"\btestings\b", lambda m: "tests"),
    (r"\btesting\b", lambda m: "tests"),
    (r"\bwhat kind of tests?\b", lambda m: "what tests"),
    (r"\bwhat types? of tests?\b", lambda m: "what tests"),
    (r"\bwhat kind of (?:tests|testing|procedures)\b", lambda m: "what tests"),
    (r"\bwhat happens? during (?:the |this )?(?:study|trial)\b",
     lambda m: "what happens in the study"),
    (r"\bwhat will happen during (?:the |this )?(?:study|trial)\b",
     lambda m: "what happens in the study"),
    (r"\bwhat happens? in (?:the |this )?(?:study|trial)\b",
     lambda m: "what happens in the study"),
    # Only rewrite when the patient already named the study/trial — bare
    # "when does this end?" stays context-sensitive for follow-up resolution.
    (r"\bwhen (?:is|does) (?:the )?(?:study|trial) (?:end|over|finish|done)\b",
     lambda m: "when does the study end"),
    (r"\bhow long (?:is|will) (?:the |this )?(?:study|trial)(?: last| take)?\b",
     lambda m: "how long do the study visits take"),
    (r"\bhow long does the study last\b",
     lambda m: "how long do the study visits take"),
    (r"\bwhat happens next\b", lambda m: "what visits and tests happen next"),
    # Stable form so spelling correction cannot turn "happens"→"happen".
    (r"\bwhat happens? after (?:that|this|it)\b",
     lambda m: "what happens after that in the visits"),
    (r"\bany days\b", lambda m: "how long will testing continue"),
    (r"\bhow many days does the tests? prolong\b",
     lambda m: "how long will testing continue"),
    (r"\bhow long will testing continue\b",
     lambda m: "how long will testing continue"),
)


def normalize_phrasing(query: str) -> str:
    """Map everyday wording onto the guide's vocabulary (appointment -> visit)."""
    for pattern, repl in _PHRASING:
        query = re.sub(pattern, repl, query, flags=re.IGNORECASE)
    return query


# Topics patients ask about that the guide has no section for. They are only
# answered if an approved item mentions the topic; otherwise the patient is
# pointed to the study team. Nothing is invented.
_OUT_OF_SCOPE = (
    ("withdrawing from the study", r"\b(withdraw|withdrawal|quit|drop out|opt out)\b",
     "withdraw"),
    ("costs, payment or insurance", r"\b(cost|costs|pay|paid|payment|compensation|reimburse\w*|insurance|free)\b",
     "cost"),
    ("how the procedures feel", r"\b(hurt|hurts|painful|pain|sore)\b", "hurt"),
    ("privacy of your information", r"\b(privacy|private|confidential\w*)\b", "privacy"),
    ("contacting the study team", r"\b(contact|phone|call|number|email|reach|emergency)\b",
     "contact"),
)


def out_of_scope_topic(query: str) -> Optional[tuple[str, str]]:
    """Return (topic label, keyword) for a known topic the guide lacks a section for."""
    text = _normalize(query)
    for label, pattern, keyword in _OUT_OF_SCOPE:
        match = re.search(pattern, text)
        if match:
            return label, match.group(1) if match.groups() else keyword
    return None


def out_of_scope_reply(label: str) -> str:
    reply = (
        f"I don't have approved information about {label} in this guide, so "
        "I don't want to guess. Your study team can help with that."
    )
    if label.startswith("contacting"):
        reply += (
            " If you feel unwell or have an urgent concern, please contact "
            "your study team or your local emergency services."
        )
    return reply
