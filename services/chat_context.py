"""
Lightweight conversation understanding for Patient Chat follow-ups.

Uses recent turns only to resolve references ("this", "that", "how long?",
"explain"). Never treats chat history as clinical evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from models.patient_models import PatientAnswer
from services.source_verifier import normalize_text

_SOFT_PREFIX = r"(?:(?:ok|okay|sure|yes|yeah|yep|alright|please),?\s+)*"

_BARE_DURATION = re.compile(
    rf"^{_SOFT_PREFIX}(?:how long|how often|how many days?)(?:[.!?])?$",
    re.IGNORECASE,
)
_DURATION_FOLLOW_UP = re.compile(
    rf"^{_SOFT_PREFIX}(?:how long (?:do|does|will|did) "
    r"(?:they|them|these|those|it|that|this|the tests?|each(?: one| test)?) "
    r"(?:take|last|go)(?:[.!?])?)$",
    re.IGNORECASE,
)
_EXPLAIN_REQUEST = re.compile(
    rf"^{_SOFT_PREFIX}(?:can you |could you |will you )?"
    r"(?:explain(?: that| this| it| please)?|tell me(?: more)?|"
    r"what does (?:that|this|it) mean|please explain(?: that| this| it)?)"
    r"(?:[.!?])?$",
    re.IGNORECASE,
)
_AFFIRM = re.compile(
    rf"^{_SOFT_PREFIX}(?:yes|yep|yeah|yup|ok|okay|sure|please|go ahead|go on|"
    r"tell me|explain|explain please|yes please|sure please|okay please|"
    r"yes,? please|tell me more|do it|please do|please explain|"
    r"can you explain|could you explain)(?:[.!?])?$",
    re.IGNORECASE,
)
_FINISH_QUESTION = re.compile(
    r"\b(?:when does (?:this|it|that) (?:end|finish|stop)\b|"
    r"when (?:is|does) (?:this|it|that) (?:over|done|finished)\b|"
    r"how long (?:will|does|do|is) "
    r"(?:this|it|that|they|them|these|those|each(?: one| test)?)"
    r"(?: take| last)?\b)",
    re.IGNORECASE,
)

# Intents that count as a meaningful clinical topic for follow-ups.
_TOPIC_INTENTS = frozenset({
    "procedures",
    "visits",
    "safety",
    "eligibility_inclusion",
    "eligibility_exclusion",
    "study_overview",
})


@dataclass(frozen=True)
class ConversationState:
    """Resolved view of recent chat — for understanding only, never as facts."""

    prior_query: Optional[str] = None
    prior_intent: Optional[str] = None
    prior_answer: Optional[PatientAnswer] = None
    offered_topic: Optional[str] = None
    last_cited_query: Optional[str] = None
    last_cited_intent: Optional[str] = None
    last_cited_answer: Optional[PatientAnswer] = None
    topic: Optional[str] = None


def _offered_topic(answer: PatientAnswer) -> Optional[str]:
    if answer.intent != "conversational":
        return None
    text = answer.answer_text.casefold()
    if "tests involved" in text or "explain the tests" in text:
        return "procedures"
    if (
        "what happens in this clinical trial" in text
        or "visits, tests, and safety" in text
    ):
        return "study_overview"
    return None


def _topic_from_answer(answer: PatientAnswer) -> Optional[str]:
    offered = _offered_topic(answer)
    if offered:
        return offered
    if answer.intent in _TOPIC_INTENTS:
        return answer.intent
    return None


def resolve_conversation_state(
    history: Optional[Sequence[dict[str, object]]],
) -> ConversationState:
    """Walk recent turns; prefer the latest meaningful cited topic."""
    if not history:
        return ConversationState()

    prior_query: Optional[str] = None
    prior_intent: Optional[str] = None
    prior_answer: Optional[PatientAnswer] = None
    offered: Optional[str] = None
    cited_query: Optional[str] = None
    cited_intent: Optional[str] = None
    cited_answer: Optional[PatientAnswer] = None
    topic: Optional[str] = None

    for turn in reversed(list(history)):
        query = turn.get("query")
        answer = turn.get("answer")
        if not isinstance(query, str) or not isinstance(answer, PatientAnswer):
            continue
        if prior_answer is None:
            prior_query, prior_intent, prior_answer = query, answer.intent, answer
            offered = _offered_topic(answer)
        if topic is None:
            topic = _topic_from_answer(answer)
        if cited_answer is None and answer.citations and answer.intent in _TOPIC_INTENTS:
            cited_query, cited_intent, cited_answer = query, answer.intent, answer
            if topic is None:
                topic = answer.intent
        if offered is None:
            offered = _offered_topic(answer)
        if (
            prior_answer is not None
            and cited_answer is not None
            and topic is not None
        ):
            break

    if topic is None:
        topic = offered or cited_intent or prior_intent

    return ConversationState(
        prior_query=prior_query,
        prior_intent=prior_intent,
        prior_answer=prior_answer,
        offered_topic=offered,
        last_cited_query=cited_query,
        last_cited_intent=cited_intent,
        last_cited_answer=cited_answer,
        topic=topic,
    )


def is_affirmative_follow_up(query: str) -> bool:
    return bool(_AFFIRM.fullmatch(normalize_text(query)))


def is_explain_request(query: str) -> bool:
    return bool(_EXPLAIN_REQUEST.fullmatch(normalize_text(query)))


def is_bare_duration_follow_up(query: str) -> bool:
    text = normalize_text(query)
    return bool(_BARE_DURATION.fullmatch(text) or _DURATION_FOLLOW_UP.fullmatch(text))


def is_more_detail_request(query: str) -> bool:
    """True for 'tell me more' / 'okay, tell me more' style follow-ups."""
    text = normalize_text(query)
    return bool(
        re.fullmatch(
            rf"{_SOFT_PREFIX}(?:tell me more|more(?: detail| info| information)?|"
            r"what else|go on|elaborate)(?:[.!?])?",
            text,
            flags=re.IGNORECASE,
        )
    )


# Split combined patient messages into related questions / sub-intents.
_QUESTION_BOUNDARY = re.compile(
    r"(?<=[?!.])\s+"
    r"(?=(?:"
    r"what|when|how|where|who|why|which|"
    r"can|could|should|would|will|do|does|did|is|are|am|"
    r"and\s+(?:what|when|how|where|who|the|my)|"
    r"what\s+about|how\s+about|and\s+what\s+about"
    r")\b)",
    re.IGNORECASE,
)


def split_related_questions(query: str) -> list[str]:
    """
    Split one message into related questions when several are asked together.

    Returns a single-item list when there is only one question. Never treats
    split parts as clinical evidence — only as separate intents to answer.
    """
    text = re.sub(r"\s+", " ", (query or "").strip())
    if not text:
        return []
    parts = [p.strip(" \t-–—") for p in _QUESTION_BOUNDARY.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text]
    cleaned: list[str] = []
    for part in parts:
        if len(part) < 4:
            continue
        cleaned.append(part)
    return cleaned if len(cleaned) > 1 else [text]


def expand_retrieval_query(query: str, state: ConversationState) -> str:
    """
    Build a normalized retrieval query from the patient message + context.

    History only clarifies references; it is never copied as clinical evidence.
    """
    text = (query or "").strip()
    if not text:
        return text

    topic = state.topic or state.last_cited_intent or state.offered_topic

    if is_bare_duration_follow_up(text) or _FINISH_QUESTION.search(normalize_text(text)):
        if topic == "procedures":
            return "how long does this test take"
        if topic in {"visits", "study_overview"}:
            return "how long do the study visits take"
        if state.last_cited_query:
            return f"{text} {state.last_cited_query}".strip()

    if is_explain_request(text) or is_affirmative_follow_up(text) or is_more_detail_request(text):
        if state.offered_topic == "procedures" or topic == "procedures":
            return "what tests and procedures are in this study"
        if state.offered_topic == "study_overview" or topic == "study_overview":
            return "what happens in the study"
        if topic == "visits":
            return "what are the study visits"
        if state.last_cited_query:
            return f"please explain {state.last_cited_query}"

    return text
