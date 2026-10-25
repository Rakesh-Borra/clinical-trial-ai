"""
Patient Assistant that answers only from the approved Patient Guide.

Patients must never see protocol content that a coordinator has not
approved. This assistant therefore answers from the approved items of the
Plain-Language Patient Guide, never from the full protocol analysis. If a
topic has not been approved yet, it says the study team is still reviewing
it.

Deterministic: no LLM calls. It reuses the Patient Assistant's intent
rules, then adds natural-language patterns and topic matching so questions
like "Do I need Type 2 diabetes to join?" can find the relevant approved
statement. Eligibility-decision and medical-advice questions are refused.
"""

from __future__ import annotations

import difflib
import re
from typing import List, Optional, Sequence, Tuple

from agents.patient_agent import classify_patient_intent
from models.patient_models import PatientAnswer, PatientCitation
from models.plain_language_models import GuideItem, GuideSection, PatientGuide
from services.chat_context import (
    ConversationState,
    expand_retrieval_query,
    is_affirmative_follow_up,
    is_bare_duration_follow_up,
    is_explain_request,
    is_more_detail_request,
    resolve_conversation_state,
    split_related_questions,
)
from services.chat_grounding import naturalize, validate_grounding
from services.chat_intents import (
    correct_query_spelling,
    normalize_phrasing,
    out_of_scope_reply,
    out_of_scope_topic,
    social_answer,
)
from services.guide_retrieval import retrieve_approved_passages
from services.source_verifier import normalize_text

APPROVED_STATUS = "coordinator_approved"

_DISCLAIMERS = [
    (
        "This assistant only shares information the study team has approved "
        "for patients. It is general information, not medical advice."
    ),
    (
        "It cannot decide whether you are eligible for a study. Talk with "
        "the study team or a qualified clinician about your personal situation."
    ),
]

_ASK_ABOUT = (
    "You can ask me about who may be able to join, who may not, the study "
    "visits, the tests and procedures, or safety."
)

# intent -> (guide section attribute, topic in everyday words, friendly opener)
_SECTION_INTENTS = {
    "eligibility_inclusion": (
        "who_can_join",
        "who may be able to join",
        "Here is who may be able to join, as approved by the study team:",
    ),
    "eligibility_exclusion": (
        "who_cannot_join",
        "who may not be able to join",
        "Here are reasons someone may not be able to take part, as approved "
        "by the study team:",
    ),
    "visits": (
        "visits",
        "the study visits",
        "Here is what the study team has shared about the visits:",
    ),
    "procedures": (
        "procedures",
        "the tests and procedures",
        "Here are the tests and procedures the study team has described:",
    ),
    "safety": (
        "safety",
        "safety",
        "Here is what the study team has shared about safety:",
    ),
}

# Extra natural-language cues used only when the shared classifier returns
# "unsupported". Keep these join/visit/procedure-scoped so bare clinical
# words alone do not force a section intent.
_NATURAL_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "eligibility_inclusion": (
        r"\b(do i|does (a |the )?person|does someone|do you) need\b.{0,60}\b"
        r"(join|enroll|participate|take part|study|trial)\b",
        r"\b(is|are) .{0,40}\brequired\b.{0,40}\b"
        r"(to )?(join|enroll|participate|take part)\b",
        r"\b(age|diabetes|hba1c)\s+requirement\b",
        r"\brequirements? to (join|enroll|participate|take part)\b",
        r"\bmust (i|someone|participants?|people)\b.{0,40}\b"
        r"(join|enroll|participate|have|be)\b",
        r"\bwho (can|may|is able to) (join|enroll|participate|take part)\b",
        r"\binclusion\b",
        r"\bqualif(?:y|ies|ication|ications)\b.{0,40}\b"
        r"(join|enroll|study|trial)\b",
    ),
    "eligibility_exclusion": (
        r"\bwho (cannot|can't|may not|is not able to) (join|enroll|participate)\b",
        r"\b(exclusion|excluded|disqualif)",
        r"\bnot (allowed|able|permitted) to (join|enroll|participate)\b",
        r"\breasons? (someone |i )?(cannot|can't|may not)\b",
    ),
    "visits": (
        r"\b(visit|visits|appointment|appointments)\b",
        r"\bstudy schedule\b",
        r"\bhow (often|many times) .{0,20}\b(come|go|visit|return)\b",
        r"\b(screening|follow[- ]?up) visit\b",
        r"\bwhen does (?:the |this )?(?:study|trial|it|this) end\b",
        r"\bhow long (?:do|does|will|is) .{0,40}\b(study|trial|visit|visits|take|last)\b",
        r"\b(study|trial) (?:timeline|duration|length)\b",
        r"\b(last|final) visit\b",
    ),
    "procedures": (
        r"\b(procedure|procedures)\b",
        r"\b(blood (work|draw|draws|glucose|sugar|pressure|tests?|testing|samples?)"
        r"|bloodwork|lab tests?|labs?)\b",
        r"\b(draw|drawing|drawn|collect|collecting|collected|take|taking|taken)"
        r"\b.{0,25}\b(blood|sample)\b",
        r"\b(blood|sample)\b.{0,25}\b(drawn|collected|taken)\b",
        r"\b(bp|a blood test)\b",
        r"\bwhat (kind|type)s? of (tests?|testing|testings|procedures?)\b",
        r"\bwhat (tests|testing|testings|procedures) (are|will|do|have)\b",
        r"\b(tests?|testing|testings) do we have\b",
        r"\bstudy (tests?|testing|testings)\b",
    ),
    "safety": (
        r"\b(side effects?|adverse events?)\b",
        r"\b(monitored|monitoring) for\b",
        r"\bis it safe\b",
        r"\bsafety (information|risks?|monitoring)\b",
        r"\bwhat are the risks\b",
    ),
    "study_overview": (
        r"\bwhat (is|study) (this|the) (study|trial)\b",
        r"\b(study|trial) (name|title|called)\b",
        r"\bprotocol id\b",
        r"\bwhat happens? (?:in|during) (?:the |this )?(?:study|trial)\b",
        r"\bwhat will happen (?:in|during) (?:the |this )?(?:study|trial)\b",
        r"\btell me about (?:the |this )?(?:study|trial)\b",
        r"\bwhat (?:is involved|do i do) (?:in|during) (?:the |this )?(?:study|trial)\b",
    ),
}

# Phrases that suggest a specific requirement rather than the whole section.
_SPECIFIC_TOPIC_CUES = (
    r"\b(do i|does someone|does a person) need\b",
    r"\b(is|are) .{0,40}\brequired\b",
    r"\bmust (i|someone)\b",
    r"\bage\b",
    r"\bage range\b",
    r"\byears? old\b",
    r"\bdiabetes\b",
    r"\bhba1c\b",
    r"\begfr\b",
    r"\btype 2\b",
    r"\bblood (glucose|pressure)\b",
    r"\bside effects?\b",
    r"\badverse events?\b",
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "is",
        "are",
        "was",
        "be",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "will",
        "i",
        "me",
        "my",
        "you",
        "your",
        "we",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "about",
        "with",
        "from",
        "into",
        "have",
        "has",
        "had",
        "need",
        "needs",
        "needed",
        "join",
        "joining",
        "study",
        "trial",
        "please",
        "tell",
        "what",
        "who",
        "when",
        "where",
        "which",
        "how",
        "why",
        "any",
        "some",
        "someone",
        "person",
        "people",
        "take",
        "part",
        "able",
        "may",
        "might",
        "also",
        "just",
        "only",
        "still",
        "more",
        "than",
        "then",
        "there",
        "here",
        "know",
        "ask",
        "question",
        "information",
        "detail",
        "details",
        "general",
        "must",
        "required",
        "requirement",
        "requirements",
        "criteria",
        "criterion",
        "eligible",
        "eligibility",
        "enroll",
        "enrolling",
        "participate",
        "participating",
        "participation",
        "allowed",
        "permission",
    }
)

# Shared phrase list for both query↔item scoring (longer phrases first).
_TOPIC_PHRASES = (
    "type 2 diabetes",
    "type 2",
    "blood glucose",
    "blood pressure",
    "adverse events",
    "adverse event",
    "side effects",
    "side effect",
    "follow-up visit",
    "follow up visit",
    "screening visit",
    "years old",
    "year old",
    "age range",
)

# Topic families: query must mention a cue; item must match related wording.
# Age deliberately excludes bare "years" and bare "between N and M" so BMI,
# dose, and disease-duration ranges are not treated as age requirements.
_TOPIC_FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "age": (
        ("age", "age range", "years old", "year old"),
        (),  # item match uses _AGE_ITEM_PATTERNS only
    ),
    "diabetes": (
        ("diabetes", "diabetic", "type 2", "t2d"),
        ("diabetes", "diabetic", "type 2", "t2d"),
    ),
    "hba1c": (
        ("hba1c", "a1c"),
        ("hba1c", "a1c"),
    ),
    "egfr": (
        ("egfr",),
        ("egfr",),
    ),
}

# Guide-item age cues: require explicit age language, never a bare numeric range.
_AGE_ITEM_PATTERNS = (
    r"\bage range\b",
    r"\bage\b",
    r"\byears? old\b",
    r"\b\d{1,3}\s*(?:to|-)\s*\d{1,3}\s+years?\s*old\b",
)


def _answer(
    text: str,
    intent: str,
    *,
    citations: Optional[List[PatientCitation]] = None,
    confidence: str = "unavailable",
    limitations: Optional[List[str]] = None,
) -> PatientAnswer:
    return PatientAnswer(
        answer_text=text,
        intent=intent,
        citations=citations or [],
        confidence=confidence,
        limitations=limitations or [],
        disclaimers=list(_DISCLAIMERS),
        audience="patient",
    )


def _approved(section: GuideSection) -> List[GuideItem]:
    return [item for item in section.items if item.approved and not item.rejected]


def _line(item: GuideItem) -> str:
    page = f" (protocol page {item.source_page})" if item.source_page else ""
    return f"- {item.plain_text}{page}"


def _citation(item: GuideItem) -> PatientCitation:
    return PatientCitation(
        category=item.section,
        statement=item.plain_text,
        source_page=item.source_page,
        verification_status=APPROVED_STATUS,
        plain_language=item.plain_text,
    )


def _pending_note(count: int) -> str:
    if count == 1:
        return (
            "One more item on this topic is still being reviewed by the "
            "study team, so it is not shown yet."
        )
    return (
        f"{count} more items on this topic are still being reviewed by the "
        "study team, so they are not shown yet."
    )


def _under_review(topic: str) -> str:
    return (
        f"The study team is still reviewing the information about {topic}, "
        "so I can't share it yet. Please check back later or ask the study "
        "team directly."
    )


def _not_published() -> PatientAnswer:
    return _answer(
        "The study team hasn't approved a patient guide for this study yet, "
        "so I can't answer questions about it yet. Please check back later "
        "or ask the study team directly.",
        "not_yet_approved",
        limitations=["No approved patient guide content is available yet."],
    )


def _significant_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", normalize_text(text))
    return {
        token
        for token in tokens
        if token not in _STOPWORDS and (len(token) > 2 or token.isdigit())
    }


def _item_blob(item: GuideItem) -> str:
    return normalize_text(f"{item.original_text} {item.plain_text}")


def _query_mentions(query_norm: str, cues: tuple[str, ...]) -> bool:
    return any(cue in query_norm for cue in cues)


def _topic_family_hits(query_norm: str, item_norm: str) -> int:
    """Boost when a question topic family aligns with item wording."""
    score = 0
    for family, (query_cues, item_cues) in _TOPIC_FAMILIES.items():
        if not _query_mentions(query_norm, query_cues):
            continue
        if family == "age":
            if any(re.search(pattern, item_norm) for pattern in _AGE_ITEM_PATTERNS):
                score += 4
            continue
        if any(cue in item_norm for cue in item_cues):
            score += 3
    return score


def _item_relevance(query: str, item: GuideItem) -> int:
    """Score how specifically a guide item matches the patient's question."""
    query_norm = normalize_text(query)
    item_norm = _item_blob(item)
    score = 0

    for phrase in _TOPIC_PHRASES:
        if phrase in query_norm and phrase in item_norm:
            score += 4

    score += _topic_family_hits(query_norm, item_norm)

    overlap = _significant_tokens(query) & _significant_tokens(
        f"{item.original_text} {item.plain_text}"
    )
    score += len(overlap)
    return score


def _looks_specific(query: str) -> bool:
    normalized = normalize_text(query)
    return any(re.search(pattern, normalized) for pattern in _SPECIFIC_TOPIC_CUES)


def _classify_intent(query: str) -> str:
    """Shared classifier first, then natural-language fallbacks."""
    intent = classify_patient_intent(query)
    if intent != "unsupported":
        return intent

    normalized = normalize_text(query)
    scored: list[tuple[int, str]] = []
    for candidate, patterns in _NATURAL_INTENT_PATTERNS.items():
        hits = sum(1 for pattern in patterns if re.search(pattern, normalized))
        if hits:
            scored.append((hits, candidate))

    if not scored:
        return "unsupported"

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _split_by_relevance(
    query: str,
    items: Sequence[GuideItem],
    *,
    min_score: int = 1,
) -> Tuple[List[GuideItem], List[GuideItem], bool]:
    """
    Partition items into relevant approved / relevant unapproved.

    Returns (approved_matches, unapproved_matches, used_topic_filter).
    - Broad question: filtered=False → caller may list the full approved section.
    - Specific question with no item match: filtered=True and empty lists →
      caller returns a safe no-information reply (never an unrelated dump).
    """
    scored = [(_item_relevance(query, item), item) for item in items]
    matches = [(score, item) for score, item in scored if score >= min_score]
    specific = _looks_specific(query)

    if not matches:
        # Specific with no overlap must not fall through to the full section.
        return [], [], specific

    best = max(score for score, _ in matches)
    # Keep the strongest matches; drop weak incidental overlaps.
    threshold = max(min_score, best - 1)
    strong = [item for score, item in matches if score >= threshold]
    approved = [item for item in strong if item.approved and not item.rejected]
    unapproved = [item for item in strong if not item.approved or item.rejected]
    return approved, unapproved, True


def _specific_answer(
    *,
    intent: str,
    topic: str,
    approved_matches: List[GuideItem],
    unapproved_matches: List[GuideItem],
) -> PatientAnswer:
    if approved_matches:
        opener = (
            "Here is what the study team has approved that relates to "
            "your question:"
        )
        lines = [opener, ""] + [_line(item) for item in approved_matches]
        limitations: List[str] = []
        if unapproved_matches:
            lines += [
                "",
                "Some related details are still being reviewed by the study "
                "team, so they are not shown yet.",
            ]
            limitations.append(
                f"{len(unapproved_matches)} related item(s) awaiting approval."
            )
        if intent == "eligibility_inclusion":
            lines += [
                "",
                "This is general study information only. Only the study team "
                "can confirm whether it applies to you.",
            ]
        return _answer(
            "\n".join(lines),
            intent,
            citations=[_citation(item) for item in approved_matches],
            confidence="high" if not unapproved_matches else "partial",
            limitations=limitations,
        )

    # Relevant content exists but is not approved — do not reveal it.
    return _answer(
        _under_review(topic),
        intent,
        limitations=[
            f"{len(unapproved_matches)} related item(s) about {topic} "
            "are awaiting approval."
        ],
    )


def _section_answer(guide: PatientGuide, intent: str, query: str) -> PatientAnswer:
    attr, topic, opener = _SECTION_INTENTS[intent]
    section: GuideSection = getattr(guide, attr)
    approved = _approved(section)
    pending = len(section.items) - len(approved)

    if not section.items:
        return _answer(
            f"The patient guide doesn't include details about {topic}. "
            "Please ask the study team about this.",
            intent,
            limitations=[f"The guide has no items about {topic}."],
        )

    approved_matches, unapproved_matches, filtered = _split_by_relevance(
        query,
        section.items,
    )
    if filtered:
        if approved_matches or unapproved_matches:
            return _specific_answer(
                intent=intent,
                topic=topic,
                approved_matches=approved_matches,
                unapproved_matches=unapproved_matches,
            )
        # Specific cue but no item overlap (e.g. asking about something
        # not in this section's guide text).
        return _answer(
            f"I don't have approved information that answers that specific "
            f"question about {topic}. {_ASK_ABOUT}",
            intent,
            limitations=[
                "No approved guide item matched the specific topic asked.",
            ],
        )

    if not approved:
        return _answer(
            _under_review(topic),
            intent,
            limitations=[f"{pending} item(s) about {topic} are awaiting approval."],
        )

    citations = [_citation(item) for item in approved]
    # Narrow procedure lists read better as one short sentence.
    if intent == "procedures" and len(approved) <= 3:
        text = (
            f"The approved guide lists these tests and procedures: "
            f"{_join_plain(approved)}."
            f"{_source_footer(citations)}"
        )
        limitations: List[str] = []
        if pending:
            text += f"\n\n{_pending_note(pending)}"
            limitations.append(
                f"{pending} item(s) about {topic} are awaiting approval."
            )
        return _answer(
            text,
            intent,
            citations=citations,
            confidence="high" if not pending else "partial",
            limitations=limitations,
        )

    lines = [opener, ""] + [_line(item) for item in approved]
    limitations = []
    if pending:
        lines += ["", _pending_note(pending)]
        limitations.append(f"{pending} item(s) about {topic} are awaiting approval.")
    if intent == "eligibility_inclusion":
        lines += [
            "",
            "Meeting these does not decide anything on its own. Only the "
            "study team can confirm who can join.",
        ]
    return _answer(
        "\n".join(lines),
        intent,
        citations=citations,
        confidence="high" if not pending else "partial",
        limitations=limitations,
    )


# --- Short / natural question lookup ---------------------------------------
# Used only when the classifier found no confident section (or the section had
# no matching approved item). It searches approved, non-rejected statements.

_SYNONYMS = {
    "bloodwork": ("blood", "test"),
    "labs": ("lab", "test"),
    "lab": ("lab", "test"),
    "sugar": ("glucose",),
    "bp": ("blood", "pressure"),
    "testing": ("test",),
    "testings": ("test",),
    "tested": ("test",),
    "tests": ("test",),
    "exam": ("test",),
    "checkup": ("visit",),
    "happen": ("happen", "visit", "test"),
    "happens": ("happen", "visit", "test"),
    "during": ("during", "visit", "test"),
    "timeline": ("visit", "day", "timing"),
    "duration": ("visit", "day", "timing"),
    "end": ("visit", "day", "follow"),
    "ends": ("visit", "day", "follow"),
}
# Words too generic to identify a topic on their own.
_LOOKUP_STOP = frozenset({"will", "get", "give", "taken", "take", "done", "kind", "type", "types", "have"})
_GENERIC_TOPIC = frozenset({"test", "lab", "testing", "testings", "procedure", "procedures"})
_BROAD_PROCEDURE_QUESTION = re.compile(
    r"\bwhat (kind|type)s? of (tests?|testing|testings|procedures?)\b"
    r"|\bwhat (tests?|testing|testings|procedures?) (will|do|are|have)\b"
    r"|\b(tests?|testing|testings) do we have\b"
    r"|\bwhat tests?\b",
    re.IGNORECASE,
)
_TIMELINE_QUESTION = re.compile(
    r"\bwhen does (?:the |this )?(?:study|trial|it|this) end\b"
    r"|\bwhen (?:is|does) (?:the )?(?:study|trial) (?:end|over|finish|done)\b"
    r"|\bhow long (?:do|does|will|is) .{0,40}\b(?:study|trial)\b"
    r"|\bhow many days (?:do|does|will|is) .{0,40}\b(?:study|trial)\b"
    r"|\bhow (?:long|many days) (?:do|does|will) the (?:study|trial) (?:last|take|run|continue)\b"
    r"|\b(?:study|trial) (?:timeline|duration|length)\b"
    r"|\b(?:last|final) visit\b",
    re.IGNORECASE,
)
_TEST_DURATION_QUESTION = re.compile(
    r"\bwhen does (?:this|the|that) tests? end\b"
    r"|\bhow long does (?:this|the|that|a|my) tests? (?:take|last|end)\b"
    r"|\bhow long (?:is|will) (?:this|the|that) tests?\b"
    r"|\bhow long (?:do|does|will|did) "
    r"(?:they|them|these|those|each(?: one| test)?) (?:take|last)\b"
    r"|\bhow long does .{0,60}\b(?:testing|tests?|procedure|glucose|pressure|blood)\b"
    r".{0,20}\b(?:take|last)\b",
    re.IGNORECASE,
)
_FIRST_OR_NEXT_VISIT = re.compile(
    r"\bwhen (?:is|are|will) (?:my |the |our )?(?:first|next|initial|upcoming) visit\b"
    r"|\b(?:first|next|initial|screening) visit\b.{0,20}\b(?:when|date|day|time)\b"
    r"|\bwhen .{0,20}\b(?:first|next|initial|screening) visit\b"
    r"|\b(?:what about|how about) (?:the |my |our )?"
    r"(?:first|next|initial|upcoming|follow[- ]?up|screening) visit\b"
    r"|\b(?:and |also )?(?:the |my )?(?:next|follow[- ]?up) visit\b",
    re.IGNORECASE,
)
_EXPLICIT_STUDY_END = re.compile(
    r"\b(?:study|trial)\b.{0,40}\b(?:end|ends|ended|complete|completed|completion|until)\b"
    r"|\b(?:end|ends|ended|complete|completed|completion)\b.{0,40}\b(?:study|trial)\b"
    r"|\bfinal visit\b|\blast visit\b|\bduration\b|\buntil day\b",
    re.IGNORECASE,
)
_TIMING_QUESTION = re.compile(
    r"\b(when|what time|how often|how long|how many times|schedule|end|ends|timeline|duration)\b"
)
_TIMING_TEXT = re.compile(
    r"\b(day|days|week|weeks|month|months|hour|hours|minutes?|daily|weekly|"
    r"monthly|every|baseline|screening|before|after|timing|at each|follow[- ]?up)\b"
)
_MISSING_DETAIL = (
    "Not specified in the provided protocol. "
    "Please check with your study coordinator if you need that detail."
)
_NOT_SPECIFIED = "Not specified in the provided protocol."
_BLOOD_COLLECTION_QUESTION = re.compile(
    r"\b(?:draw|drawing|drawn|collect|collecting|collected|take|taking|taken|"
    r"obtain|obtaining|give|giving|provide|providing)\b.{0,30}\b(?:blood|sample)\b"
    r"|\b(?:blood|sample)\b.{0,30}\b(?:drawn|collected|taken|obtained)\b"
    r"|\b(?:blood draw|blood collection|blood sample)\b"
    r"|\b(?:test|check|analy[sz]e)\s+(?:my|your|our|their)\s+blood\b",
    re.IGNORECASE,
)
_BLOOD_COLLECTION_FOLLOW_UP = re.compile(
    r"\b(?:will|would|can|could|do|does)\s+they\s+"
    r"(?:draw|collect|take|obtain)\s+(?:it|that|one)\b"
    r"|\b(?:draw|collect|take|obtain)\s+(?:it|that|one)\b",
    re.IGNORECASE,
)
_BLOOD_TEST_STATEMENT = re.compile(
    r"\bblood\b", re.IGNORECASE
)
_BLOOD_GLUCOSE_STATEMENT = re.compile(
    r"\bblood\s+(?:glucose|sugar)\b|\bblood\s+(?:test|testing)\b",
    re.IGNORECASE,
)
_CONFIRMED_BLOOD_COLLECTION = re.compile(
    r"\b(?:blood|blood sample|blood samples)\b.{0,40}"
    r"\b(?:drawn|collected|taken|obtained)\b"
    r"|\b(?:draw|collect|take|obtain)(?:s|ed|ing)?\b.{0,30}"
    r"\b(?:blood|blood sample)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_REFERENCE = re.compile(r"\b(?:it|that|they|them|those|these)\b")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _lookup_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", normalize_text(text)):
        for tok in _SYNONYMS.get(raw, (raw,)):
            tok = _stem(tok)
            if tok in _STOPWORDS or tok in _LOOKUP_STOP:
                continue
            if len(tok) > 2 or tok.isdigit():
                out.add(tok)
    return out


def _lookup_items(guide: PatientGuide) -> List[GuideItem]:
    return [i for i in guide.all_items() if i.approved and not i.rejected]


def _correct_spelling(tokens: set[str], vocabulary: set[str]) -> set[str]:
    fixed = set()
    for tok in tokens:
        if tok in vocabulary or len(tok) < 5:
            fixed.add(tok)
            continue
        close = difflib.get_close_matches(tok, vocabulary, n=1, cutoff=0.84)
        fixed.add(close[0] if close else tok)
    return fixed


def _answer_from_items(
    matches: Sequence[GuideItem],
    *,
    intent: str = "unsupported",
) -> PatientAnswer:
    lines = [
        "Here is what the study team has approved that relates to your question:",
        "",
    ]
    lines += [_line(item) for item in matches]
    resolved_intent = (
        "procedures"
        if matches and matches[0].section == "procedures" and intent == "unsupported"
        else intent
    )
    return _answer(
        "\n".join(lines),
        resolved_intent,
        citations=[_citation(item) for item in matches],
        confidence="high",
    )


def _lookup_answer(guide: PatientGuide, query: str) -> Optional[PatientAnswer]:
    """Answer from approved items via BM25 (Beacon-style) with token fallback."""
    items = _lookup_items(guide)
    if not items:
        return None
    item_tokens = {id(i): _lookup_tokens(_item_blob(i)) for i in items}
    vocabulary = set().union(*item_tokens.values()) if item_tokens else set()
    query_tokens = _correct_spelling(_lookup_tokens(query), vocabulary)
    if not query_tokens:
        return None

    # Ambiguous one-word questions still ask for clarification.
    if len(query_tokens) == 1:
        scored = [(len(query_tokens & item_tokens[id(i)]), i) for i in items]
        scored = [(n, i) for n, i in scored if n]
        if not scored:
            return None
        best = max(n for n, _ in scored)
        matches = [i for n, i in scored if n == best]
        phrases = sorted(
            {
                m.group(0)
                for i in matches
                for m in re.finditer(
                    rf"\b{re.escape(next(iter(query_tokens)))}\w*\s+\w+",
                    normalize_text(i.plain_text),
                )
            }
        )
        if len(matches) > 1:
            options = " or ".join(phrases[:3]) if phrases else "which topic"
            return _answer(
                f"Could you tell me a little more? For example, do you mean {options}?"
                if phrases
                else "Could you tell me a little more about what you would like to know?",
                "clarification",
                limitations=["The question was too short to match one approved topic."],
            )

    passages = retrieve_approved_passages(guide, query, top_k=5, min_score=0.5)
    if passages:
        best_score = passages[0].score
        # Keep near-best hits only, then require token overlap so shared words
        # like "blood" do not pull every blood-related statement.
        min_overlap = 2 if len(query_tokens) > 1 else 1
        tight: List[GuideItem] = []
        for row in passages:
            if row.score < best_score * 0.8 and row.score < best_score - 0.35:
                continue
            overlap = len(query_tokens & item_tokens[id(row.item)])
            if overlap >= min_overlap:
                tight.append(row.item)
        if tight:
            return _answer_from_items(tight)

    # Fallback: strict token overlap when BM25 finds nothing strong enough.
    scored = [(len(query_tokens & item_tokens[id(i)]), i) for i in items]
    scored = [(n, i) for n, i in scored if n]
    if not scored:
        return None
    best = max(n for n, _ in scored)
    if len(query_tokens) > 1 and best < 2:
        return None
    matches = [i for n, i in scored if n == best]
    return _answer_from_items(matches)


def _with_timing_note(query: str, answer: PatientAnswer) -> PatientAnswer:
    """Do not let a topic answer imply timing the approved guide never states."""
    if (
        not answer.citations
        or not _TIMING_QUESTION.search(normalize_text(query))
        or any(_TIMING_TEXT.search(normalize_text(c.statement)) for c in answer.citations)
    ):
        return answer
    note = (
        "The approved guide doesn't say exactly when this happens. Please ask "
        "the clinical trial coordinator for the timing."
    )
    return answer.model_copy(update={"answer_text": f"{note}\n\n{answer.answer_text}"})


def _blood_follow_up_context(
    query: str,
    history: Optional[Sequence[dict[str, object]]],
) -> str:
    """Use only the last cited, approved blood-related answer to resolve a follow-up."""
    if not history or len(_lookup_tokens(query)) > 5:
        return ""
    if not _FOLLOW_UP_REFERENCE.search(normalize_text(query)):
        return ""

    previous = history[-1]
    previous_query = previous.get("query")
    answer = previous.get("answer")
    if not isinstance(previous_query, str) or not isinstance(answer, PatientAnswer):
        return ""
    if answer.intent != "procedures" or not answer.citations:
        return ""
    if not any(
        re.search(r"\b(?:blood|glucose|sugar)\b", citation.statement, re.IGNORECASE)
        for citation in answer.citations
    ):
        return ""
    return previous_query


def _blood_collection_answer(guide: PatientGuide) -> PatientAnswer:
    """Answer blood-draw questions without conflating them with glucose testing."""
    approved = [
        item
        for item in _approved(guide.procedures)
        if _BLOOD_TEST_STATEMENT.search(item.plain_text)
    ]
    collection_items = [
        item for item in approved
        if _CONFIRMED_BLOOD_COLLECTION.search(item.plain_text)
    ]
    if collection_items:
        return _answer(
            "Here is what the study team has approved about blood collection:\n\n"
            + "\n".join(_line(item) for item in collection_items),
            "procedures",
            citations=[_citation(item) for item in collection_items],
            confidence="high",
        )

    glucose_items = [
        item for item in approved
        if _BLOOD_GLUCOSE_STATEMENT.search(item.plain_text)
    ]
    lines = []
    if glucose_items:
        lines = [
            "The approved guide says:",
            "",
            *[_line(item) for item in glucose_items],
            "",
        ]
    lines.append(
        "The approved guide does not specify whether a blood sample will be "
        "drawn. Please ask the clinical trial coordinator to confirm."
    )
    return _answer(
        "\n".join(lines),
        "procedures",
        citations=[_citation(item) for item in glucose_items],
        confidence="partial" if glucose_items else "unavailable",
        limitations=["The approved guide does not specify whether blood is drawn."],
    )


def _eligibility_refusal(guide: PatientGuide) -> PatientAnswer:
    inclusions = _approved(guide.who_can_join)
    exclusions = _approved(guide.who_cannot_join)
    pending = sum(
        len(s.items) - len(_approved(s))
        for s in (guide.who_can_join, guide.who_cannot_join)
    )

    lines = [
        "I can't tell you whether you personally are eligible for this "
        "study. That depends on your individual situation, and only the "
        "study team can decide.",
    ]
    if inclusions or exclusions:
        lines += [
            "",
            "Here are the general criteria the study team has approved, "
            "which you can talk through with them:",
        ]
    if inclusions:
        lines += ["", "**Who may be able to join:**"] + [_line(i) for i in inclusions]
    if exclusions:
        lines += ["", "**Who may not be able to join:**"] + [_line(i) for i in exclusions]
    if not inclusions and not exclusions:
        lines += [
            "",
            "The study team is still reviewing the joining criteria, so "
            "I can't share them yet.",
        ]
    elif pending:
        lines += ["", _pending_note(pending)]

    return _answer(
        "\n".join(lines),
        "eligibility_decision_refused",
        citations=[_citation(i) for i in inclusions + exclusions],
        confidence="high" if (inclusions or exclusions) else "unavailable",
        limitations=["This assistant does not evaluate personal eligibility."],
    )


def _medical_advice_refusal(query: str = "") -> PatientAnswer:
    text = normalize_text(query)
    if re.search(
        r"\bguarante|\bpromise\b|\bassure\b|\bcompletely safe\b|\btotally safe\b",
        text,
    ):
        return _answer(
            "I can't guarantee that this study is safe or promise any "
            "outcome. I can only share general safety information the study "
            "team has approved. For questions about your personal risks, "
            "please talk with the study team or a qualified clinician.",
            "medical_advice_refused",
            limitations=["Safety guarantees and outcome promises are out of scope."],
        )
    return _answer(
        "I can't give medical advice, a diagnosis, or treatment "
        "recommendations. I can only share general information the study "
        "team has approved, such as visits, procedures, and the listed "
        "criteria. Please talk with the study team or a qualified clinician "
        "about your health.",
        "medical_advice_refused",
        limitations=["Medical advice and personal clinical decisions are out of scope."],
    )


def _format_item_clause(item: GuideItem, *, with_page: bool = True) -> str:
    page = (
        f" (protocol page {item.source_page})"
        if with_page and item.source_page
        else ""
    )
    return f"{item.plain_text.rstrip('.')}{page}"


def _join_plain(items: Sequence[GuideItem]) -> str:
    texts = [item.plain_text.rstrip(".") for item in items]
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    if len(texts) == 2:
        return f"{texts[0]} and {texts[1]}"
    return ", ".join(texts[:-1]) + f", and {texts[-1]}"


def _source_footer(citations: Sequence[PatientCitation]) -> str:
    pages = sorted({c.source_page for c in citations if c.source_page is not None})
    if not pages:
        return ""
    label = "pages" if len(pages) > 1 else "page"
    return f"\n\nSource: Protocol {label} {', '.join(map(str, pages))}."


def _study_experience_answer(guide: PatientGuide) -> PatientAnswer:
    """Broad 'what happens in the study' answer as readable Markdown bullets."""
    visits = _approved(guide.visits)
    procedures = _approved(guide.procedures)
    safety = _approved(guide.safety)
    citations = [_citation(item) for item in visits + procedures + safety]
    if not citations:
        return _answer(
            f"{_MISSING_DETAIL} {_ASK_ABOUT}",
            "study_overview",
            limitations=["No approved visit, procedure, or safety items available."],
        )

    lines = ["Here's what the approved information says about this study:", ""]
    if visits:
        lines.append(f"- **Study visits:** {_join_plain(visits)}.")
    if procedures:
        lines.append(f"- **Tests:** {_join_plain(procedures)}.")
    if safety:
        lines.append(f"- **Safety:** {_join_plain(safety)}.")
    text = "\n".join(lines) + _source_footer(citations)
    return _answer(
        text,
        "study_overview",
        citations=citations,
        confidence="high",
    )


def _overview(guide: PatientGuide) -> PatientAnswer:
    """Prefer a multi-section experience summary when any approved facts exist."""
    if (
        _approved(guide.visits)
        or _approved(guide.procedures)
        or _approved(guide.safety)
    ):
        return _study_experience_answer(guide)
    title = guide.trial_title or "this clinical trial"
    protocol = f" (protocol ID {guide.protocol_id})" if guide.protocol_id else ""
    return _answer(
        f"This guide is for the study “{title}”{protocol}. {_ASK_ABOUT}",
        "study_overview",
        confidence="high",
    )


def _timeline_answer(
    guide: PatientGuide,
    *,
    concise: bool = False,
) -> PatientAnswer:
    """Explain approved visit timing without inventing an official study end."""
    visits = _approved(guide.visits)
    if not visits:
        return _answer(
            f"{_MISSING_DETAIL} I can still help with who may join, tests, "
            "or safety if those are in the approved guide.",
            "visits",
            limitations=["No approved visit schedule is available."],
        )

    citations = [_citation(item) for item in visits]
    explicit_end = any(
        _EXPLICIT_STUDY_END.search(f"{item.plain_text} {item.original_text}")
        for item in visits
    )
    follow_up_days = [
        item.plain_text
        for item in visits
        if re.search(r"follow[- ]?up", item.plain_text, re.I)
    ]
    if not follow_up_days:
        follow_up_days = [
            item.plain_text
            for item in visits
            if re.search(r"day\s+\d+", item.plain_text, re.I)
        ]

    if concise and not explicit_end:
        highlight = follow_up_days[-1] if follow_up_days else _join_plain(visits)
        body = (
            f"The approved guide lists {highlight.rstrip('.')}. "
            f"{_NOT_SPECIFIED} when the study officially ends. Please ask "
            "your study coordinator for the complete timeline."
        )
        return _answer(
            body + _source_footer(citations),
            "visits",
            citations=citations,
            confidence="partial",
            limitations=["Approved guide does not state an official study end date."],
        )

    body = f"The approved guide lists these visit times: {_join_plain(visits)}."
    if explicit_end:
        note = " That is what the approved guide says about the study timeline."
    else:
        note = (
            f" {_NOT_SPECIFIED} when the study officially ends, so a listed "
            "follow-up day should not be treated as the end of the study. "
            "Please ask your study coordinator for the full timeline."
        )
    return _answer(
        body + note + _source_footer(citations),
        "visits",
        citations=citations,
        confidence="high" if explicit_end else "partial",
        limitations=(
            []
            if explicit_end
            else ["Approved guide does not state an official study end date."]
        ),
    )


def _test_duration_answer(
    guide: PatientGuide,
    history: Optional[Sequence[dict[str, object]]],
    query: str = "",
) -> PatientAnswer:
    """Handle test-duration questions without inventing procedure timing."""
    procedures = _approved(guide.procedures)
    previous = _previous_cited_answer(history)
    # Prefer a named test in the question when it matches an approved item.
    named = []
    qtokens = _lookup_tokens(query) if query else set()
    if qtokens and procedures:
        named = [
            item
            for item in procedures
            if _lookup_tokens(_item_blob(item)) & (qtokens - _GENERIC_TOPIC)
        ]
    if named:
        citations = [_citation(item) for item in named]
        text = (
            f"The approved guide lists: {_join_plain(named)}. "
            f"{_NOT_SPECIFIED} how long that test takes. "
            "Please ask your study coordinator if you need that timing."
            f"{_source_footer(citations)}"
        )
        return _answer(
            text,
            "procedures",
            citations=citations,
            confidence="partial",
            limitations=["Approved guide does not state how long a test lasts."],
        )
    if previous and previous[1].intent == "procedures" and previous[1].citations:
        related = "; ".join(c.statement for c in previous[1].citations[:3])
        text = (
            f"The approved tests mentioned earlier are: {related}. "
            f"{_NOT_SPECIFIED} how long each individual test takes. "
            "Please ask your study coordinator if you need that timing."
        )
        return _answer(
            text,
            "procedures",
            citations=list(previous[1].citations),
            confidence="partial",
            limitations=["Approved guide does not state how long a test lasts."],
        )
    if procedures:
        citations = [_citation(item) for item in procedures]
        text = (
            f"The approved tests are: {_join_plain(procedures)}. "
            f"{_NOT_SPECIFIED} how long each individual test takes. "
            "Please ask your study coordinator if you need that timing."
            f"{_source_footer(citations)}"
        )
        return _answer(
            text,
            "procedures",
            citations=citations,
            confidence="partial",
            limitations=["Approved guide does not state how long a test lasts."],
        )
    return _answer(
        f"{_NOT_SPECIFIED} {_ASK_ABOUT}",
        "procedures",
        limitations=["No approved procedure timing is available."],
    )


def _ambiguous_end_clarification() -> PatientAnswer:
    return _answer(
        "Do you mean when the study itself ends, or how long a specific test "
        "takes? I can share the approved visit schedule, or the listed tests.",
        "clarification",
        limitations=["Ambiguous end/duration question; clarification requested."],
    )


_VAGUE_FOLLOW_UP = re.compile(
    r"(?:(?:ok|okay|sure|yes|yeah|yep|alright|please),?\s+)*"
    r"(?:and |what about |how about |so )?(?:it|that|this|those|them|more|"
    r"why|how|really|tell me more|more info|more information|explain|"
    r"explain more|go on|elaborate|what else)"
)
_AFFIRM_FOLLOW_UP = re.compile(
    r"^(?:(?:ok|okay|sure|yes|yeah|yep|alright|please),?\s+)*"
    r"(?:yes|yep|yeah|yup|ok|okay|sure|please|go ahead|go on|tell me|"
    r"explain|explain please|yes please|yes,? please|tell me more|do it|"
    r"please do|please explain|sure please|okay please|"
    r"can you explain|could you explain)(?:[.!?])?$"
)
_TESTING_PERIOD_QUESTION = re.compile(
    r"\b(?:how (?:long|many days) (?:will|does|do) (?:the )?test(?:s|ing)? "
    r"(?:continue|last|take|prolong|go on)\b|"
    r"how many days does the tests? prolong\b|"
    r"any days\b|"
    r"how long will testing continue\b|"
    r"testing (?:period|duration|schedule)\b)",
    re.IGNORECASE,
)


_FOLLOW_UP_INTENTS = set(_SECTION_INTENTS) | {
    "study_overview",
    "unsupported",
    "clarification",
}


def _previous_turn(
    history: Optional[Sequence[dict[str, object]]],
) -> Optional[Tuple[str, PatientAnswer]]:
    if not history:
        return None
    previous = history[-1]
    query = previous.get("query")
    answer = previous.get("answer")
    if isinstance(query, str) and isinstance(answer, PatientAnswer):
        return query, answer
    return None


def _previous_cited_answer(
    history: Optional[Sequence[dict[str, object]]],
) -> Optional[Tuple[str, PatientAnswer]]:
    """Return the latest cited turn, skipping conversational ACK noise."""
    state = resolve_conversation_state(history)
    if state.last_cited_answer is None or state.last_cited_query is None:
        return None
    if state.last_cited_intent not in _FOLLOW_UP_INTENTS:
        return None
    return state.last_cited_query, state.last_cited_answer


def _offered_topic(answer: PatientAnswer) -> Optional[str]:
    """Topic offered in a conversational clarification (no citations yet)."""
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


def _duration_follow_up_from_state(
    query: str,
    guide: PatientGuide,
    state: ConversationState,
    history: Optional[Sequence[dict[str, object]]],
) -> Optional[PatientAnswer]:
    """Resolve bare 'How long?' / 'How long do they take?' using prior topic."""
    if not (
        is_bare_duration_follow_up(query)
        or re.search(
            r"\bwhen does (?:this|it|that) (?:end|finish|stop)\b"
            r"|\bhow long (?:will|does|do|is) "
            r"(?:this|it|that|they|them|these|those|each(?: one| test)?)\b",
            normalize_text(query),
        )
    ):
        return None
    topic = state.topic or state.last_cited_intent or state.offered_topic or state.prior_intent
    if topic == "procedures":
        return _test_duration_answer(guide, history, query)
    if topic in {"visits", "study_overview"}:
        return _timeline_answer(guide, concise=True)
    if topic is None and is_bare_duration_follow_up(query):
        return _ambiguous_end_clarification()
    return None


def _pick_visit(
    visits: Sequence[GuideItem],
    *,
    prefer: str,
) -> GuideItem:
    if prefer == "first":
        return next(
            (
                item
                for item in visits
                if re.search(r"screening|day\s*0|baseline|first", item.plain_text, re.I)
            ),
            visits[0],
        )
    follow = next(
        (
            item
            for item in visits
            if re.search(r"follow[- ]?up|day\s*[1-9]\d*", item.plain_text, re.I)
        ),
        None,
    )
    if follow is not None:
        return follow
    return visits[-1] if len(visits) > 1 else visits[0]


def _first_or_next_visit_answer(guide: PatientGuide, query: str) -> PatientAnswer:
    """Answer first/next visit asks; cover both when the message asks for both."""
    visits = _approved(guide.visits)
    if not visits:
        return _answer(
            f"{_NOT_SPECIFIED} I don't have an approved visit schedule yet. "
            "Please ask your study team.",
            "visits",
            limitations=["No approved visit schedule is available."],
        )
    text = normalize_text(query)
    want_first = bool(re.search(r"\b(?:first|initial|screening)\b", text))
    want_next = bool(re.search(r"\b(?:next|upcoming|follow[- ]?up)\b", text))
    if want_first and want_next and len(visits) >= 1:
        first = _pick_visit(visits, prefer="first")
        nxt = _pick_visit(visits, prefer="next")
        chosen = [first]
        if nxt.item_id != first.item_id or nxt.plain_text != first.plain_text:
            chosen.append(nxt)
        citations = [_citation(item) for item in chosen]
        lines = [f"- {item.plain_text.rstrip('.')}" for item in chosen]
        return _answer(
            "The approved guide lists these visits:\n"
            + "\n".join(lines)
            + _source_footer(citations),
            "visits",
            citations=citations,
            confidence="high",
        )
    if want_first and not want_next:
        chosen_item = _pick_visit(visits, prefer="first")
    elif want_next and not want_first:
        chosen_item = _pick_visit(visits, prefer="next")
    else:
        chosen_item = visits[0]
    citation = _citation(chosen_item)
    return _answer(
        f"The approved guide lists: {chosen_item.plain_text.rstrip('.')}."
        f"{_source_footer([citation])}",
        "visits",
        citations=[citation],
        confidence="high",
    )


def _visit_meaning_answer(guide: PatientGuide, prior: PatientAnswer) -> PatientAnswer:
    """Restate the last visit answer briefly — no extra study-end disclaimer."""
    if prior.citations:
        lines = [f"- {c.statement}" for c in prior.citations[:3]]
        pages = sorted({c.source_page for c in prior.citations if c.source_page})
        source = (
            f"\n\n(Source: protocol page{'s' if len(pages) > 1 else ''} "
            f"{', '.join(map(str, pages))})"
            if pages
            else ""
        )
        return _answer(
            "Here is what that visit information means, from the approved "
            "guide:\n\n" + "\n".join(lines) + source,
            "visits",
            citations=list(prior.citations),
            confidence="high",
        )
    return _first_or_next_visit_answer(guide, "first visit")


def _procedures_summary_answer(guide: PatientGuide) -> PatientAnswer:
    procedures = _approved(guide.procedures)
    if not procedures:
        return _answer(
            f"{_NOT_SPECIFIED} I don't have approved tests or procedures to "
            "share yet. Please ask your study coordinator.",
            "procedures",
            limitations=["No approved procedure items available."],
        )
    citations = [_citation(item) for item in procedures]
    text = (
        f"Here are the approved tests and procedures: {_join_plain(procedures)}."
        f"{_source_footer(citations)}"
    )
    return _answer(
        text,
        "procedures",
        citations=citations,
        confidence="high",
    )


def _testing_period_answer(guide: PatientGuide) -> PatientAnswer:
    """
    Distinguish visit schedule facts from unspecified test duration / study end.
    """
    visits = _approved(guide.visits)
    procedures = _approved(guide.procedures)
    citations = [_citation(item) for item in visits + procedures]
    parts: List[str] = []
    if procedures:
        parts.append(
            f"The approved guide lists these tests: {_join_plain(procedures)}."
        )
    if visits:
        parts.append(
            f"Visit timing in the approved guide: {_join_plain(visits)}."
        )
    parts.append(
        f"{_NOT_SPECIFIED} how long each individual test takes, how many days "
        "testing continues overall, or when the study officially ends. "
        "Please ask your study coordinator for those details."
    )
    if not citations:
        return _answer(
            " ".join(parts),
            "visits",
            limitations=["No approved visit or procedure timing available."],
        )
    return _answer(
        " ".join(parts) + _source_footer(citations),
        "visits",
        citations=citations,
        confidence="partial",
        limitations=[
            "Approved guide does not state individual test duration, overall "
            "testing period, or official study end."
        ],
    )


def _affirmative_follow_up(
    query: str,
    guide: Optional[PatientGuide],
    history: Optional[Sequence[dict[str, object]]],
) -> Optional[PatientAnswer]:
    """Accept 'yes' / 'explain' after a conversational offer to describe tests."""
    if guide is None:
        return None
    text = normalize_text(query)
    if not (
        _AFFIRM_FOLLOW_UP.fullmatch(text)
        or is_affirmative_follow_up(query)
        or is_explain_request(query)
    ):
        return None
    state = resolve_conversation_state(history)
    topic = state.offered_topic
    if topic is None and state.prior_answer is not None:
        topic = _offered_topic(state.prior_answer)
    if topic == "procedures":
        return _procedures_summary_answer(guide)
    if topic == "study_overview":
        return _overview(guide)
    return None


def _meaning_follow_up(
    query: str,
    guide: Optional[PatientGuide],
    history: Optional[Sequence[dict[str, object]]],
) -> Optional[PatientAnswer]:
    """Clarify 'what does that mean?' using the last cited topic — no new facts."""
    if guide is None:
        return None
    text = normalize_text(query)
    wants_meaning = bool(
        re.search(r"\bwhat does (?:that|this|it) mean\b", text)
        or re.search(r"\b(?:can|could) you explain\b", text)
        or text in {"please explain", "explain please", "explain that"}
    )
    if not wants_meaning:
        return None
    state = resolve_conversation_state(history)
    topic = state.topic or state.last_cited_intent or state.offered_topic
    if state.offered_topic == "procedures" or topic == "procedures":
        return _procedures_summary_answer(guide)
    if state.offered_topic == "study_overview" or topic == "study_overview":
        return _overview(guide)
    if topic == "visits" and state.last_cited_answer is not None:
        return _visit_meaning_answer(guide, state.last_cited_answer)
    if state.last_cited_answer is not None:
        prior = state.last_cited_answer
        return prior.model_copy(
            update={
                "answer_text": (
                    "Here is that again, in simpler terms:\n\n" + prior.answer_text
                )
            }
        )
    return None


_FOLLOW_UP_FILLER = frozenset({
    "long", "last", "lasts", "often", "many", "much", "times", "soon",
    "happen", "happens", "take", "takes", "long", "more", "else", "work",
})


def _topic_follow_up_context(
    query: str,
    history: Optional[Sequence[dict[str, object]]],
    vocabulary: set[str],
) -> str:
    """
    Carry the previous topic into a short pronoun / bare-duration follow-up.

    The new question's own content words must be known to the approved guide;
    otherwise the old topic is NOT carried, so unrelated answers cannot leak
    in (e.g. "is it tasty?" after a visits answer).
    """
    previous = _previous_cited_answer(history)
    tokens = _lookup_tokens(query)
    if previous is None or len(tokens) > 3:
        return ""
    text = normalize_text(query)
    has_reference = bool(_FOLLOW_UP_REFERENCE.search(text))
    bare_duration = is_bare_duration_follow_up(query)
    if not has_reference and not bare_duration:
        return ""
    unknown = {t for t in tokens if t not in vocabulary and t not in _FOLLOW_UP_FILLER}
    return "" if unknown else previous[0]


_ORDINAL = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "last": -1, "final": -1,
}
_ORDINAL_QUERY = re.compile(
    r"(?:and |what about |how about |tell me about |explain |so )?(?:the )?"
    r"(first|1st|second|2nd|third|3rd|last|final)(?: one| item| point| visit| test| procedure)?"
)


def _ordinal_follow_up(
    query: str,
    history: Optional[Sequence[dict[str, object]]],
) -> Optional[PatientAnswer]:
    """Resolve "what about the second one?" against the last cited answer."""
    match = _ORDINAL_QUERY.fullmatch(normalize_text(query))
    if not match:
        return None
    previous = _previous_cited_answer(history)
    if previous is None:
        return _answer(
            "Which one do you mean? Ask me about a topic first, such as "
            "the visits or tests, and then I can go through them with you.",
            "unsupported",
            limitations=["Ambiguous reference; clarification requested."],
        )
    citations = previous[1].citations
    index = _ORDINAL[match.group(1)]
    if index >= len(citations):
        return _answer(
            f"I only listed {len(citations)} item(s) just now. Which one would "
            "you like to hear about?",
            previous[1].intent,
            limitations=["Reference beyond the listed items."],
        )
    chosen = citations[index]
    page = f" (protocol page {chosen.source_page})" if chosen.source_page else ""
    return _answer(
        f"That one is:\n\n- {chosen.statement}{page}",
        previous[1].intent,
        citations=[chosen],
        confidence="high",
    )


def _merge_multi_answers(
    parts: Sequence[Tuple[str, PatientAnswer]],
) -> PatientAnswer:
    """Combine per-question answers into one response with shared citations."""
    if not parts:
        return _answer(
            f"{_NOT_SPECIFIED} {_ASK_ABOUT}",
            "unsupported",
            limitations=["Empty multi-part question."],
        )
    if len(parts) == 1:
        return parts[0][1]

    blocks: List[str] = []
    citations: List[PatientCitation] = []
    seen_statements: set[str] = set()
    limitations: List[str] = []
    refusal_intent: Optional[str] = None
    clinical_intent: Optional[str] = None

    for _part_query, answer in parts:
        body = (answer.answer_text or "").strip()
        if body:
            blocks.append(body)
        for citation in answer.citations:
            key = citation.statement.strip()
            if key and key not in seen_statements:
                seen_statements.add(key)
                citations.append(citation)
        limitations.extend(answer.limitations or [])
        if answer.intent in {
            "medical_advice_refused",
            "eligibility_decision_refused",
        }:
            refusal_intent = answer.intent
        elif clinical_intent is None and answer.intent not in {
            "conversational",
            "unsupported",
            "clarification",
        }:
            clinical_intent = answer.intent

    intent = refusal_intent or clinical_intent or parts[-1][1].intent
    confidence = "partial" if refusal_intent and citations else parts[-1][1].confidence
    # One shared source footer at the end when several blocks already embed pages.
    text = "\n\n".join(blocks)
    return _answer(
        text,
        intent,
        citations=citations,
        confidence=confidence,
        limitations=limitations[:8],
    )


def _answer_multi_questions(
    parts: Sequence[str],
    guide: Optional[PatientGuide],
    history: Optional[Sequence[dict[str, object]]],
) -> PatientAnswer:
    """
    Answer each related question in one message.

    Safe parts use approved evidence. Unsafe medical/eligibility asks are
    refused for that part only and never bypass validation.
    """
    synthetic: List[dict[str, object]] = list(history or [])
    answered: List[Tuple[str, PatientAnswer]] = []

    for part in parts:
        part_intent = classify_patient_intent(part)
        if part_intent == "medical_advice_refused":
            answer = _medical_advice_refusal(part)
        elif part_intent == "eligibility_decision_refused":
            answer = (
                _eligibility_refusal(guide)
                if guide is not None
                else _answer(
                    "I can't decide whether you personally qualify. Please "
                    "ask the study team.",
                    "eligibility_decision_refused",
                    limitations=["Personal eligibility decisions are out of scope."],
                )
            )
        else:
            answer = _answer_single(part, guide, synthetic)
        answered.append((part, answer))
        synthetic.append({"query": part, "answer": answer})

    return _merge_multi_answers(answered)


def _answer_core(
    query: str,
    guide: Optional[PatientGuide],
    history: Optional[Sequence[dict[str, object]]] = None,
) -> PatientAnswer:
    """Answer a patient question using approved Patient Guide items only."""
    if not query or not query.strip():
        return _answer(
            f"Please ask a question about the study. {_ASK_ABOUT}",
            "unsupported",
            limitations=["Empty query."],
        )

    # Multiple related questions in one message: answer each part (safety first).
    parts = split_related_questions(query)
    if len(parts) > 1:
        return _answer_multi_questions(parts, guide, history)

    return _answer_single(query, guide, history)


def _answer_single(
    query: str,
    guide: Optional[PatientGuide],
    history: Optional[Sequence[dict[str, object]]] = None,
) -> PatientAnswer:
    """Answer one (already-split) patient question from the approved guide."""
    state = resolve_conversation_state(history)

    # Affirmatives after "are you asking about the tests?" must run before
    # social ack handling ("sure"/"ok") so we explain instead of re-asking.
    affirmed = _affirmative_follow_up(query, guide, history)
    if affirmed is not None:
        return affirmed
    meaning = _meaning_follow_up(query, guide, history)
    if meaning is not None:
        return meaning

    # Small talk and arrival-style statements never invent protocol facts.
    social = social_answer(query)
    if social is not None:
        return social

    # Explicit "say that again" restates the last cited answer only.
    if history and re.search(
        r"\b(say that again|repeat that|repeat what you said|one more time)\b",
        normalize_text(query),
    ):
        previous = _previous_cited_answer(history)
        if previous is not None:
            return previous[1].model_copy(
                update={
                    "answer_text": (
                        "Here is that again, in case it helps:\n\n"
                        + previous[1].answer_text
                    )
                }
            )

    vocabulary = set()
    if guide is not None:
        for item in _lookup_items(guide):
            vocabulary |= _lookup_tokens(_item_blob(item))
    # Normalize everyday phrasing first so spelling correction cannot break
    # multi-word intents (e.g. "happens" → "happen" before overview matching).
    query = correct_query_spelling(normalize_phrasing(query), vocabulary)

    ordinal = _ordinal_follow_up(query, history)
    if ordinal is not None:
        return ordinal

    if (
        _VAGUE_FOLLOW_UP.fullmatch(normalize_text(query))
        or is_explain_request(query)
        or is_more_detail_request(query)
    ):
        offered = _affirmative_follow_up("explain", guide, history)
        if offered is not None:
            return offered
        state_now = resolve_conversation_state(history)
        previous = _previous_cited_answer(history)
        topic = state_now.topic or (previous[1].intent if previous else None)
        if previous is None and topic is None:
            return _answer(
                "Could you tell me a little more about what you'd like to "
                "know? For example, you could ask about visits, tests, risks "
                "or who can join the study.",
                "clarification",
                limitations=["Ambiguous question; clarification requested."],
            )
        # "Tell me more" after a cited answer: no new facts — point to the team.
        if is_more_detail_request(query) and previous is not None:
            label = {
                "procedures": "the tests and procedures",
                "visits": "the visits",
                "safety": "safety",
                "study_overview": "the study",
            }.get(topic or previous[1].intent, "this topic")
            return _answer(
                f"That's everything the approved guide says about {label}. "
                "For more detail, please ask your study team. Or ask me about "
                "another part of the study.",
                previous[1].intent,
                limitations=["No further approved information to add."],
            )
        if topic == "procedures" and not is_more_detail_request(query):
            return _procedures_summary_answer(guide)
        if (
            topic == "visits"
            and previous
            and re.search(r"\bwhat does (?:that|this|it) mean\b", normalize_text(query))
        ):
            return _visit_meaning_answer(guide, previous[1])
        if previous is None:
            return _answer(
                "Could you tell me a little more about what you'd like to "
                "know? For example, you could ask about visits, tests, risks "
                "or who can join the study.",
                "clarification",
                limitations=["Ambiguous question; clarification requested."],
            )
        return _answer(
            "That's everything the approved guide says on this topic. For "
            "more detail, please ask your study team. Or ask me about "
            "another part of the study.",
            previous[1].intent,
            limitations=["No further approved information to add."],
        )

    follow_up_context = _blood_follow_up_context(query, history)
    if not follow_up_context:
        follow_up_context = _topic_follow_up_context(query, history, vocabulary)
    # History expands references only; retrieval still uses approved guide text.
    retrieval_query = expand_retrieval_query(query, state)
    contextual_query = f"{follow_up_context} {retrieval_query}".strip()
    intent = _classify_intent(contextual_query)
    if intent == "procedures" and re.search(
        r"\bwhat (?:will|would|is going to) happen\b", normalize_text(query)
    ) and re.search(r"\bvisits?\b", normalize_text(query)):
        intent = "visits"
    if follow_up_context and re.search(
        r"\b(skip|miss|avoid|refuse|decline|delay|postpone|stop|change|cancel)\b",
        normalize_text(query),
    ):
        intent = "medical_advice_refused"

    # Refusals never depend on what has been approved.
    if intent == "medical_advice_refused":
        return _medical_advice_refusal(query)

    if guide is not None and guide.analysis_complete is not True:
        return _answer(
            "The study team cannot share this guide yet because protocol "
            "analysis is incomplete or its completeness could not be verified.",
            "not_yet_approved",
            limitations=[
                "Patient publication is blocked until protocol analysis is complete."
            ],
        )

    if guide is None or guide.approved_count() == 0:
        return _not_published()

    oos = out_of_scope_topic(query)
    if oos and _classify_intent(query) == "unsupported":
        # Answer only if an approved item itself mentions the topic.
        found = _lookup_answer(guide, oos[1])
        return found or _answer(
            out_of_scope_reply(oos[0]),
            "unsupported",
            limitations=["Topic is not covered by the approved guide."],
        )

    blood_collection_question = bool(
        _BLOOD_COLLECTION_QUESTION.search(query)
        or (
            follow_up_context
            and _BLOOD_COLLECTION_FOLLOW_UP.search(query)
        )
    )
    if blood_collection_question:
        return _blood_collection_answer(guide)

    if intent == "eligibility_decision_refused":
        return _eligibility_refusal(guide)

    # Context-aware duration / end questions (after refusals and publication gates).
    duration = _duration_follow_up_from_state(query, guide, state, history)
    if duration is not None:
        return duration

    # Individual test duration before visit/study timeline (avoids Day-30 answers).
    if _TEST_DURATION_QUESTION.search(normalize_text(query)) or (
        _TEST_DURATION_QUESTION.search(normalize_text(contextual_query))
    ):
        previous = _previous_cited_answer(history)
        named_test = bool(
            re.search(
                r"\b(?:blood|glucose|pressure|testing|tests?|procedure|procedures?)\b",
                normalize_text(query),
            )
        )
        if previous is None and not named_test and not re.search(
            r"\b(study|trial|visit|visits)\b", normalize_text(query)
        ):
            return _ambiguous_end_clarification()
        return _test_duration_answer(guide, history, query)

    if _TESTING_PERIOD_QUESTION.search(normalize_text(query)) or (
        _TESTING_PERIOD_QUESTION.search(normalize_text(contextual_query))
    ):
        return _testing_period_answer(guide)

    if _FIRST_OR_NEXT_VISIT.search(normalize_text(query)):
        return _first_or_next_visit_answer(guide, query)

    if re.search(
        r"\bwhat happens? after (?:that|this|it)\b",
        normalize_text(query),
    ):
        previous = _previous_cited_answer(history) or _previous_turn(history)
        if previous and _offered_topic(previous[1]) == "procedures":
            return _testing_period_answer(guide)
        if previous and previous[1].intent in {"procedures", "visits", "study_overview"}:
            return _timeline_answer(guide, concise=True)
        return _timeline_answer(guide)

    def _concise_timeline() -> bool:
        previous = _previous_cited_answer(history)
        if previous is None:
            return False
        prior = previous[1]
        prior_text = prior.answer_text.casefold()
        return prior.intent in {"visits", "study_overview"} and (
            "day" in prior_text
            or "visit" in prior_text
            or "officially ends" in prior_text
            or "not specified" in prior_text
            or "timeline" in prior_text
        )

    if _TIMELINE_QUESTION.search(normalize_text(contextual_query)) or (
        _TIMELINE_QUESTION.search(normalize_text(query))
    ):
        return _timeline_answer(guide, concise=_concise_timeline())

    # "When does this end?" after a study/visit turn → timeline; after tests → clarify.
    if re.search(
        r"\bwhen does (?:this|it|that) (?:end|finish|stop)\b",
        normalize_text(query),
    ):
        previous = _previous_cited_answer(history)
        if previous and previous[1].intent == "procedures":
            return _test_duration_answer(guide, history, query)
        if previous and previous[1].intent in {"visits", "study_overview"}:
            return _timeline_answer(guide, concise=True)
        if previous is None:
            return _ambiguous_end_clarification()
        return _timeline_answer(guide, concise=_concise_timeline())

    if intent == "study_overview":
        return _overview(guide)
    if intent in _SECTION_INTENTS:
        answer = None
        # A named test ("blood glucose") is matched precisely; a general
        # "what tests will I have?" still lists the whole approved section.
        broad_procedures = bool(
            intent == "procedures"
            and _BROAD_PROCEDURE_QUESTION.search(normalize_text(query))
        )
        named_topic = (
            intent == "procedures"
            and not broad_procedures
            and bool(_lookup_tokens(query) - _GENERIC_TOPIC)
        )
        if named_topic:
            answer = _lookup_answer(guide, contextual_query)
            if answer is None:
                return _answer(
                    f"{_MISSING_DETAIL} {_ASK_ABOUT}",
                    intent,
                    limitations=["No approved guide item matched the topic asked."],
                )
        if answer is None:
            answer = _section_answer(guide, intent, contextual_query)
        if not answer.citations and intent == "procedures":
            answer = _lookup_answer(guide, contextual_query) or answer
        # First/next visit already handled above; other visit timing stays concise.
        if intent == "visits" and _TIMING_QUESTION.search(normalize_text(query)):
            if re.search(r"\b(?:first|next|initial|screening)\b", normalize_text(query)):
                return _first_or_next_visit_answer(guide, query)
            return _timeline_answer(guide, concise=True)
        return _with_timing_note(query, answer)

    found = _lookup_answer(guide, contextual_query)
    if found is not None:
        return _with_timing_note(query, found)

    # Broad study questions that still classified as unsupported.
    if re.search(
        r"\bwhat happens\b|\bduring the (?:study|trial)\b|\bin the (?:study|trial)\b",
        normalize_text(query),
    ):
        return _overview(guide)

    return _answer(
        f"{_NOT_SPECIFIED} {_ASK_ABOUT}",
        "unsupported",
        limitations=["Query did not match approved guide content."],
    )


def answer_patient_from_guide(
    query: str,
    guide: Optional[PatientGuide],
    history: Optional[Sequence[dict[str, object]]] = None,
) -> PatientAnswer:
    """
    Conversational answer from approved, published guide items only.

    message -> social/typo/phrasing normalization -> context -> intent ->
    approved evidence (deterministic answer) -> grounding validation ->
    optional natural wording (CTAI_CHAT_LLM) -> repeat handling.
    """
    answer = _answer_core(query, guide, history)
    answer = validate_grounding(answer, guide)
    answer = naturalize(query, answer, history=history)
    return _avoid_repeat(answer, history)


def _avoid_repeat(
    answer: PatientAnswer,
    history: Optional[Sequence[dict[str, object]]],
) -> PatientAnswer:
    """Avoid dumping the same answer twice unless the patient asked to repeat."""
    if not history or not answer.citations:
        return answer
    previous_turn = history[-1]
    previous = previous_turn.get("answer")
    prior_query = str(previous_turn.get("query") or "")
    if not isinstance(previous, PatientAnswer):
        return answer
    if previous.answer_text != answer.answer_text:
        return answer

    asked_to_repeat = bool(
        re.search(
            r"\b(again|repeat|say that again|remind me|one more time)\b",
            normalize_text(prior_query),
        )
    )
    if asked_to_repeat:
        return answer.model_copy(
            update={
                "answer_text": (
                    "Here is that again, in case it helps:\n\n"
                    + answer.answer_text
                )
            }
        )

    # Timeline / end follow-ups: short reminder instead of a full restatement.
    if answer.intent == "visits" and "officially ends" in answer.answer_text.casefold():
        pages = sorted(
            {c.source_page for c in answer.citations if c.source_page is not None}
        )
        day_mention = next(
            (
                c.statement
                for c in answer.citations
                if re.search(r"follow[- ]?up", c.statement, re.I)
            ),
            None,
        )
        if day_mention is None:
            dated = [
                c.statement
                for c in answer.citations
                if re.search(r"day\s+\d+", c.statement, re.I)
            ]
            day_mention = dated[-1] if dated else None
        focus = (
            day_mention.rstrip(".")
            if day_mention
            else "a follow-up visit in the approved schedule"
        )
        short = (
            f"Quick reminder: the approved guide mentions {focus}, but it does "
            "not specify when the study officially ends. Please ask your study "
            "coordinator for the complete timeline."
        )
        if pages:
            label = "pages" if len(pages) > 1 else "page"
            short += f"\n\nSource: Protocol {label} {', '.join(map(str, pages))}."
        return answer.model_copy(update={"answer_text": short})

    return answer
