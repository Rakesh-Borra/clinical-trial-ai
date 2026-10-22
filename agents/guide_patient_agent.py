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
    ),
    "procedures": (
        r"\b(procedure|procedures)\b",
        r"\b(blood (work|draw|draws|glucose|sugar|pressure|tests?|testing|samples?)"
        r"|bloodwork|lab tests?|labs?)\b",
        r"\b(bp|a blood test)\b",
        r"\bwhat (tests|procedures) (are|will|do)\b",
        r"\bstudy (tests?|testing)\b",
    ),
    "safety": (
        r"\b(side effects?|adverse events?)\b",
        r"\b(monitored|monitoring) for\b",
        r"\bis it safe\b",
        r"\bsafety (information|risks?|monitoring)\b",
    ),
    "study_overview": (
        r"\bwhat (is|study) (this|the) (study|trial)\b",
        r"\b(study|trial) (name|title|called)\b",
        r"\bprotocol id\b",
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
    return [item for item in section.items if item.approved]


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
    approved = [item for item in strong if item.approved]
    unapproved = [item for item in strong if not item.approved]
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

    lines = [opener, ""] + [_line(item) for item in approved]
    limitations: List[str] = []
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
        citations=[_citation(item) for item in approved],
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
    "tested": ("test",),
    "tests": ("test",),
    "exam": ("test",),
    "checkup": ("visit",),
}
# Words too generic to identify a topic on their own.
_LOOKUP_STOP = frozenset({"will", "get", "give", "taken", "take", "done"})
_GENERIC_TOPIC = frozenset({"test", "lab"})
_TIMING_QUESTION = re.compile(
    r"\b(when|what time|how often|how long|how many times|schedule)\b"
)
_TIMING_TEXT = re.compile(
    r"\b(day|days|week|weeks|month|months|hour|hours|minutes?|daily|weekly|"
    r"monthly|every|baseline|screening|before|after|timing|at each)\b"
)


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


def _lookup_answer(guide: PatientGuide, query: str) -> Optional[PatientAnswer]:
    """Answer a short or loosely worded question from approved items only."""
    items = _lookup_items(guide)
    if not items:
        return None
    item_tokens = {id(i): _lookup_tokens(_item_blob(i)) for i in items}
    vocabulary = set().union(*item_tokens.values())
    query_tokens = _correct_spelling(_lookup_tokens(query), vocabulary)
    if not query_tokens:
        return None

    scored = [(len(query_tokens & item_tokens[id(i)]), i) for i in items]
    scored = [(n, i) for n, i in scored if n]
    if not scored:
        return None
    best = max(n for n, _ in scored)
    # Several meaningful words must agree, otherwise only a one-word query may match.
    if len(query_tokens) > 1 and best < 2:
        return None
    matches = [i for n, i in scored if n == best]

    if len(query_tokens) == 1:
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

    lines = ["Here is what the study team has approved that relates to your question:", ""]
    lines += [_line(i) for i in matches]
    return _answer(
        "\n".join(lines),
        "procedures" if matches[0].section == "procedures" else "unsupported",
        citations=[_citation(i) for i in matches],
        confidence="high",
    )


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


def _medical_advice_refusal() -> PatientAnswer:
    return _answer(
        "I can't give medical advice, a diagnosis, or treatment "
        "recommendations. I can only share general information the study "
        "team has approved, such as visits, procedures, and the listed "
        "criteria. Please talk with the study team or a qualified clinician "
        "about your health.",
        "medical_advice_refused",
        limitations=["Medical advice and personal clinical decisions are out of scope."],
    )


def _overview(guide: PatientGuide) -> PatientAnswer:
    title = guide.trial_title or "this clinical trial"
    protocol = f" (protocol ID {guide.protocol_id})" if guide.protocol_id else ""
    return _answer(
        f"This guide is for the study “{title}”{protocol}. {_ASK_ABOUT}",
        "study_overview",
        confidence="high",
    )


def answer_patient_from_guide(
    query: str,
    guide: Optional[PatientGuide],
) -> PatientAnswer:
    """Answer a patient question using approved Patient Guide items only."""
    if not query or not query.strip():
        return _answer(
            f"Please ask a question about the study. {_ASK_ABOUT}",
            "unsupported",
            limitations=["Empty query."],
        )

    intent = _classify_intent(query)

    # Refusals never depend on what has been approved.
    if intent == "medical_advice_refused":
        return _medical_advice_refusal()

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

    if intent == "eligibility_decision_refused":
        return _eligibility_refusal(guide)
    if intent == "study_overview":
        return _overview(guide)
    if intent in _SECTION_INTENTS:
        answer = None
        # A named test ("blood glucose") is matched precisely; a general
        # "what tests will I have?" still lists the whole approved section.
        named_topic = bool(_lookup_tokens(query) - _GENERIC_TOPIC)
        if intent == "procedures" and named_topic:
            answer = _lookup_answer(guide, query)
            if answer is None:
                return _answer(
                    "I don't have approved information that answers that "
                    f"question about the tests and procedures. {_ASK_ABOUT}",
                    intent,
                    limitations=["No approved guide item matched the topic asked."],
                )
        if answer is None:
            answer = _section_answer(guide, intent, query)
        if not answer.citations and intent == "procedures":
            answer = _lookup_answer(guide, query) or answer
        return _with_timing_note(query, answer)

    found = _lookup_answer(guide, query)
    if found is not None:
        return _with_timing_note(query, found)

    return _answer(
        "I'm not sure I understood that question. " + _ASK_ABOUT + " I can't "
        "give medical advice or decide whether you personally can join.",
        "unsupported",
        limitations=["Query did not match a supported patient question."],
    )
