"""
Deterministic source-grounding checks for patient-facing wording.

A rewrite or a summary may only say what the protocol says. These helpers find
clinical content (tests, treatment claims, risks, rights, contact details,
sequence and schedule wording) that is absent from the supporting text, and
truncated phrases such as "Type ." They never call a model.

Each term group lists words that mean the same thing for this check, so a
plain-language paraphrase passes when the source uses any word of its group
("side effect" for "adverse event", "sugar" for "glucose"). Specific items (a
named test, symptom or route) are single-word groups and must appear in the
support themselves.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from services.source_verifier import normalize_text

# Prefix of every grounding problem, so the UI can show it as a warning.
UNSUPPORTED_PREFIX = "Unsupported by the protocol: "

_GROUPS: List[tuple[str, tuple[str, ...]]] = [
    ("glucose", (r"blood glucose", r"blood sugar", r"glucose", r"sugar", r"hba1c", r"a1c")),
    ("blood pressure", (r"blood pressure", r"sphygmomanometer", r"mmhg", r"millimet(?:er|re)s?")),
    ("heart tests", (r"ecg", r"ekg", r"electrocardiogram", r"heart rate", r"pulse")),
    ("imaging", (r"mri", r"ct scan", r"x-?ray", r"ultrasound", r"scan", r"imaging")),
    ("biopsy", (r"biops\w*",)),
    ("urine test", (r"urine", r"urinalysis")),
    ("pregnancy test", (r"pregnancy test", r"pregnan\w*")),
    ("body measures", (r"weight", r"height", r"bmi", r"temperature")),
    ("lab values", (r"cholesterol", r"hemoglobin", r"haemoglobin", r"ferritin", r"blood count")),
    ("treatment", (
        r"treatments?", r"therapy", r"therapies", r"medications?", r"medicines?",
        r"drugs?", r"tablets?", r"capsules?", r"pills?", r"doses?", r"dosing",
        r"oral", r"mg", r"iron", r"supplements?", r"regimen", r"study product",
    )),
    ("novelty", (r"new", r"novel", r"experimental", r"innovative", r"investigational")),
    ("effect claims", (
        r"cures?", r"effective", r"effectiveness", r"benefits?", r"improves?",
        r"help(?:s)? manage", r"manage your", r"relieves?", r"reduces?",
        r"prevents?", r"safe",
    )),
    ("test location", (
        r"(?:at|in) (?:the|a|your) (?:clinic|hospital|laboratory|lab|study site|"
        r"research site|doctor'?s office|home)",
        r"at home", r"clinic", r"hospital", r"laborator\w+",
    )),
    ("blood sample", (
        r"blood (?:draws?|drawn|samples?|collection|tests?)", r"venipuncture",
        r"finger-?stick", r"needles?", r"samples?",
    )),
    ("equipment", (r"devices?", r"machines?", r"cuffs?", r"monitors?", r"meters?", r"app")),
    ("injection", (r"injections?", r"infusions?")),
    ("placebo", (r"placebo",)),
    ("general risk", (
        r"risks?", r"side effects?", r"adverse", r"reactions?", r"unwanted",
        r"harm", r"health problems?", r"serious",
    )),
    ("pain", (r"pain\w*", r"discomfort", r"sore\w*")),
    ("bruising", (r"bruis\w*",)),
    ("dizziness", (r"dizz\w*",)),
    ("nausea", (r"nause\w*", r"vomit\w*")),
    ("headache", (r"headaches?",)),
    ("infection", (r"infect\w*",)),
    ("fatigue", (r"fatigue", r"tired\w*")),
    ("allergy", (r"allerg\w*",)),
    ("withdrawal", (r"withdraw\w*", r"leave the (?:study|trial)", r"drop out", r"stop taking part")),
    ("penalty", (r"penalt\w*", r"punish\w*")),
    ("consent", (r"consent\w*",)),
    ("voluntary", (r"voluntar\w*", r"your choice", r"free to")),
    ("privacy", (r"privacy", r"confidential\w*", r"anonymous\w*")),
    ("medical care", (r"medical care", r"your care", r"health ?care", r"regular care")),
    ("contact", (
        r"contact\w*", r"phone", r"telephone", r"e-?mail", r"call", r"hotline",
        r"address", r"research team", r"study team", r"study staff",
        r"study doctor", r"investigators?", r"next steps?",
    )),
    ("sequence", (r"first", r"initial", r"last", r"final", r"second", r"third", r"next")),
    ("frequency", (
        r"daily", r"weekly", r"monthly", r"yearly", r"twice", r"once", r"hourly",
        r"regular(?:ly)?", r"routine(?:ly)?", r"periodic(?:ally)?",
    )),
    ("scheduling", (
        r"scheduled", r"specific (?:times?|days?|visits?)",
        r"at (?:some|several|certain|specific|scheduled|different|various)",
    )),
]

_COMPILED = [
    (label, [re.compile(rf"\b(?:{alias})\b", re.IGNORECASE) for alias in aliases])
    for label, aliases in _GROUPS
]

# Technical-looking words that are not in the support are suspect even when
# they are not in a group above.
_TECHNICAL_WORD = re.compile(
    r"\b[a-z]{15,}\b|\b[a-z]{4,}(?:itis|emia|aemia|osis|ectomy|ology|meter|graphy|pathy|plasty)\b",
    re.IGNORECASE,
)

# Capitalised only: "of this type." is ordinary prose, "Type ." is cut off.
_DANGLING = re.compile(r"\b(Type|Stage|Grade|Class|Phase|Cycle)\s*(?=[.,;:)]|$)")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").casefold()))


def unsupported_terms(text: str, support: str) -> List[str]:
    """Clinical wording in `text` that nothing in `support` backs up."""
    support = support or ""
    found: List[str] = []
    for _label, patterns in _COMPILED:
        hit = next((m for p in patterns if (m := p.search(text))), None)
        if hit and not any(p.search(support) for p in patterns):
            found.append(hit.group(0))
    supported_words = _words(support)
    for match in _TECHNICAL_WORD.finditer(text):
        word = match.group(0)
        if word.casefold() not in supported_words and not any(
            word.casefold() in f.casefold() or f.casefold() in word.casefold()
            for f in found
        ):
            found.append(word)
    return list(dict.fromkeys(found))


def truncated_phrases(text: str) -> List[str]:
    """Phrases cut off before the name or number they need, e.g. "Type ."."""
    return [m.group(0).strip() + "…" for m in _DANGLING.finditer(text or "")]


def statement_problems(rewritten: str, support: str) -> List[str]:
    """Grounding problems of one patient statement against its source passage."""
    problems = []
    terms = unsupported_terms(rewritten, support)
    if terms:
        problems.append(
            UNSUPPORTED_PREFIX
            + "the rewrite adds wording the protocol passage does not contain: "
            + ", ".join(f'"{t}"' for t in terms)
        )
    cut = truncated_phrases(rewritten)
    if cut:
        problems.append(
            UNSUPPORTED_PREFIX
            + "the wording is cut off: " + ", ".join(f'"{c}"' for c in cut)
        )
    return problems


def passage_support(statement: str, page_text: Optional[str]) -> str:
    """Page lines that carry the statement (at least half of its words)."""
    if not page_text:
        return ""
    wanted = {w for w in _words(statement) if len(w) >= 3}
    if not wanted:
        return ""
    lines = [l for l in page_text.splitlines() if l.strip()]
    return "\n".join(
        line for line in lines
        if len(wanted & _words(line)) / len(wanted) >= 0.5
    )


# ---------------------------------------------------------------------------
# Summary sentences
# ---------------------------------------------------------------------------

_FILLER = {
    "this", "that", "these", "those", "with", "from", "will", "your", "you",
    "they", "them", "their", "also", "have", "been", "during", "study",
    "trial", "people", "person", "participants", "participant", "should",
    "would", "could", "about", "which", "what", "when", "where", "while",
    "there", "here", "more", "most", "some", "other", "into", "over", "such",
    "only", "many", "both", "must", "take", "part", "between", "after",
    "before", "through", "without", "within", "than", "then", "does", "done",
    "make", "made", "include", "includes", "following", "being", "were",
    "clinical", "protocol", "joining", "join", "able", "required", "listed",
}


def _stems(text: str) -> set[str]:
    return {
        w[:5] for w in re.findall(r"[a-z]+", (text or "").casefold())
        if len(w) >= 4 and w not in _FILLER
    }


def _sentences(text: str) -> Iterable[str]:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if sentence.strip():
                yield sentence.strip()


def summary_sentence_problems(text: str, pool: str) -> List[str]:
    """Summary sentences that no verified statement supports."""
    pool_stems = _stems(pool)
    problems = []
    for sentence in _sentences(text):
        terms = unsupported_terms(sentence, pool)
        stems = _stems(sentence)
        # Very short sentences have too few content words to compare; the term
        # check above still applies to them.
        weak = len(stems) >= 3 and len(stems & pool_stems) / len(stems) < 0.6
        if terms or weak:
            detail = (
                "adds wording no verified statement contains: "
                + ", ".join(f'"{t}"' for t in terms)
                if terms else "is not supported by any verified statement"
            )
            problems.append(f'The sentence "{sentence}" {detail}.')
    cut = truncated_phrases(text)
    if cut:
        problems.append(
            "Medical wording is cut off: " + ", ".join(f'"{c}"' for c in cut)
        )
    return problems


def headings(text: str) -> List[str]:
    return [
        normalize_text(re.sub(r"^#+\s*", "", line))
        for line in (text or "").splitlines()
        if line.strip().startswith("#")
    ]
