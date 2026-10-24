"""
Rule-based classification of protocol sections for patient-relevant analysis.

Recognizes common protocol structures, including ICH M11-style headings
("Trial Population", "Schedule of Activities", "Discontinuation of Trial
Intervention and Participant Withdrawal from the Trial", ...) as well as
traditional numbered headings. No model calls and no dependencies.

Categories:
  purpose, eligibility, visits, procedures, risks, safety, consent,
  withdrawal, responsibilities, administrative

Heading classification is the primary signal. Content classification is a
weaker fallback used only to label blocks without a recognized heading (for
example tables); callers must treat it as uncertain.
"""

from __future__ import annotations

import re
from typing import FrozenSet

PATIENT_CATEGORIES = (
    "purpose", "eligibility", "visits", "procedures", "risks", "safety",
    "consent", "withdrawal", "responsibilities",
)
# Categories whose content feeds each extraction pass.
PASS_A_CATEGORIES = frozenset(
    {"purpose", "eligibility", "risks", "safety", "consent", "withdrawal"}
)
PASS_B_CATEGORIES = frozenset({"visits", "procedures"})

_HEADING_RULES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("purpose",
         r"purpose|objectives?|\baims?\b|rationale|background|introduction|"
         r"protocol summary|synopsis|trial summary|study summary|"
         r"trial design|study design|overall design|estimands?|endpoints?|"
         r"hypothes"),
        ("eligibility",
         r"eligib|inclusion|exclusion|trial population|study population|"
         r"selection of (?:participants|subjects|patients)|screen failures?|"
         r"lifestyle considerations|\bcriteria\b|prohibited|restrictions"),
        ("visits",
         r"schedule of (?:activities|assessments|events)|visit|"
         r"time and events|study calendar|follow[- ]?up|"
         r"trial period|study duration|duration of (?:the )?(?:trial|study|participation)"),
        ("procedures",
         r"procedures?|assessments?|trial intervention|study intervention|"
         r"study treatment|investigational (?:product|medicinal)|dosing|"
         r"dose and|concomitant|laboratory|biological samples?|specimen|"
         r"efficacy assessments?|pharmacokinetic|blood"),
        ("risks",
         r"risks?|benefit[- ]risk|benefits?|side effects?|toxicit|"
         r"known and potential|contraindicat"),
        ("safety",
         r"safety|adverse (?:events?|reactions?)|serious adverse|pregnan|"
         r"special safety|reporting of|stopping rules|overdose|"
         r"unanticipated|emergency"),
        ("consent",
         r"informed consent|consent process|assent|participant information|"
         r"data protection|confidentiality|compensation|insurance|injury"),
        ("withdrawal",
         r"withdraw|discontinu|early termination|premature|lost to follow"),
        ("responsibilities",
         r"participant responsibilit|subject responsibilit|"
         r"participant (?:requirements|obligations|compliance)|"
         r"adherence|compliance|diary|diaries|what (?:you|participants) (?:need|must)"),
        ("administrative",
         r"sponsor|monitoring plan|data management|data handling|statistic|"
         r"sample size|record[- ]?keeping|archiv|audit|inspection|regulatory|"
         r"publication|dissemination|intellectual|financial|conflict of interest|"
         r"quality (?:assurance|control)|references|bibliography|abbreviations|"
         r"table of contents|signature|document history|version history|"
         r"supply|accountability|logistics|staff|training|delegation|"
         r"oversight|committee|site (?:selection|initiation|closure|personnel)|"
         r"investigator.s? brochure|appendix"),
    )
)

_TEXT_CUES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("eligibility",
         r"\binclusion criteria|\bexclusion criteria|\beligib\w+|"
         r"\bmust (?:be|have|not)\b|\bage[sd]? \d+|years? (?:old|of age)|"
         r"\begfr\b|\bhba1c\b"),
        ("visits",
         r"\bvisit\b|\bday\s*-?\d+|\bweek\s*-?\d+|\bmonth\s*\d+|"
         r"\bscreening\b|\bbaseline\b|end of (?:study|treatment)|\bfollow[- ]?up\b"),
        ("procedures",
         r"\bblood\b|\burine\b|\bsample\b|\becg\b|\bphysical exam|"
         r"\bvital signs?|\bquestionnaire|\bscan\b|\bx-?ray\b|\bbiops|"
         r"\binjection|\binfusion|\blaborator"),
        ("safety",
         r"\badverse (?:event|reaction)|\bserious adverse|\bpregnan|"
         r"\bside effects?|\bhypoglyc|\bdiscontinu|\bstop(?:ped|ping)? "
         r"(?:the )?(?:study|treatment)"),
        ("withdrawal", r"\bwithdraw"),
        ("consent", r"\binformed consent"),
    )
)


def classify_heading(heading: str) -> FrozenSet[str]:
    """Return the categories a heading belongs to (empty when unrecognized)."""
    if not heading:
        return frozenset()
    # Strip leading numbering such as "6.1.2".
    title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", heading.strip())
    found = {name for name, rule in _HEADING_RULES if rule.search(title)}
    patient = found - {"administrative"}
    # A heading naming both patient content and administration is kept as
    # patient content: it is never dropped on the strength of the admin word.
    return frozenset(patient) if patient else frozenset(found)


def classify_text(text: str, *, minimum_hits: int = 2) -> FrozenSet[str]:
    """Content-based categories; uncertain, used only without a heading."""
    found = set()
    for name, cue in _TEXT_CUES:
        if len(cue.findall(text)) >= minimum_hits or (
            name in {"withdrawal", "consent"} and cue.search(text)
        ):
            found.add(name)
    return frozenset(found)


def is_administrative_only(categories: FrozenSet[str]) -> bool:
    return categories == frozenset({"administrative"})
