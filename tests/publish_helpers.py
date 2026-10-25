"""Test helper: satisfy the publication requirements added for coverage and summary."""

from agents.plain_language_agent import generate_patient_guide
from services import review_store

# Claim-free on purpose: a summary may only say what its statements say, and
# fixtures differ in which statements they keep. The words are the cues the
# critical-information check looks for.
GENERIC_SUMMARY = "Who may join. Visit schedule. Safety and adverse events."


def make_publishable(protocol_id: str) -> None:
    """Record N/A for topics test fixtures lack, and approve a grounded summary."""
    for key in ("eligibility", "visits", "procedures", "interventions", "risks", "consent"):
        review_store.set_coverage_override(
            protocol_id, key, "Fixture protocol has no such section."
        )
    review_store.save_original_summary(protocol_id, GENERIC_SUMMARY, "llm")
    review_store.approve_summary(protocol_id)


_FIELDS = {
    "inclusion_criteria": "description",
    "exclusion_criteria": "description",
    "procedures": "name",
    "visits": "visit_name",
    "safety_information": "description",
    "patient_information": "description",
}


def exact_rows(analysis):
    """Source-verification rows saying every extracted item matched its page."""
    return [
        {
            "category": category,
            "statement": getattr(item, field),
            "source_page": item.source_page,
            "status": "exact_match",
        }
        for category, field in _FIELDS.items()
        for item in getattr(analysis, category)
    ]


def generate_verified_guide(analysis, rewriter=None, **kwargs):
    """generate_patient_guide for fixtures whose facts all passed source checks."""
    return generate_patient_guide(
        analysis, rewriter=rewriter, verification=exact_rows(analysis), **kwargs
    )


def gk(at, section_index: int, item_index: int, suffix: str) -> str:
    """Widget key of a statement, which is built from its stable id."""
    guide = at.session_state["patient_guide"]
    item = guide.sections()[section_index].items[item_index]
    return f"guide_{item.item_id}_{suffix}"
