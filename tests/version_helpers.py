"""Small deterministic guides for version-management tests (no model calls)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from models.plain_language_models import GuideItem, GuideSection, PatientGuide

# (plain text, source page, protocol excerpt)
Spec = Tuple[str, int, str]


def _section(title: str, section: str, specs: List[Spec]) -> GuideSection:
    items = [
        GuideItem(
            section=section,
            original_text=excerpt,
            plain_text=text,
            source_page=page,
            approved=True,
            source_status="verified",
            item_id=f"{section}-{index}",
        )
        for index, (text, page, excerpt) in enumerate(specs)
    ]
    return GuideSection(title=title, introduction="", items=items)


def make_guide(content: Dict[str, List[Spec]] | None = None) -> PatientGuide:
    content = content or {}
    return PatientGuide(
        trial_title="Study",
        who_can_join=_section("Who can join", "inclusion_criteria", content.get("inclusion", [])),
        who_cannot_join=_section("Who cannot join", "exclusion_criteria", content.get("exclusion", [])),
        visits=_section("Visits", "visits", content.get("visits", [])),
        procedures=_section("Procedures", "procedures", content.get("procedures", [])),
        safety=_section("Safety", "safety_information", content.get("safety", [])),
        disclaimer="Ask your study team.",
    )
