"""Publication after complete chunked analysis — splits must not force incomplete."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.plain_language_agent import approve_item
from agents.protocol_analysis_agent import (
    MetadataEligibilitySafetyExtraction,
    VisitsProceduresExtraction,
    analyze_protocol_in_chunks,
)
from app import publish_block_reason
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialVisit,
)
from services.guide_review import build_published_snapshot, decision_block_reason
from services.patient_relevance import Block, filter_protocol_blocks
from services.protocol_pipeline import (
    PROTOCOL_CHUNK_CHARS,
    PROTOCOL_CHUNK_OVERLAP_CHARS,
    PROTOCOL_CHUNK_TOKENS,
    PROTOCOL_MAX_CALLS,
    PROTOCOL_MAX_INPUT_CHARS,
)
from services.section_chunker import build_chunks, uncovered_source_blocks
from tests.publish_helpers import exact_rows, generate_verified_guide, make_publishable
from tests.test_patient_guide_ui import _approved_guide, _eight_statement_result


def _page(n: int, text: str) -> dict:
    return {"page_number": n, "text": text, "requires_ocr": False}


def test_intentional_splits_with_full_coverage_are_complete_not_blocking():
    """Oversized sections may split; preserved coverage must not mark incomplete."""
    long = "2. ELIGIBILITY CRITERIA\n" + "\n".join(
        f"Inclusion item {i}: age at least {18 + i % 5} years." for i in range(250)
    )
    pages = [_page(1, "TITLE\nNCT1"), _page(10, long)]
    blocks = filter_protocol_blocks(pages).blocks
    chunks = build_chunks(
        blocks, 400, overlap_chars=PROTOCOL_CHUNK_OVERLAP_CHARS,
    )
    assert any(p.continuation for c in chunks for p in c.pages)
    assert uncovered_source_blocks(blocks, chunks) == []

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        return_value=MetadataEligibilitySafetyExtraction(
            inclusion_criteria=[
                EligibilityCriterion(description="Age 18 or older.", source_page=10),
            ],
        ),
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ), patch(
        "agents.protocol_analysis_agent.should_trigger_eligibility_rescue",
        return_value=False,
    ):
        analysis, warnings, failed, incomplete, _calls = analyze_protocol_in_chunks(
            "\n\n".join(c.text for c in chunks),
            chunk_chars=max(len(c.text) for c in chunks),
            max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
            max_calls=PROTOCOL_MAX_CALLS,
            rescue_max_chars=8000,
            chunks=chunks,
            pass_retries=0,
        )

    assert failed == []
    assert incomplete == []
    assert analysis.analysis_complete is True
    assert any("split across chunk input boundaries" in w for w in warnings)


def test_split_table_with_full_coverage_stays_complete():
    rows = "\n".join(f"Visit {i}|Day {i}|Blood draw {i} mL" for i in range(100))
    table = Block(
        page_number=23,
        heading="23.1",
        depth=2,
        text=f"[TABLE]\nVisit|Day|Activity\n{rows}\n[/TABLE]",
        categories=frozenset({"visits"}),
        category_source="heading",
        is_table=True,
    )
    blocks = list(filter_protocol_blocks([_page(1, "TITLE\nNCT1")]).blocks) + [table]
    chunks = build_chunks(blocks, 300, overlap_chars=0)
    assert uncovered_source_blocks(blocks, chunks) == []
    assert sum(1 for c in chunks if 23 in {p.page_number for p in c.pages}) >= 2

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        return_value=MetadataEligibilitySafetyExtraction(),
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(
            visits=[TrialVisit(visit_name="Visit 1", timing="Day 1", source_page=23)],
        ),
    ), patch(
        "agents.protocol_analysis_agent.should_trigger_eligibility_rescue",
        return_value=False,
    ):
        analysis, _warnings, failed, incomplete, _calls = analyze_protocol_in_chunks(
            "\n\n".join(c.text for c in chunks),
            chunk_chars=max(len(c.text) for c in chunks),
            max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
            max_calls=PROTOCOL_MAX_CALLS,
            rescue_max_chars=8000,
            chunks=chunks,
            pass_retries=0,
        )

    assert failed == [] and incomplete == []
    assert analysis.analysis_complete is True


def test_complete_chunked_analysis_with_approvals_is_publishable(publish_context):
    result = _eight_statement_result()
    result["incomplete_reasons"] = []
    result["incomplete_chunks"] = []
    result["accuracy_warnings"] = [
        "Oversized source pages were split across chunk input boundaries: [10, 11, 17]."
    ]
    guide = _approved_guide(result)
    assert guide.analysis_complete is True
    assert publish_block_reason(guide, result) is None
    snapshot = build_published_snapshot(guide)
    assert snapshot.all_items()
    assert decision_block_reason(guide) is None


def test_failed_chunks_and_missing_evidence_block_publication(publish_context):
    result = _eight_statement_result()
    guide = _approved_guide(result)

    failed = dict(result)
    failed["analysis_complete"] = False
    failed["incomplete_chunks"] = [{"chunk_index": 12, "failed_passes": ["Pass A failed"]}]
    failed["incomplete_reasons"] = [
        "Chunk 12 did not complete both extraction passes."
    ]
    failed["analysis"] = failed["analysis"].model_copy(update={"analysis_complete": False})
    assert "incomplete" in (publish_block_reason(guide, failed) or "").lower()

    missing = dict(result)
    missing["analysis_complete"] = False
    missing["incomplete_reasons"] = [
        "Source block 53 (page 10, heading '') was not packed into any chunk."
    ]
    missing["analysis"] = missing["analysis"].model_copy(
        update={"analysis_complete": False}
    )
    assert publish_block_reason(guide, missing)

    # Verified complete analysis, but a statement without evidence fails closed.
    unverified = dict(result)
    bad_guide = _approved_guide(result)
    item = bad_guide.all_items()[0]
    item.source_status = "unverified"
    item.source_issue = "Not found on the cited page."
    item.approved = True
    item.override = False
    assert decision_block_reason(bad_guide)
    assert publish_block_reason(bad_guide, unverified)


def test_ai_flagged_statements_require_recorded_override(publish_context):
    result = _eight_statement_result()
    guide = _approved_guide(result)
    flagged = guide.all_items()[0]
    flagged.source_status = "unverified"
    flagged.source_issue = "Closest match was on another page."
    flagged.approved = True
    flagged.override = False
    flagged.override_at = None
    assert decision_block_reason(guide)
    assert publish_block_reason(guide, result)

    approve_item(flagged)  # records override without fabricating evidence
    assert flagged.override and flagged.override_at
    assert flagged.source_status == "unverified"
    make_publishable("protocol-eight")
    assert publish_block_reason(guide, result) is None


def test_coverage_gap_still_marks_pipeline_incomplete():
    """Pipeline must keep blocking when a kept block is absent from chunks."""
    from services.protocol_pipeline import analyze_protocol_pdf

    pages = [
        _page(1, "TITLE\nNCT1\n"),
        _page(2, "2. ELIGIBILITY CRITERIA\nAge 18 or older.\n"),
    ]
    analysis = ProtocolAnalysis(
        analysis_complete=True,
        inclusion_criteria=[
            EligibilityCriterion(description="Age 18 or older.", source_page=2),
        ],
        exclusion_criteria=[
            EligibilityCriterion(description="Pregnancy.", source_page=2),
        ],
        visits=[TrialVisit(visit_name="Screening", timing="Day -7", source_page=2)],
        safety_information=[
            SafetyInformation(description="Monitor adverse events.", source_page=2),
        ],
    )
    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value={"pages": pages, "warnings": []},
    ), patch(
        "services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", 10,
    ), patch(
        "services.protocol_pipeline.uncovered_source_blocks",
        return_value=["Source block 1 (page 2, heading '2.') was not packed into any chunk."],
    ), patch(
        "services.protocol_pipeline.analyze_protocol_in_chunks",
        return_value=(analysis, [], [], [], 4),
    ), patch(
        "services.protocol_pipeline.verify_source_pages",
        return_value=exact_rows(analysis),
    ), patch(
        "services.protocol_pipeline.apply_protocol_accuracy_checks",
        return_value=[],
    ):
        result = analyze_protocol_pdf("gap.pdf")

    assert result["analysis_complete"] is False
    assert any("not packed into any chunk" in r for r in result["incomplete_reasons"])
