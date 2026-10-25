"""Offline regression tests for large clinical-trial protocol chunking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.protocol_analysis_agent import (
    MetadataEligibilitySafetyExtraction,
    VisitsProceduresExtraction,
    _merge_chunk_analyses,
    analyze_protocol_in_chunks,
)
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from services.patient_relevance import Block, filter_protocol_blocks
from services.protocol_chunker import ProtocolChunk, ProtocolPage
from services.protocol_pipeline import (
    PROTOCOL_CHUNK_CHARS,
    PROTOCOL_CHUNK_OVERLAP_CHARS,
    PROTOCOL_CHUNK_TOKENS,
    PROTOCOL_MAX_CALLS,
    PROTOCOL_MAX_INPUT_CHARS,
    analyze_protocol_pdf,
)
from services.section_chunker import (
    CHARS_PER_TOKEN,
    build_chunks,
    estimate_tokens,
    uncovered_source_blocks,
)


def _patient_page(page_number: int, body: str) -> dict:
    return {
        "page_number": page_number,
        "text": body,
        "requires_ocr": False,
    }


def _synthetic_protocol_pages(page_count: int, chars_per_page: int = 2800):
    """Build a multi-page protocol with patient-relevant sections."""
    pages = [
        _patient_page(
            1,
            "PROTOCOL TITLE: Large Synthetic Study\n"
            "Protocol ID: SYN-LARGE-001\nNCT00000001\n",
        )
    ]
    filler = "Clinical observation recorded for participant monitoring. "
    for page in range(2, page_count + 1):
        if page % 17 == 0:
            section = (
                f"{page}. SCHEDULE OF ASSESSMENTS\n"
                f"Visit Day {page}: blood draw of 5 mL and vital signs.\n"
            )
        elif page % 11 == 0:
            section = (
                f"{page}. SAFETY INFORMATION\n"
                f"Risk on page {page}: monitor liver enzymes weekly.\n"
            )
        elif page % 7 == 0:
            section = (
                f"{page}. EXCLUSION CRITERIA\n"
                f"Exclude if prior therapy within {page} days.\n"
            )
        elif page % 5 == 0:
            section = (
                f"{page}. INCLUSION CRITERIA\n"
                f"Age must be at least {18 + (page % 5)} years on page {page}.\n"
            )
        else:
            section = (
                f"{page}. STUDY PROCEDURES\n"
                f"Procedure P{page}: assessment described on page {page}.\n"
            )
        pad = filler * max(1, (chars_per_page - len(section)) // len(filler))
        pages.append(_patient_page(page, section + pad))
    return pages


def test_377k_character_input_is_accepted_and_batched_offline():
    """A 123-page extract (~377,769 chars) must not hit the old 320k reject."""
    target = 377_769
    page_body = "Inclusion criteria text for large protocol batching. "
    pages = []
    built = 0
    page_number = 1
    while built < target:
        chunk = page_body * 40
        pages.append(_patient_page(
            page_number,
            f"{page_number}. INCLUSION CRITERIA\n{chunk}",
        ))
        built += len(pages[-1]["text"]) + len(f"[SOURCE PAGE {page_number}]\n") + 2
        page_number += 1

    blocks = filter_protocol_blocks(pages).blocks
    chunks = build_chunks(
        blocks, PROTOCOL_CHUNK_TOKENS, overlap_chars=PROTOCOL_CHUNK_OVERLAP_CHARS,
    )
    chunk_text = "\n\n".join(chunk.text for chunk in chunks)
    assert len(chunk_text) >= 320_000
    assert len(chunk_text) <= PROTOCOL_MAX_INPUT_CHARS
    assert uncovered_source_blocks(blocks, chunks) == []
    assert all(len(chunk.text) <= PROTOCOL_CHUNK_CHARS for chunk in chunks)
    assert all(estimate_tokens(chunk.text) <= PROTOCOL_CHUNK_TOKENS + 80 for chunk in chunks)

    worst_case_calls = len(chunks) * 2 + 1
    assert worst_case_calls <= PROTOCOL_MAX_CALLS
    # ~5k-char batches for a ~378k extract → roughly 70–90 batches.
    assert 50 <= len(chunks) <= 120

    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        return_value=MetadataEligibilitySafetyExtraction(),
    ) as pass_a, patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ) as pass_b, patch(
        "agents.protocol_analysis_agent.should_trigger_eligibility_rescue",
        return_value=False,
    ):
        analysis, _warnings, failed, incomplete, calls = analyze_protocol_in_chunks(
            chunk_text,
            chunk_chars=PROTOCOL_CHUNK_CHARS,
            max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
            max_calls=PROTOCOL_MAX_CALLS,
            rescue_max_chars=8000,
            chunks=chunks,
            route=True,
            pass_retries=0,
        )

    assert failed == []
    assert incomplete == []
    assert analysis.analysis_complete is True
    assert pass_a.call_count + pass_b.call_count == calls
    assert calls <= worst_case_calls


@pytest.mark.parametrize("page_count", [123, 150, 180])
def test_synthetic_large_protocols_cover_all_pages_and_fit_call_budget(page_count):
    pages = _synthetic_protocol_pages(page_count)
    blocks = filter_protocol_blocks(pages).blocks
    chunks = build_chunks(
        blocks, PROTOCOL_CHUNK_TOKENS, overlap_chars=PROTOCOL_CHUNK_OVERLAP_CHARS,
    )
    assert uncovered_source_blocks(blocks, chunks) == []

    kept_pages = {b.page_number for b in blocks if b.keep}
    chunk_pages = {
        page.page_number
        for chunk in chunks
        for page in chunk.pages
        if page.page_number is not None
    }
    assert kept_pages <= chunk_pages
    assert all(chunk.source_block_ids for chunk in chunks)
    assert all(len(chunk.text) <= PROTOCOL_CHUNK_CHARS for chunk in chunks)

    chunk_text = "\n\n".join(chunk.text for chunk in chunks)
    assert len(chunk_text) <= PROTOCOL_MAX_INPUT_CHARS
    assert len(chunks) * 2 + 1 <= PROTOCOL_MAX_CALLS

    for chunk in chunks:
        for page in chunk.pages:
            if page.page_number is None:
                continue
            assert f"[SOURCE PAGE {page.page_number}]" in chunk.text


def test_oversized_section_and_table_split_preserve_page_and_overlap():
    long_section = "2. ELIGIBILITY CRITERIA\n" + "\n".join(
        f"Inclusion item {i}: HbA1c must be at least 7.{i % 10}%."
        for i in range(400)
    )
    table_rows = "\n".join(
        f"Visit {i}|Day {i}|Blood draw {i} mL" for i in range(120)
    )
    table_block = Block(
        page_number=9,
        heading="Schedule of Activities",
        depth=1,
        text=f"[TABLE]\nVisit|Day|Activity\n{table_rows}\n[/TABLE]",
        categories=frozenset({"visits"}),
        category_source="heading",
        is_table=True,
    )
    pages = [
        _patient_page(1, "TITLE\nNCT1"),
        _patient_page(4, long_section),
    ]
    blocks = list(filter_protocol_blocks(pages).blocks) + [table_block]
    chunks = build_chunks(blocks, 400, overlap_chars=80)
    assert uncovered_source_blocks(blocks, chunks) == []

    page4 = [p for c in chunks for p in c.pages if p.page_number == 4]
    assert len(page4) > 1
    assert any(p.continuation for p in page4)
    joined = "".join(p.text for p in page4)
    assert "Inclusion item 0:" in joined
    assert "Inclusion item 399:" in joined

    table_pages = [p for c in chunks for p in c.pages if p.page_number == 9]
    assert table_pages
    assert any("[TABLE]" in p.text and "Visit|" in p.text for p in table_pages)


def test_duplicate_and_conflicting_findings_merge_behavior():
    first = ProtocolAnalysis(
        trial_title="Study A",
        inclusion_criteria=[
            EligibilityCriterion(description="Age >= 18 years.", source_page=2),
        ],
        procedures=[
            TrialProcedure(
                name="Blood draw",
                description="Collect 5 mL.",
                source_page=4,
            ),
        ],
        visits=[
            TrialVisit(visit_name="Screening", timing="Day -14", source_page=5),
        ],
        safety_information=[
            SafetyInformation(description="Monitor liver enzymes.", source_page=8),
        ],
    )
    second = ProtocolAnalysis(
        trial_title="Study B",
        inclusion_criteria=[
            EligibilityCriterion(description="Age >= 18 years.", source_page=2),
            EligibilityCriterion(description="BMI between 18 and 40.", source_page=3),
        ],
        procedures=[
            TrialProcedure(
                name="Blood draw",
                description="Collect 10 mL.",
                source_page=4,
            ),
        ],
        visits=[
            TrialVisit(visit_name="Screening", timing="Day -7", source_page=5),
        ],
        safety_information=[
            SafetyInformation(description="Monitor liver enzymes.", source_page=8),
        ],
    )
    merged, warnings = _merge_chunk_analyses([first, second])
    assert merged.trial_title == "Study A"
    assert any("Conflicting trial_title" in w for w in warnings)
    assert len(merged.inclusion_criteria) == 2
    assert len(merged.procedures) == 2
    assert any("Conflicting procedure details" in w for w in warnings)
    assert len(merged.visits) == 2
    assert any("Conflicting visit schedule details" in w for w in warnings)
    assert len(merged.safety_information) == 1


def test_failed_batch_marks_incomplete_and_blocks_publication_flag():
    chunks = [
        ProtocolChunk(
            index=0,
            pages=(ProtocolPage(1, "INCLUSION CRITERIA\nAge 18.", ("eligibility",)),),
            text="[SOURCE PAGE 1]\nINCLUSION CRITERIA\nAge 18.",
            sections=("eligibility",),
            source_block_ids=(0,),
        ),
        ProtocolChunk(
            index=1,
            pages=(ProtocolPage(2, "SAFETY\nRisk of rash.", ("safety",)),),
            text="[SOURCE PAGE 2]\nSAFETY\nRisk of rash.",
            sections=("safety",),
            source_block_ids=(1,),
        ),
    ]
    with patch(
        "agents.protocol_analysis_agent._pass_a_metadata_eligibility_safety",
        side_effect=[
            MetadataEligibilitySafetyExtraction(
                inclusion_criteria=[
                    EligibilityCriterion(description="Age 18.", source_page=1),
                ],
            ),
            RuntimeError("batch boom"),
            RuntimeError("batch boom"),
        ],
    ), patch(
        "agents.protocol_analysis_agent._pass_b_visits_procedures",
        return_value=VisitsProceduresExtraction(),
    ):
        analysis, _warnings, failed, incomplete, _calls = analyze_protocol_in_chunks(
            "\n\n".join(c.text for c in chunks),
            chunk_chars=5000,
            max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
            max_calls=PROTOCOL_MAX_CALLS,
            rescue_max_chars=8000,
            chunks=chunks,
            pass_retries=1,
        )

    assert failed
    assert any("did not complete both extraction passes" in r for r in incomplete)
    assert analysis.analysis_complete is False


def test_small_pdf_still_uses_short_path():
    short_pages = [
        _patient_page(1, "Short protocol title.\nInclusion: age 18."),
    ]
    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value={"pages": short_pages, "warnings": []},
    ), patch(
        "services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS",
        50_000,
    ), patch(
        "services.protocol_pipeline.analyze_protocol_with_rescue",
        return_value=(ProtocolAnalysis(analysis_complete=True), []),
    ) as short_path, patch(
        "services.protocol_pipeline.analyze_protocol_in_chunks",
    ) as long_path:
        result = analyze_protocol_pdf("small.pdf")

    short_path.assert_called_once()
    long_path.assert_not_called()
    assert result["analysis_complete"] is True


def test_per_batch_limit_rejects_oversized_prebuilt_chunk():
    huge = ProtocolChunk(
        index=0,
        pages=(ProtocolPage(1, "x" * 6000, ()),),
        text="[SOURCE PAGE 1]\n" + "x" * 6000,
        sections=(),
        source_block_ids=(0,),
    )
    with pytest.raises(ValueError, match="per-batch character limit"):
        analyze_protocol_in_chunks(
            huge.text,
            chunk_chars=5000,
            max_input_chars=PROTOCOL_MAX_INPUT_CHARS,
            max_calls=10,
            rescue_max_chars=8000,
            chunks=[huge],
        )


def test_default_chunk_budget_is_four_to_six_thousand_characters():
    assert 4000 <= int(PROTOCOL_CHUNK_TOKENS * CHARS_PER_TOKEN) <= 6000
    assert 4000 <= PROTOCOL_CHUNK_CHARS <= 6000
    assert PROTOCOL_MAX_INPUT_CHARS >= 377_769
    assert PROTOCOL_MAX_CALLS >= 200
