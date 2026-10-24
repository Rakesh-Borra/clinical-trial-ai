"""Chunking, section filtering, call routing and cache tests (no model calls)."""

from unittest.mock import patch

import pytest

from agents import protocol_analysis_agent as agent
from agents.protocol_analysis_agent import (
    MetadataEligibilitySafetyExtraction,
    VisitsProceduresExtraction,
    analyze_protocol_in_chunks,
    route_passes,
)
from models.protocol_models import EligibilityCriterion, ProtocolAnalysis
from services import analysis_cache
from services.patient_relevance import (
    filter_protocol_blocks,
    filter_protocol_pages,
    split_page_blocks,
)
from services.protocol_chunker import chunk_protocol
from services.protocol_pipeline import analyze_protocol_pdf
from services.section_chunker import build_chunks, estimate_tokens

ADMIN = "The sponsor will archive records and manage site logistics. " * 8
PATIENT = "Participants will attend a screening visit and give a blood sample. " * 6


def _page(number, body):
    return {"page_number": number, "text": body}


def _fixture_pages():
    pages = [_page(1, "STUDY TITLE\nProtocol NCT00000001\nA test study.")]
    for n in range(2, 8):
        pages.append(_page(
            n,
            f"{n}.1 STUDY VISITS\n{PATIENT}\n\n"
            f"{n}.2 SPONSOR RESPONSIBILITIES\n{ADMIN}\n\n"
            f"{n}.3 DATA MANAGEMENT\n{ADMIN}",
        ))
    return pages


def test_blocks_split_at_headings_and_keep_every_character():
    text = "1. ELIGIBILITY\nAge 18+.\n\n2. SPONSOR RESPONSIBILITIES\nArchive files."
    blocks = split_page_blocks(3, text)
    assert [b.heading for b in blocks] == ["ELIGIBILITY", "SPONSOR RESPONSIBILITIES"]
    assert "\n".join(b.text for b in blocks).replace("\n\n", "\n") .count("Age 18+.") == 1
    assert all(b.page_number == 3 for b in blocks)


def test_administrative_blocks_skipped_patient_blocks_kept():
    result = filter_protocol_blocks(_fixture_pages())
    kept = " ".join(b.text for b in result.blocks if b.keep)
    assert "screening visit" in kept
    assert "archive records" not in kept
    assert result.summary()["skipped_blocks"]


def test_safety_critical_text_kept_inside_administrative_section():
    pages = [
        _page(1, "TITLE\nNCT1"),
        _page(2, "5. SPONSOR RESPONSIBILITIES\nThe sponsor archives files. "
                 "Participants who become pregnant must stop treatment and "
                 "report serious adverse events immediately."),
    ]
    result = filter_protocol_blocks(pages)
    assert all(b.keep for b in result.blocks)


def test_first_page_and_empty_pages_are_kept():
    pages = [_page(1, "SPONSOR RESPONSIBILITIES\nlogistics"), _page(2, "")]
    result = filter_protocol_blocks(pages)
    assert all(b.keep for b in result.blocks)
    assert [b.page_number for b in result.blocks] == [1, 2]


def test_filter_never_empties_the_document():
    pages = [_page(1, "TITLE\nx"), _page(2, "9. SPONSOR RESPONSIBILITIES\nonly admin")]
    result = filter_protocol_blocks(pages)
    assert any(b.keep for b in result.blocks)


def test_chunks_respect_token_limit_and_page_numbers():
    result = filter_protocol_blocks(_fixture_pages())
    chunks = build_chunks(result.blocks, 600)
    assert chunks
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 600 + 40
        for page in chunk.pages:
            assert f"[SOURCE PAGE {page.page_number}]" in chunk.text
    pages_seen = {p.page_number for c in chunks for p in c.pages}
    assert pages_seen == set(range(1, 8))


def test_sections_are_not_split_unless_oversized():
    blocks = filter_protocol_blocks([
        _page(1, "TITLE\nNCT1"),
        _page(2, "2. ELIGIBILITY CRITERIA\nInclusion: age 18 to 65.\nExclusion: pregnancy."),
    ]).blocks
    chunks = build_chunks(blocks, 500)
    assert len(chunks) == 1
    assert "Inclusion: age 18 to 65." in chunks[0].text
    assert not any(p.continuation for p in chunks[0].pages)


def test_oversized_block_split_keeps_page_and_marks_continuation():
    body = "3. VISITS\n" + "\n".join(f"Visit {i}: blood draw of 5 mL." for i in range(80))
    blocks = filter_protocol_blocks([_page(1, "T"), _page(4, body)]).blocks
    chunks = build_chunks(blocks, 200)
    page4 = [p for c in chunks for p in c.pages if p.page_number == 4]
    assert len(page4) > 1
    assert any(p.continuation for p in page4)
    joined = "".join(p.text for p in page4)
    assert "Visit 79: blood draw of 5 mL." in joined


def test_before_and_after_chunk_and_call_counts_on_fixture():
    pages = _fixture_pages()
    old_text = "\n\n".join(
        f"[SOURCE PAGE {p['page_number']}]\n{p['text']}"
        for p in filter_protocol_pages(pages).kept_pages
    )
    old_chunks = chunk_protocol(old_text, max_chars=3000).chunks
    new_chunks = build_chunks(
        filter_protocol_blocks(pages).blocks, int(3000 / 3.2)
    )
    old_calls = 2 * len(old_chunks)

    calls = []
    with patch.object(agent, "_pass_a_metadata_eligibility_safety",
                      side_effect=lambda t: calls.append("A") or
                      MetadataEligibilitySafetyExtraction()), \
         patch.object(agent, "_pass_b_visits_procedures",
                      side_effect=lambda t: calls.append("B") or
                      VisitsProceduresExtraction()):
        analyze_protocol_in_chunks(
            "\n\n".join(c.text for c in new_chunks),
            chunk_chars=3000, max_input_chars=10**6, max_calls=65,
            rescue_max_chars=8000, chunks=new_chunks, route=True,
        )
    print(f"before: {len(old_chunks)} chunks / {old_calls} calls; "
          f"after: {len(new_chunks)} chunks / {len(calls)} calls")
    assert len(new_chunks) < len(old_chunks)
    assert len(calls) < old_calls


def test_route_passes_never_skips_on_keywords_alone():
    # Without heading-based categories, both passes always run.
    assert route_passes("Visit schedule: day 1, week 4, blood sample.", False) == (True, True)
    assert route_passes("Inclusion criteria: age 18.", False) == (True, True)
    assert route_passes("Lorem ipsum.", False) == (True, True)
    # Heading categories and the absence of cues for the skipped pass agree.
    assert route_passes("Lorem ipsum.", False, ("visits",), True) == (False, True)
    assert route_passes("Lorem ipsum.", False, ("eligibility",), True) == (True, False)


def test_time_budget_stops_new_calls_with_actionable_message():
    chunks = build_chunks(filter_protocol_blocks(_fixture_pages()).blocks, 600)
    with patch.object(agent, "_pass_a_metadata_eligibility_safety") as a, \
         patch.object(agent, "_pass_b_visits_procedures") as b:
        _, warnings, failed, reasons, calls = analyze_protocol_in_chunks(
            "x", chunk_chars=3000, max_input_chars=10**6, max_calls=65,
            rescue_max_chars=8000, chunks=chunks, route=True,
            max_seconds=1e-9,
        )
    assert calls == 0 and not a.called and not b.called
    assert failed and any("CTAI_ANALYSIS_MAX_SECONDS" in r for r in reasons + warnings)


def test_call_cap_error_is_actionable():
    chunks = build_chunks(filter_protocol_blocks(_fixture_pages()).blocks, 100)
    with pytest.raises(ValueError, match="CTAI_PROTOCOL_MAX_CALLS"):
        analyze_protocol_in_chunks(
            "x", chunk_chars=3000, max_input_chars=10**6, max_calls=5,
            rescue_max_chars=8000, chunks=chunks, route=True,
        )


def test_extraction_disables_thinking_and_caps_output():
    with patch.object(agent, "chat_completion", return_value="{}") as chat:
        agent._chat_structured("p", VisitsProceduresExtraction)
    kwargs = chat.call_args.kwargs
    assert kwargs["think"] is False
    assert kwargs["options"]["num_predict"] == agent.NUM_PREDICT
    assert kwargs["options"]["num_ctx"] == agent.NUM_CTX


# ---- cache -----------------------------------------------------------------

def _analysis():
    return ProtocolAnalysis(
        inclusion_criteria=[EligibilityCriterion(description="Age 18+.", source_page=2)]
    )


def test_cache_roundtrip_and_key_invalidation():
    settings = {"model": "m", "think": False, "version": "2"}
    key = analysis_cache.make_key("text", settings)
    assert analysis_cache.load(key) is None
    analysis_cache.store(key, _analysis(), ["w"])
    hit = analysis_cache.load(key)
    assert hit["analysis"].inclusion_criteria[0].source_page == 2
    assert hit["warnings"] == ["w"]
    assert analysis_cache.make_key("text2", settings) != key
    for change in ({"model": "m2"}, {"think": True}, {"version": "3"}):
        assert analysis_cache.make_key("text", {**settings, **change}) != key


def test_pipeline_reuses_cache_but_not_for_incomplete_results():
    pages = [_page(1, "TITLE\nNCT1\n" + PATIENT), _page(2, "2. ELIGIBILITY\nAge 18+.")]
    complete = ProtocolAnalysis(
        inclusion_criteria=[EligibilityCriterion(description="Age 18+.", source_page=2)],
        analysis_complete=True,
    )
    with patch("services.protocol_pipeline.extract_pdf",
               return_value={"pages": pages, "warnings": []}), \
         patch("services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", 10), \
         patch("services.protocol_pipeline.analyze_protocol_in_chunks",
               return_value=(complete, [], [], [], 3)) as run:
        first = analyze_protocol_pdf("a.pdf")
        second = analyze_protocol_pdf("a.pdf")
        assert run.call_count == 1
        assert second["llm_call_count"] == 0
        assert first["analysis"].inclusion_criteria[0].source_page == 2
        assert second["analysis"].inclusion_criteria[0].source_page == 2
        assert "chunks_analyzed" in second["performance"]
        assert "verification" in second["performance"]["stage_seconds"]

    with patch("services.protocol_pipeline.extract_pdf",
               return_value={"pages": pages + [_page(3, "3. VISITS\nDay 1.")], "warnings": []}), \
         patch("services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", 10), \
         patch("services.protocol_pipeline.analyze_protocol_in_chunks",
               return_value=(complete, [], [{"chunk_index": 1}], ["x"], 3)) as run:
        analyze_protocol_pdf("b.pdf")
        analyze_protocol_pdf("b.pdf")
        assert run.call_count == 2


# ---- mixed-content pages (regression: no whole-page drops) -------------------

def _mixed_pages():
    admin = "The sponsor will archive records and manage site logistics. " * 30
    return [
        _page(1, "STUDY TITLE\nProtocol NCT00000002"),
        _page(
            2,
            f"7. SPONSOR RESPONSIBILITIES\n{admin}\n\n"
            f"8. DATA MANAGEMENT\n{admin}\n\n"
            "9. ELIGIBILITY CRITERIA\nParticipants must be 18 or older.\n\n"
            f"10. RECORD KEEPING\n{admin}",
        ),
        _page(3, f"11. SPONSOR RESPONSIBILITIES\n{admin}\n"
                 "Participants who become pregnant must stop treatment and "
                 "report serious adverse events."),
    ]


def _run_pipeline(pages, long_path):
    captured = {}
    analysis = ProtocolAnalysis(analysis_complete=True)

    def fake_chunked(text, **kwargs):
        captured["text"] = text
        return analysis, [], [], [], 0

    def fake_short(text):
        captured["text"] = text
        return analysis, []

    threshold = 10 if long_path else 10**9
    with patch("services.protocol_pipeline.extract_pdf",
               return_value={"pages": pages, "warnings": []}), \
         patch("services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", threshold), \
         patch("services.protocol_pipeline.analyze_protocol_in_chunks",
               side_effect=fake_chunked), \
         patch("services.protocol_pipeline.analyze_protocol_with_rescue",
               side_effect=fake_short):
        result = analyze_protocol_pdf("mixed.pdf")
    return captured["text"], result


@pytest.mark.parametrize("long_path", [True, False])
def test_mixed_page_keeps_patient_and_safety_text_and_drops_admin(long_path):
    text, result = _run_pipeline(_mixed_pages(), long_path)
    assert "Participants must be 18 or older." in text
    assert "must stop treatment and report serious adverse events" in text
    assert "[SOURCE PAGE 2]" in text and "[SOURCE PAGE 3]" in text
    assert text.count("archive records") < 30 * 3
    assert result["relevance"]["skipped_pages"] == []
    assert result["relevance"]["blocks"]["skipped_blocks"]


def test_page_with_mostly_admin_content_is_not_dropped_whole():
    pages = _mixed_pages()
    text, _ = _run_pipeline(pages, True)
    assert "Participants must be 18 or older." in text


def test_pure_admin_page_is_reported_skipped_but_pdf_pages_untouched():
    admin = "The sponsor will archive records and manage logistics. " * 10
    pages = [
        _page(1, "TITLE\nNCT3"),
        _page(2, f"5. SPONSOR RESPONSIBILITIES\n{admin}"),
        _page(3, "6. ELIGIBILITY\nAge 18+."),
    ]
    original = [dict(p) for p in pages]
    text, result = _run_pipeline(pages, True)
    assert [s["page"] for s in result["relevance"]["skipped_pages"]] == [2]
    assert "[SOURCE PAGE 2]" not in text
    assert "[SOURCE PAGE 3]\n6. ELIGIBILITY" in text
    assert pages == original
