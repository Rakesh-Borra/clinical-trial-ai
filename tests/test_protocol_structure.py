"""Section classification, mixed pages, tables, completeness audit (no model calls)."""

from unittest.mock import patch

import pymupdf
import pytest

from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from services.completeness_audit import audit_completeness
from services.patient_relevance import (
    filter_protocol_blocks,
    parse_heading,
    render_table,
    split_page_blocks,
)
from services.pdf_processor import extract_pdf
from services.protocol_pipeline import analyze_protocol_pdf
from agents.protocol_analysis_agent import route_passes
from services.section_chunker import build_chunks
from services.section_classifier import classify_heading, classify_text

ADMIN = "The sponsor will archive records and coordinate logistics. " * 12


def _page(number, text, tables=None):
    page = {"page_number": number, "text": text}
    if tables:
        page["tables"] = tables
    return page


SCHEDULE = [{"rows": [
    ["Procedure", "Screening", "Day 1", "Day 28 +/- 3", "Week 12"],
    ["Blood draw (5 mL)", "X", "X", "X", "X"],
    ["ECG", "X", "", "X", ""],
    ["HbA1c (%)", "X", "", "X", "X"],
]}]


# ---- section classification -------------------------------------------------

@pytest.mark.parametrize("heading,category", [
    ("Trial Population", "eligibility"),
    ("Inclusion Criteria", "eligibility"),
    ("5.2 Exclusion Criteria", "eligibility"),
    ("Schedule of Activities", "visits"),
    ("Trial Assessments and Procedures", "procedures"),
    ("Benefit-Risk Assessment", "risks"),
    ("Adverse Events, Serious Adverse Events, Product Complaints, Pregnancy "
     "and Postpartum Information, and Special Safety Situations", "safety"),
    ("Informed Consent Process", "consent"),
    ("Discontinuation of Trial Intervention and Participant Withdrawal "
     "from the Trial", "withdrawal"),
    ("Trial Objectives, Estimands, and Endpoints", "purpose"),
    ("Participant Responsibilities", "responsibilities"),
])
def test_ich_m11_style_headings_are_classified(heading, category):
    assert category in classify_heading(heading)


@pytest.mark.parametrize("heading", [
    "Trial Oversight Considerations", "Dissemination of Clinical Study Information",
    "Sponsor Responsibilities", "Data Management", "Statistical Considerations",
])
def test_administrative_headings_are_administrative_only(heading):
    assert classify_heading(heading) == frozenset({"administrative"})


def test_unrecognized_heading_has_no_category_and_text_cues_are_weak():
    assert classify_heading("Miscellaneous Notes") == frozenset()
    assert "visits" in classify_text("Visit 1 on Day 1 and Visit 2 on Week 4.")
    assert classify_text("Lorem ipsum dolor.") == frozenset()


def test_long_title_case_m11_heading_is_detected_without_numbering():
    title = "Discontinuation of Trial Intervention and Participant Withdrawal from the Trial"
    assert parse_heading(title) is not None
    assert parse_heading("The participant will visit the clinic and give consent.") is None


# ---- mixed pages ------------------------------------------------------------

def _m11_pages():
    return [
        _page(1, "Example Trial\nProtocol NCT01234567"),
        _page(
            2,
            "Trial Population\nAdults aged 18 to 65 with type 2 diabetes.\n\n"
            f"Trial Oversight Considerations\n{ADMIN}\n\n"
            "Informed Consent Process\nParticipants sign consent before any procedure.\n\n"
            f"Dissemination of Clinical Study Information\n{ADMIN}",
        ),
        _page(
            3,
            f"Data Management\n{ADMIN}\n\n{ADMIN}\n\n"
            "Participants who become pregnant must stop treatment and report "
            "serious adverse events at once.\n\n"
            "Participant Withdrawal from the Trial\nYou may withdraw at any time.",
        ),
    ]


def test_mixed_m11_pages_keep_patient_content_and_drop_admin_blocks():
    result = filter_protocol_blocks(_m11_pages())
    kept = " ".join(b.text for b in result.blocks if b.keep)
    assert "aged 18 to 65" in kept
    assert "sign consent before any procedure" in kept
    assert "withdraw at any time" in kept
    assert kept.count("archive records") == 0
    assert {s["heading"] for s in result.skipped} >= {
        "Trial Oversight Considerations",
        "Dissemination of Clinical Study Information",
    }


def test_safety_critical_text_survives_inside_administrative_section():
    result = filter_protocol_blocks(_m11_pages())
    kept = " ".join(b.text for b in result.blocks if b.keep)
    assert "must stop treatment and report serious adverse events" in kept


def test_block_categories_and_page_numbers_are_recorded():
    blocks = filter_protocol_blocks(_m11_pages()).blocks
    population = next(b for b in blocks if b.heading == "Trial Population")
    assert population.page_number == 2
    assert "eligibility" in population.categories
    assert population.category_source == "heading"


# ---- tables -----------------------------------------------------------------

def test_table_blocks_keep_rows_columns_values_and_page():
    pages = [_page(1, "T\nNCT1"), _page(4, "Schedule of Activities\nSee table.", SCHEDULE)]
    blocks = filter_protocol_blocks(pages).blocks
    table = next(b for b in blocks if b.is_table)
    assert table.page_number == 4 and table.keep
    assert "Procedure | Screening | Day 1 | Day 28 +/- 3 | Week 12" in table.text
    assert "ECG | X |  | X | " in table.text
    assert "Blood draw (5 mL) | X | X | X | X" in table.text
    assert "HbA1c (%) | X |  | X | X" in table.text


def test_render_table_preserves_cell_text_exactly():
    assert render_table([["Dose", "10 mg  \n twice daily"], ["eGFR", "< 45 mL/min/1.73 m2"]]) == (
        "Dose | 10 mg twice daily\neGFR | < 45 mL/min/1.73 m2"
    )


def test_oversized_table_splits_between_rows_with_header_repeated():
    rows = [["Procedure", "Day 1", "Day 28"]] + [
        [f"Assessment {i} (dose {i * 5} mg)", "X", "X"] for i in range(60)
    ]
    pages = [_page(1, "T\nN"), _page(6, "Schedule of Activities\nsee below", [{"rows": rows}])]
    chunks = build_chunks(filter_protocol_blocks(pages).blocks, 300)
    table_chunks = [c for c in chunks if "[TABLE]" in c.text]
    assert len(table_chunks) > 1
    seen = set()
    for chunk in table_chunks:
        assert "Procedure | Day 1 | Day 28" in chunk.text
        assert "[SOURCE PAGE 6]" in chunk.text
        assert chunk.has_table
        assert not any(p.continuation for p in chunk.pages)
        seen.update(line for line in chunk.text.splitlines() if line.startswith("Assessment"))
    assert len(seen) == 60
    assert "Assessment 59 (dose 295 mg) | X | X" in seen


def test_real_pdf_table_extraction_keeps_text_and_cell_values(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "The schedule of activities for the study is shown in the "
                     "table below and all visits are required.", fontsize=9)
    rows = [["Procedure", "Screening", "Day 1", "Day 28 +/- 3"],
            ["Blood draw (5 mL)", "X", "X", "X"], ["ECG", "X", "", "X"],
            ["Urine sample", "X", "", "X"], ["Vital signs", "X", "X", "X"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.draw_rect(pymupdf.Rect(50 + c * 100, 100 + r * 22, 150 + c * 100, 122 + r * 22))
            page.insert_text((54 + c * 100, 115 + r * 22), cell, fontsize=8)
    path = tmp_path / "t.pdf"
    doc.save(str(path))
    doc.close()

    result = extract_pdf(str(path))
    entry = result["pages"][0]
    assert entry["text"] == pymupdf.open(str(path))[0].get_text("text", sort=True).strip()
    table = entry["tables"][0]["rows"]
    assert table[0] == ["Procedure", "Screening", "Day 1", "Day 28 +/- 3"]
    assert table[2] == ["ECG", "X", "", "X"]
    assert table[1][0] == "Blood draw (5 mL)"


def test_table_extraction_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("CTAI_EXTRACT_TABLES", "0")
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "A plain page with enough ordinary words in it.")
    path = tmp_path / "p.pdf"
    doc.save(str(path))
    assert "tables" not in extract_pdf(str(path))["pages"][0]


# ---- routing ----------------------------------------------------------------

def test_table_forces_procedure_pass_and_unknown_headings_run_both():
    assert route_passes("x", False, ("eligibility",), True, has_table=True) == (True, True)
    assert route_passes("x", False, ("eligibility",), False) == (True, True)
    assert route_passes("x", False, ("responsibilities",), True) == (True, True)


def test_a_cue_for_the_other_pass_prevents_skipping():
    text = "The visit schedule requires a blood sample on Day 1."
    assert route_passes(text, False, ("eligibility",), True) == (True, True)
    assert route_passes("Adverse events must be reported.", False, ("visits",), True) == (True, True)


def test_pipeline_sends_tables_but_never_modifies_original_pages():
    pages = [
        _page(1, "TITLE\nNCT1"),
        _page(2, "Schedule of Activities\nSee the table.", SCHEDULE),
    ]
    original = [dict(p) for p in pages]
    sent = {}

    def fake(text, **kwargs):
        sent["text"] = text
        return ProtocolAnalysis(analysis_complete=True), [], [], [], 0

    with patch("services.protocol_pipeline.extract_pdf",
               return_value={"pages": pages, "warnings": []}), \
         patch("services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", 10), \
         patch("services.protocol_pipeline.analyze_protocol_in_chunks", side_effect=fake):
        result = analyze_protocol_pdf("t.pdf")
    assert "[SOURCE PAGE 2]" in sent["text"]
    assert "Day 28 +/- 3" in sent["text"] and "[TABLE]" in sent["text"]
    assert pages == original
    assert "completeness_audit" in result


# ---- completeness audit -----------------------------------------------------

def _analysis(**kwargs):
    return ProtocolAnalysis(**kwargs)


def test_audit_reports_missing_and_not_extracted_categories():
    blocks = filter_protocol_blocks([
        _page(1, "T\nN"),
        _page(2, "Trial Population\nAdults 18 to 65.\n\nSchedule of Activities\nVisits on Day 1."),
    ]).blocks
    audit = audit_completeness(blocks, _analysis())
    assert audit["categories"]["eligibility"]["status"] == "not_extracted"
    assert audit["categories"]["visits"]["source_pages"] == [2]
    assert audit["categories"]["procedures"]["status"] == "missing"
    assert "procedures and tests" in audit["missing"]
    assert audit["needs_attention"]


def test_audit_ok_when_extracted_and_flags_unrecognized_structure():
    blocks = filter_protocol_blocks([
        _page(1, "T\nN"),
        _page(2, "Trial Population\nAdults 18 to 65."),
    ]).blocks
    analysis = _analysis(
        inclusion_criteria=[EligibilityCriterion(description="Adults 18 to 65.", source_page=2)],
        visits=[TrialVisit(visit_name="Day 1", source_page=2)],
        procedures=[TrialProcedure(name="Blood draw", source_page=2)],
        safety_information=[SafetyInformation(description="Report events.", source_page=2)],
    )
    audit = audit_completeness(blocks, analysis)
    assert audit["categories"]["eligibility"]["status"] == "ok"
    assert audit["missing"] == []

    unstructured = filter_protocol_blocks([_page(1, "Just some text without headings. " * 20)]).blocks
    notes = audit_completeness(unstructured, _analysis())["notes"]
    assert notes and "no recognized section heading" in notes[0]


def test_audit_marks_heading_less_matches_as_uncertain_classification():
    blocks = filter_protocol_blocks([
        _page(1, "T\nN"),
        _page(2, "The visit on Day 1 and the visit on Week 4 are required."),
    ]).blocks
    audit = audit_completeness(blocks, _analysis())
    assert audit["categories"]["visits"]["classification"].startswith("content cues only")


def test_no_extra_model_calls_beyond_two_passes_per_chunk():
    from agents import protocol_analysis_agent as agent
    from agents.protocol_analysis_agent import analyze_protocol_in_chunks
    chunks = build_chunks(filter_protocol_blocks(_m11_pages() + [
        _page(4, "Schedule of Activities\nsee", SCHEDULE)]).blocks, 400)
    with patch.object(agent, "_pass_a_metadata_eligibility_safety",
                      return_value=agent.MetadataEligibilitySafetyExtraction()) as a, \
         patch.object(agent, "_pass_b_visits_procedures",
                      return_value=agent.VisitsProceduresExtraction()) as b:
        *_, calls = analyze_protocol_in_chunks(
            "x", chunk_chars=3000, max_input_chars=10**6, max_calls=65,
            rescue_max_chars=8000, chunks=chunks, route=True)
    assert a.call_count + b.call_count <= 2 * len(chunks)
    assert calls >= a.call_count + b.call_count


def test_input_character_limit_still_enforced():
    from agents.protocol_analysis_agent import analyze_protocol_in_chunks
    with pytest.raises(ValueError, match="input limit"):
        analyze_protocol_in_chunks(
            "x" * 320_001, chunk_chars=8000, max_input_chars=320_000,
            max_calls=65, rescue_max_chars=8000)
