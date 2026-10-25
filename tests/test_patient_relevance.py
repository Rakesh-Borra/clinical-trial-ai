"""Patient-relevance filtering: conservative, safety-preserving, auditable."""

from services import patient_relevance
from services.patient_relevance import classify_pages, filter_protocol_pages

TITLE = "Protocol XYZ-1: A Study of Example Drug\nProtocol ID: XYZ-1"

ADMIN = (
    "7. Monitoring\n"
    "The sponsor will conduct monitoring visits. Source documents and case "
    "report forms are checked. Data management follows the data entry plan. "
    "Record retention and audits are handled by site staff training."
)
ELIGIBILITY = "Inclusion Criteria\nAdults aged 18 to 65.\nExclusion Criteria\nPregnancy."
ELIGIBILITY_CONTINUED = "Adults must not take drug Y.\nNo kidney disease."
HIDDEN_SAFETY = (
    "Sponsor responsibilities\nThe sponsor handles monitoring visits, source "
    "documents and case report forms. Any serious adverse event must be "
    "reported within 24 hours and participants may be hospitalized."
)
SCHEDULE = "Visit schedule\nScreening at week 0, blood draw at weeks 4 and 8."


def _pages(*texts):
    return [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)]


def _kept(result):
    return [p["page_number"] for p in result.kept_pages]


def test_administrative_page_is_skipped_and_reported():
    result = filter_protocol_pages(_pages(TITLE, ELIGIBILITY, ADMIN, SCHEDULE))
    assert _kept(result) == [1, 2, 4]
    assert result.skipped[0]["page"] == 3
    assert "administrative" in result.skipped[0]["reason"]
    assert result.summary()["analyzed_pages"] == 3


def test_safety_wording_is_never_filtered_even_in_admin_section():
    result = filter_protocol_pages(_pages(TITLE, ELIGIBILITY, HIDDEN_SAFETY))
    assert 3 in _kept(result)
    assert not result.skipped


def test_continuation_of_eligibility_list_is_kept():
    result = filter_protocol_pages(_pages(TITLE, ELIGIBILITY, ELIGIBILITY_CONTINUED))
    assert _kept(result) == [1, 2, 3]


def test_first_page_and_uncertain_pages_are_kept():
    uncertain = "Lorem ipsum dolor sit amet, consectetur."
    result = filter_protocol_pages(_pages(ADMIN, uncertain))
    assert _kept(result) == [1, 2]


def test_first_page_survives_even_when_everything_looks_administrative():
    result = filter_protocol_pages(_pages(ADMIN, ADMIN))
    assert _kept(result) == [1]


def test_original_text_and_page_numbers_are_unchanged():
    pages = _pages(TITLE, ELIGIBILITY, ADMIN, SCHEDULE)
    snapshot = [dict(p) for p in pages]
    result = filter_protocol_pages(pages)
    assert pages == snapshot
    assert all(p in pages for p in result.kept_pages)
    assert [p["text"] for p in result.kept_pages] == [
        p["text"] for p in pages if p["page_number"] != 3
    ]


def test_filter_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CTAI_PATIENT_RELEVANCE_FILTER", "0")
    result = filter_protocol_pages(_pages(TITLE, ELIGIBILITY, ADMIN))
    assert _kept(result) == [1, 2, 3] and not result.skipped
    assert result.summary()["enabled"] is False


def test_blank_pages_are_ignored_in_classification():
    decisions = classify_pages(_pages(TITLE, "   ", ADMIN))
    assert [d.page_number for d in decisions] == [1, 3]


def test_pipeline_sends_only_kept_pages_to_the_model(monkeypatch):
    from services import protocol_pipeline as pipeline

    pages = _pages(TITLE, ELIGIBILITY, ADMIN, SCHEDULE)
    monkeypatch.setattr(pipeline, "extract_pdf", lambda path: {"pages": pages, "warnings": []})
    seen = {}

    def fake_analyze(text):
        from models.protocol_models import ProtocolAnalysis

        seen["text"] = text
        return ProtocolAnalysis(), []

    monkeypatch.setattr(pipeline, "analyze_protocol_with_rescue", fake_analyze)
    monkeypatch.setattr(pipeline, "apply_protocol_accuracy_checks", lambda *a: [])
    monkeypatch.setattr(pipeline, "verify_source_pages", lambda *a: [])
    result = pipeline.analyze_protocol_pdf("ignored.pdf")

    assert "[SOURCE PAGE 3]" not in seen["text"]
    assert "[SOURCE PAGE 2]" in seen["text"] and "[SOURCE PAGE 4]" in seen["text"]
    assert result["relevance"]["skipped_pages"][0]["page"] == 3
    perf = result["performance"]
    assert perf["total_seconds"] >= 0 and "llm_analysis" in perf["stage_seconds"]
    assert perf["input_tokens"] is None  # no provider usage was reported


def test_performance_tracker_sums_reported_token_counts():
    from services import performance

    with performance.track() as tracker:
        performance.record_response({"prompt_eval_count": 100, "eval_count": 20})
        performance.record_response({"prompt_eval_count": 50, "eval_count": 5})
    summary = tracker.summary()
    assert summary["llm_calls"] == 2
    assert summary["input_tokens"] == 150 and summary["output_tokens"] == 25


def test_module_exposes_no_model_calls():
    # The filter must stay model-free.
    assert not hasattr(patient_relevance, "chat_completion")
