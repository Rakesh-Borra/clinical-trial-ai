"""Opt-in benchmark timing log: off by default, never affects the app."""

import json
from unittest.mock import patch

import pytest

from services import benchmark_log, protocol_pipeline
from services.summary_review import generate_patient_summary
from tests.test_patient_guide_ui import _approved_guide


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(benchmark_log.ENV_VAR, raising=False)
    benchmark_log.record("analysis", "x.pdf", seconds=1)
    assert not benchmark_log.enabled()
    assert list(tmp_path.iterdir()) == []


def test_enabled_appends_json_lines(tmp_path, monkeypatch):
    log = tmp_path / "run.benchmark.jsonl"
    monkeypatch.setenv(benchmark_log.ENV_VAR, str(log))
    benchmark_log.record("analysis", "a.pdf", seconds=1.5, success=True)
    benchmark_log.record("summary", "a.pdf", seconds=0.5, success=True)
    rows = _rows(log)
    assert [r["event"] for r in rows] == ["analysis", "summary"]
    assert rows[0]["seconds"] == 1.5 and "at" in rows[0]


def test_unwritable_log_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(benchmark_log.ENV_VAR, str(tmp_path / "missing" / "dir" / "log.jsonl"))
    benchmark_log.record("analysis", "a.pdf", seconds=1)  # must not raise


def test_pipeline_logs_analysis_stage_and_failures(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv(benchmark_log.ENV_VAR, str(log))
    benchmark_log.name_file("/tmp/ctai_protocol_abc.pdf", "Real Name.pdf")
    fake = {
        "analysis_complete": True, "verification_passed": False,
        "unsupported_statements": 2, "pages": {"total": 10, "extracted": 10, "analyzed": 9},
        "_chunk_counts": {"pages_total": 10, "pages_analyzed": 9, "chars_without_filter": 500},
    }
    with patch.object(protocol_pipeline, "_analyze_protocol_pdf", return_value=fake):
        result = protocol_pipeline.analyze_protocol_pdf("/tmp/ctai_protocol_abc.pdf")
    assert "performance" in result  # normal result is unchanged
    row = _rows(log)[0]
    assert row["label"] == "Real Name.pdf" and row["success"] is True
    assert row["unsupported_statements"] == 2 and row["pages_total"] == 10
    assert "stage_seconds" in row and row["seconds"] >= 0

    with patch.object(protocol_pipeline, "_analyze_protocol_pdf", side_effect=ValueError("boom")):
        with pytest.raises(ValueError):
            protocol_pipeline.analyze_protocol_pdf("/tmp/ctai_protocol_abc.pdf")
    failed = _rows(log)[-1]
    assert failed["success"] is False and "boom" in failed["error"]


def test_pipeline_result_identical_when_logging_off(monkeypatch):
    monkeypatch.delenv(benchmark_log.ENV_VAR, raising=False)
    fake = {"analysis_complete": True, "pages": {"extracted": 1}, "_chunk_counts": {"x": 1}}
    with patch.object(protocol_pipeline, "_analyze_protocol_pdf", return_value=fake):
        result = protocol_pipeline.analyze_protocol_pdf("/tmp/whatever.pdf")
    assert result["performance"]["x"] == 1 and "_chunk_counts" not in result


def test_summary_generation_is_timed_and_unchanged(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv(benchmark_log.ENV_VAR, str(log))
    benchmark_log.name_file("/tmp/y.pdf", "y.pdf")
    text, by = generate_patient_summary(_approved_guide(), writer=lambda _p: "unusable")
    assert by == "fallback" and text  # deterministic fallback, no model
    row = _rows(log)[-1]
    assert row["event"] == "summary" and row["label"] == "y.pdf"
    assert row["generated_by"] == "fallback" and row["success"] is True


def test_summarize_joins_stages_per_file(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv(benchmark_log.ENV_VAR, str(log))
    benchmark_log.record("analysis", "a.pdf", success=True, seconds=10, pages_total=5, chars_without_filter=900)
    benchmark_log.record("statements", "a.pdf", success=True, seconds=20, statements=7, warnings=1)
    benchmark_log.record("summary", "a.pdf", success=True, seconds=3)
    table = benchmark_log.summarize(str(log))
    assert "a.pdf | 5 | 900 | 10 | 20 | 3 | 33 | 7 | 1 | yes" in table
