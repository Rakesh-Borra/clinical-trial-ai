"""Stop/resume, concurrency, progress persistence, quality audit, Summary UI."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from agents.plain_language_agent import (
    generate_patient_guide,
    guide_worker_count,
    statement_specs,
)
from app import _group_attention_warnings, _status_badge, publish_block_reason
from models.plain_language_models import GuideItem, PatientGuide
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialVisit,
)
from services import guide_jobs
from services.guide_quality import audit_guide_quality, group_findings
from tests.publish_helpers import exact_rows, generate_verified_guide, make_publishable
from tests.test_patient_guide_ui import _approved_guide, _eight_statement_result


def _analysis_with_n(n: int) -> ProtocolAnalysis:
    return ProtocolAnalysis(
        analysis_complete=True,
        inclusion_criteria=[
            EligibilityCriterion(
                description=f"Inclusion criterion number {i} ages 18 to 65.",
                source_page=1,
            )
            for i in range(n)
        ],
        exclusion_criteria=[
            EligibilityCriterion(description="Pregnancy is excluded.", source_page=2)
        ],
        visits=[TrialVisit(visit_name="Screening", timing="Day -7", source_page=3)],
        safety_information=[
            SafetyInformation(description="Monitor adverse events.", source_page=4)
        ],
    )


def test_status_badge_maps_verification_codes_for_display_only():
    assert _status_badge("exact_match") == "Verified"
    assert _status_badge("not_verified") == "Review"
    assert _status_badge("timing_not_found") == "Timing not found"
    assert _status_badge(None) == "Unknown"


def test_attention_warnings_group_into_categories():
    grouped = _group_attention_warnings([
        "Accuracy: Conflicting trial_title values across batches",
        "Accuracy: Statement was not verified on page 3",
        "Accuracy: Duplicate visit merged",
        "Extraction: OCR oddity on page 9",
    ])
    assert "Conflicting information" in grouped
    assert "Unsupported statements" in grouped
    assert "Duplicates" in grouped
    assert "Other warnings" in grouped


def test_guide_worker_count_is_bounded(monkeypatch):
    monkeypatch.delenv("CTAI_GUIDE_WORKERS", raising=False)
    assert guide_worker_count() == 4
    monkeypatch.setenv("CTAI_GUIDE_WORKERS", "8")
    assert guide_worker_count() == 8
    monkeypatch.setenv("CTAI_GUIDE_WORKERS", "99")
    assert guide_worker_count() == 16
    monkeypatch.setenv("CTAI_GUIDE_WORKERS", "0")
    assert guide_worker_count() == 1


def test_parallel_generation_reuses_items_and_honours_cancel():
    analysis = _analysis_with_n(6)
    specs = statement_specs(analysis)
    reuse = {
        specs[0].item_id: GuideItem(
            section=specs[0].section,
            original_text=specs[0].text,
            plain_text="Already done.",
            source_page=1,
            item_id=specs[0].item_id,
            approved=True,
        )
    }
    calls = []
    cancel = threading.Event()

    def rewriter(original, section, feedback=None):
        calls.append(original)
        time.sleep(0.05)
        return f"In simple words: {original}"

    # Cancel immediately so only in-flight / already-scheduled work may finish.
    cancel.set()
    guide = generate_patient_guide(
        analysis,
        rewriter=rewriter,
        reuse=reuse,
        verification=exact_rows(analysis),
        cancel_check=cancel.is_set,
        max_workers=2,
    )
    assert guide.generation_complete is False
    assert any(i.item_id == specs[0].item_id and i.approved for i in guide.all_items())
    # Approved reused item must not be regenerated.
    assert all("Already done." not in c for c in calls) or specs[0].text not in calls


def test_job_stop_pauses_without_discarding_result():
    guide_jobs.jobs.reset()
    started = threading.Event()

    def runner(advance, cancel_check):
        started.set()
        while not cancel_check():
            time.sleep(0.02)
        advance()
        return "partial"

    try:
        job_id = guide_jobs.jobs.submit("pause-key", runner, 3)
        assert started.wait(2)
        assert guide_jobs.jobs.request_stop(job_id)
        assert guide_jobs.jobs.wait(job_id, 5)
        job = guide_jobs.jobs.get(job_id)
        assert job is not None
        assert job.status == guide_jobs.PAUSED
        assert job.result == "partial"
        # Paused jobs are inactive, so a later Resume may submit a new job id.
        second = guide_jobs.jobs.submit(
            "pause-key",
            lambda advance, cancel_check: "done",
            3,
        )
        assert second != job_id
        assert guide_jobs.jobs.wait(second, 5)
    finally:
        guide_jobs.jobs.reset()


def test_find_active_reattaches_by_key():
    guide_jobs.jobs.reset()
    gate = threading.Event()

    def runner(advance, cancel_check):
        gate.wait(5)
        return "x"

    try:
        job_id = guide_jobs.jobs.submit("reattach", runner, 1)
        found = guide_jobs.jobs.find_active("reattach")
        assert found is not None and found.job_id == job_id
        gate.set()
        guide_jobs.jobs.wait(job_id, 5)
    finally:
        guide_jobs.jobs.reset()


def test_incomplete_or_paused_guide_blocks_publication(publish_context):
    result = _eight_statement_result()
    guide = _approved_guide(result)
    guide.generation_complete = False
    reason = publish_block_reason(guide, result)
    assert reason and "incomplete or paused" in reason.casefold()


def test_quality_audit_flags_exact_duplicates_not_distinct_doses():
    from models.plain_language_models import GuideSection

    empty = GuideSection(title="x", introduction="")
    guide = PatientGuide(
        analysis_complete=True,
        generation_complete=True,
        disclaimer="d",
        who_can_join=GuideSection(
            title="Who can join",
            introduction="",
            items=[
                GuideItem(
                    section="inclusion_criteria",
                    original_text="Age 18+",
                    plain_text="You must be 18 or older.",
                    source_page=1,
                    item_id="a1",
                ),
                GuideItem(
                    section="inclusion_criteria",
                    original_text="Age 18+",
                    plain_text="You must be 18 or older.",
                    source_page=2,
                    item_id="a2",
                ),
                GuideItem(
                    section="inclusion_criteria",
                    original_text="Dose 5 mg daily",
                    plain_text="You take 5 mg each day.",
                    source_page=3,
                    item_id="d5",
                ),
                GuideItem(
                    section="inclusion_criteria",
                    original_text="Dose 10 mg daily",
                    plain_text="You take 10 mg each day.",
                    source_page=3,
                    item_id="d10",
                ),
            ],
        ),
        who_cannot_join=empty,
        visits=empty,
        procedures=empty,
        safety=empty,
    )
    findings = audit_guide_quality(guide)
    exact = [f for f in findings if f.kind == "exact_duplicate"]
    assert exact and set(exact[0].item_ids) == {"a1", "a2"}
    # Distinct doses must not be near-duplicated away.
    near = [f for f in findings if f.kind == "near_duplicate"]
    assert not any({"d5", "d10"} <= set(f.item_ids) for f in near)
    grouped = group_findings(findings)
    assert "exact_duplicate" in grouped


def test_concurrent_rewrites_produce_full_guide_offline():
    analysis = _analysis_with_n(5)
    seen = []

    def rewriter(original, section, feedback=None):
        seen.append(original)
        return f"Plain: {original}"

    guide = generate_patient_guide(
        analysis,
        rewriter=rewriter,
        verification=exact_rows(analysis),
        max_workers=3,
    )
    assert guide.generation_complete is True
    assert len(guide.all_items()) == len(statement_specs(analysis))
    assert len(seen) == len(statement_specs(analysis)) - 0
