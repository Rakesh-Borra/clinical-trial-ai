import threading

import pytest

from services import analysis_jobs, analysis_progress
from services.analysis_progress import ProgressState, report_stage, stage_state


@pytest.fixture()
def manager():
    mgr = analysis_jobs.AnalysisJobManager()
    yield mgr
    mgr.reset()


def test_report_stage_is_noop_outside_a_job():
    report_stage("extract", 1, 2)


def test_stage_state_uses_completed_units_within_milestones():
    state = stage_state("extract", 5, 10)
    assert state.determinate and (state.completed, state.total) == (5, 10)
    assert state.percent == pytest.approx(10.0)
    assert stage_state("analyze", 1, 4).percent == pytest.approx(35.0)


def test_unmeasurable_stage_is_indeterminate_and_not_advanced():
    state = stage_state("rescue")
    assert not state.determinate
    assert state.percent == analysis_progress.STAGES["rescue"][1]


def test_no_stage_reports_100_percent():
    for key in analysis_progress.STAGES:
        assert stage_state(key, 7, 7).percent < 100


def test_job_progress_is_monotonic_capped_and_completes_at_100(manager):
    seen = {}
    release = threading.Event()
    reported = threading.Event()

    def runner(_path):
        report_stage("analyze", 2, 4)
        report_stage("extract", 1, 10)  # earlier stage must not move backwards
        reported.set()
        release.wait(5)
        report_stage("verify", 10, 10)
        return "result"

    job_id = manager.submit(runner, "x.pdf", "p1", "fp")
    assert reported.wait(5)
    seen["during"] = manager.get(job_id).progress
    during = seen["during"]
    assert during.percent == pytest.approx(50.0)
    assert manager.get(job_id).status == analysis_jobs.RUNNING
    release.set()
    assert manager.wait(job_id)
    job = manager.get(job_id)
    assert job.status == analysis_jobs.COMPLETED
    assert job.result == "result"
    assert job.progress.percent == 100.0


def test_failed_job_never_reports_100_and_keeps_stage(manager):
    def runner(_path):
        report_stage("analyze", 1, 4)
        raise RuntimeError("boom")

    job_id = manager.submit(runner, "x.pdf", "p1", "fp")
    assert manager.wait(job_id)
    job = manager.get(job_id)
    assert job.status == analysis_jobs.FAILED
    assert job.progress.percent == pytest.approx(35.0)
    assert job.progress.stage == analysis_progress.STAGES["analyze"][0]


def test_duplicate_submission_returns_same_job(manager):
    release = threading.Event()
    job_id = manager.submit(lambda _p: release.wait(5), "x.pdf", "p1", "fp")
    assert manager.submit(lambda _p: None, "x.pdf", "p1", "fp") == job_id
    release.set()
    assert manager.wait(job_id)


def test_progress_state_default_is_zero_and_indeterminate():
    state = ProgressState()
    assert state.percent == 0.0 and not state.determinate


def test_extract_pdf_reports_page_progress(tmp_path):
    import pymupdf

    from services.pdf_processor import extract_pdf

    pdf = tmp_path / "t.pdf"
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Page {i} " + "text " * 40)
    doc.save(pdf)
    doc.close()

    states = []
    with analysis_progress.bind(states.append):
        extract_pdf(str(pdf))
    assert [s.completed for s in states] == [0, 1, 2, 3]
    assert all(s.total == 3 for s in states)
    assert states[-1].percent == pytest.approx(20.0)
