"""
In-process background jobs for protocol analysis.

Jobs run in daemon threads owned by this module, independent of Streamlit's
script reruns. Worker threads never touch Streamlit session state: they only
update the job record under a lock, and the Streamlit script copies finished
results into session state from the main execution context.

Jobs live in this server process's memory. They survive page navigation,
reruns, and logout, but not a server restart. A browser refresh starts a new
Streamlit session that has no reference to earlier job ids.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from services import analysis_progress
from services.analysis_progress import ProgressState

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
DISCARDED = "discarded"

ACTIVE_STATUSES = (QUEUED, RUNNING)
_RETAIN_FINISHED_SECONDS = 3600


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    protocol_id: str
    fingerprint: str
    status: str
    result: Any = None
    error: Optional[BaseException] = None
    progress: ProgressState = ProgressState()

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES


class AnalysisJobManager:
    """Thread-safe registry of analysis jobs, one running at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_slot = threading.Lock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._finished_at: dict[str, float] = {}
        self._done: dict[str, threading.Event] = {}

    def submit(
        self,
        runner: Callable[[str], Any],
        pdf_path: str,
        protocol_id: str,
        fingerprint: str,
    ) -> str:
        """Start a job, or return the active job for the same protocol and PDF."""
        with self._lock:
            self._prune_locked()
            for job in self._jobs.values():
                if (
                    job.active
                    and job.protocol_id == protocol_id
                    and job.fingerprint == fingerprint
                ):
                    return job.job_id
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = JobSnapshot(job_id, protocol_id, fingerprint, QUEUED)
            self._done[job_id] = threading.Event()
        threading.Thread(
            target=self._work,
            args=(job_id, runner, pdf_path),
            name=f"analysis-{job_id[:8]}",
            daemon=True,
        ).start()
        return job_id

    def get(self, job_id: Optional[str]) -> Optional[JobSnapshot]:
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def discard(self, job_id: Optional[str]) -> None:
        """Forget a job; a still-running worker's result is dropped."""
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.active:
                self._set_locked(job_id, replace(job, status=DISCARDED))
            else:
                self._jobs.pop(job_id, None)
                self._finished_at.pop(job_id, None)
                self._done.pop(job_id, None)

    def release(self, job_id: Optional[str]) -> None:
        """Drop a finished job once its outcome has been collected."""
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and not job.active:
                self._jobs.pop(job_id, None)
                self._finished_at.pop(job_id, None)
                self._done.pop(job_id, None)

    def wait(self, job_id: str, timeout: float = 10.0) -> bool:
        """Block until the worker has finished; used by tests."""
        with self._lock:
            event = self._done.get(job_id)
        return event.wait(timeout) if event is not None else True

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._finished_at.clear()
            self._done.clear()

    def _report_progress(self, job_id: str, state: ProgressState) -> None:
        """Record worker progress; the percentage never moves backwards."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != RUNNING:
                return
            percent = min(max(state.percent, job.progress.percent), 99.0)
            self._jobs[job_id] = replace(
                job, progress=replace(state, percent=percent)
            )

    def _set_locked(self, job_id: str, job: JobSnapshot) -> None:
        self._jobs[job_id] = job
        if not job.active:
            self._finished_at[job_id] = time.monotonic()

    def _prune_locked(self) -> None:
        cutoff = time.monotonic() - _RETAIN_FINISHED_SECONDS
        for job_id, finished in list(self._finished_at.items()):
            if finished < cutoff:
                self._jobs.pop(job_id, None)
                self._finished_at.pop(job_id, None)
                self._done.pop(job_id, None)

    def _work(
        self,
        job_id: str,
        runner: Callable[[str], Any],
        pdf_path: str,
    ) -> None:
        event = None
        try:
            with self._run_slot:
                with self._lock:
                    event = self._done.get(job_id)
                    job = self._jobs.get(job_id)
                    if job is None or job.status == DISCARDED:
                        return
                    self._jobs[job_id] = replace(
                        job,
                        status=RUNNING,
                        progress=ProgressState(0.0, "Starting analysis"),
                    )
                try:
                    with analysis_progress.bind(
                        lambda state: self._report_progress(job_id, state)
                    ):
                        outcome: dict[str, Any] = {"result": runner(pdf_path)}
                    status = COMPLETED
                except BaseException as exc:  # noqa: BLE001 — reported to the UI
                    outcome = {"error": exc}
                    status = FAILED
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None and job.status != DISCARDED:
                        if status == COMPLETED:
                            outcome["progress"] = ProgressState(
                                100.0, "Analysis complete", True
                            )
                        self._set_locked(job_id, replace(job, status=status, **outcome))
        finally:
            if event is not None:
                event.set()


jobs = AnalysisJobManager()
