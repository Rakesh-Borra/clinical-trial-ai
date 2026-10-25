"""
In-process background jobs for patient-statement generation.

Same model as services/analysis_jobs.py: a daemon thread does the work and only
touches this registry under a lock; the Streamlit script collects the finished
result on the main thread. A job is identified by a key (protocol + analysis
fingerprint), so reruns and repeated clicks reuse the active job instead of
starting a second one. One generation runs at a time.

Cooperative cancellation: ``request_stop`` sets a flag the worker checks between
statements. In-flight model calls finish and may still be saved; no new work is
scheduled. A cancelled run ends as ``paused`` (not discarded) so Resume can
continue from SQLite without redoing completed items.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

QUEUED = "queued"
RUNNING = "running"
STOPPING = "stopping"
PAUSED = "paused"
COMPLETED = "completed"
FAILED = "failed"
DISCARDED = "discarded"

ACTIVE_STATUSES = (QUEUED, RUNNING, STOPPING)
_RETAIN_FINISHED_SECONDS = 3600

# runner(advance, cancel_check) -> result
GuideRunner = Callable[[Callable[[], None], Callable[[], bool]], Any]


@dataclass(frozen=True)
class GuideJob:
    job_id: str
    key: str
    status: str
    total: int = 0
    completed: int = 0
    result: Any = None
    error: Optional[BaseException] = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def fraction(self) -> float:
        if self.status == COMPLETED:
            return 1.0
        if self.total <= 0:
            return 0.0
        return min(self.completed / self.total, 0.99)


class GuideJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_slot = threading.Lock()
        self._jobs: dict[str, GuideJob] = {}
        self._finished_at: dict[str, float] = {}
        self._done: dict[str, threading.Event] = {}
        self._cancel: dict[str, threading.Event] = {}
        self.started = 0

    def submit(
        self,
        key: str,
        runner: GuideRunner,
        total: int,
    ) -> str:
        """Start generation, or return the active job for the same key."""
        with self._lock:
            self._prune_locked()
            for job in self._jobs.values():
                if job.active and job.key == key:
                    return job.job_id
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = GuideJob(job_id, key, QUEUED, total=total)
            self._done[job_id] = threading.Event()
            self._cancel[job_id] = threading.Event()
            self.started += 1
        threading.Thread(
            target=self._work,
            args=(job_id, runner),
            name=f"guide-{job_id[:8]}",
            daemon=True,
        ).start()
        return job_id

    def get(self, job_id: Optional[str]) -> Optional[GuideJob]:
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def find_active(self, key: str) -> Optional[GuideJob]:
        """Locate a running/queued/stopping job by protocol fingerprint key."""
        with self._lock:
            for job in self._jobs.values():
                if job.active and job.key == key:
                    return job
        return None

    def find_by_key(self, key: str) -> Optional[GuideJob]:
        """Most recent job for this key (active or finished, not discarded)."""
        with self._lock:
            matches = [
                job for job in self._jobs.values()
                if job.key == key and job.status != DISCARDED
            ]
        if not matches:
            return None
        return matches[-1]

    def request_stop(self, job_id: Optional[str]) -> bool:
        """Ask an active job to stop after finishing in-flight work."""
        if not job_id:
            return False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.active:
                return False
            flag = self._cancel.get(job_id)
            if flag is not None:
                flag.set()
            if job.status != STOPPING:
                self._jobs[job_id] = replace(job, status=STOPPING)
            return True

    def is_cancelled(self, job_id: str) -> bool:
        flag = self._cancel.get(job_id)
        return flag is not None and flag.is_set()

    def discard(self, job_id: Optional[str]) -> None:
        """Forget a job; a still-running worker's result is dropped."""
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.active:
                flag = self._cancel.get(job_id)
                if flag is not None:
                    flag.set()
                self._jobs[job_id] = replace(job, status=DISCARDED)
            else:
                self._forget_locked(job_id)

    def release(self, job_id: Optional[str]) -> None:
        """Drop a finished job once its outcome has been collected."""
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and not job.active:
                self._forget_locked(job_id)

    def wait(self, job_id: Optional[str], timeout: float = 15.0) -> bool:
        """Block until the worker has finished; used by tests."""
        if not job_id:
            return True
        with self._lock:
            event = self._done.get(job_id)
        return event.wait(timeout) if event is not None else True

    def wait_idle(self, timeout: float = 15.0) -> bool:
        """Block until no job is active; used by tests."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                active = [j.job_id for j in self._jobs.values() if j.active]
            if not active:
                return True
            for job_id in active:
                self.wait(job_id, max(deadline - time.monotonic(), 0.01))
        return False

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._finished_at.clear()
            self._done.clear()
            self._cancel.clear()
            self.started = 0

    def _forget_locked(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._finished_at.pop(job_id, None)
        self._done.pop(job_id, None)
        self._cancel.pop(job_id, None)

    def _prune_locked(self) -> None:
        cutoff = time.monotonic() - _RETAIN_FINISHED_SECONDS
        for job_id, finished in list(self._finished_at.items()):
            if finished < cutoff:
                self._forget_locked(job_id)

    def _advance(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in (RUNNING, STOPPING):
                return
            self._jobs[job_id] = replace(job, completed=job.completed + 1)

    def _work(self, job_id: str, runner: GuideRunner) -> None:
        event = None
        try:
            with self._run_slot:
                with self._lock:
                    event = self._done.get(job_id)
                    job = self._jobs.get(job_id)
                    if job is None or job.status == DISCARDED:
                        return
                    if self.is_cancelled(job_id):
                        self._jobs[job_id] = replace(job, status=PAUSED)
                        self._finished_at[job_id] = time.monotonic()
                        return
                    self._jobs[job_id] = replace(job, status=RUNNING)
                outcome: dict[str, Any]
                status = COMPLETED
                try:
                    outcome = {
                        "result": runner(
                            lambda: self._advance(job_id),
                            lambda: self.is_cancelled(job_id),
                        )
                    }
                    if self.is_cancelled(job_id):
                        status = PAUSED
                except BaseException as exc:  # noqa: BLE001 — reported to the UI
                    outcome = {"error": exc}
                    status = FAILED
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None and job.status != DISCARDED:
                        self._jobs[job_id] = replace(job, status=status, **outcome)
                        self._finished_at[job_id] = time.monotonic()
                    elif job is not None:
                        self._forget_locked(job_id)
        finally:
            if event is not None:
                event.set()


jobs = GuideJobManager()
