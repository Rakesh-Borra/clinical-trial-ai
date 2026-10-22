"""
Progress reporting for background protocol analysis.

The job manager installs a reporter for the worker thread via ``bind``. Pipeline
code calls ``report_stage``; outside a job it is a no-op, so the analysis
functions behave exactly as before.

Percentages are derived from completed work units (PDF pages, model passes)
inside fixed pipeline milestones. A stage whose size is unknown is reported as
indeterminate: its percentage stays at the stage's starting milestone.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class ProgressState:
    percent: float = 0.0
    stage: str = "Queued"
    determinate: bool = False
    completed: Optional[int] = None
    total: Optional[int] = None


# stage key -> (label, start %, end %); the pipeline never reports 100.
STAGES: dict[str, tuple[str, float, float]] = {
    "extract": ("Reading PDF pages", 0.0, 20.0),
    "analyze": ("Analyzing protocol with the AI model", 20.0, 80.0),
    "rescue": ("Checking eligibility criteria", 80.0, 85.0),
    "accuracy": ("Running accuracy checks", 85.0, 92.0),
    "verify": ("Verifying source pages", 92.0, 99.0),
}

Reporter = Callable[[ProgressState], None]
_reporter: contextvars.ContextVar[Optional[Reporter]] = contextvars.ContextVar(
    "analysis_progress_reporter", default=None
)


@contextmanager
def bind(reporter: Reporter) -> Iterator[None]:
    token = _reporter.set(reporter)
    try:
        yield
    finally:
        _reporter.reset(token)


def stage_state(
    stage: str, completed: Optional[int] = None, total: Optional[int] = None
) -> ProgressState:
    label, start, end = STAGES[stage]
    if total and total > 0 and completed is not None:
        done = max(0, min(completed, total))
        percent = start + (end - start) * done / total
        return ProgressState(percent, label, True, done, total)
    return ProgressState(start, label, False)


def report_stage(
    stage: str, completed: Optional[int] = None, total: Optional[int] = None
) -> None:
    reporter = _reporter.get()
    if reporter is not None:
        reporter(stage_state(stage, completed, total))
