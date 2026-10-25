"""
Lightweight performance measurements for one protocol analysis.

A tracker is activated for the duration of a pipeline run; the model provider
reports each response's token counts to whichever tracker is active. Token
counts are only available when the provider returns them (Ollama reports
prompt_eval_count and eval_count); otherwise they stay None, never guessed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

_current: ContextVar[Optional["PerfTracker"]] = ContextVar("ctai_perf", default=None)


def _count(response: Any, name: str) -> Optional[int]:
    value = response.get(name) if isinstance(response, dict) else getattr(response, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class PerfTracker:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.stage_seconds: Dict[str, float] = {}
        self._started = time.perf_counter()

    def record_response(self, response: Any) -> None:
        self.llm_calls += 1
        for attr, name in (("input_tokens", "prompt_eval_count"), ("output_tokens", "eval_count")):
            count = _count(response, name)
            if count is not None:
                setattr(self, attr, (getattr(self, attr) or 0) + count)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_seconds": round(time.perf_counter() - self._started, 3),
            "stage_seconds": {k: round(v, 3) for k, v in self.stage_seconds.items()},
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@contextmanager
def track() -> Iterator[PerfTracker]:
    tracker = PerfTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def record_response(response: Any) -> None:
    """Called by the model provider after every successful response."""
    tracker = _current.get()
    if tracker is not None:
        tracker.record_response(response)
