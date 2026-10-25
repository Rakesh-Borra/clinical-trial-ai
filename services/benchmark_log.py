"""
Opt-in benchmark timing log (JSON Lines).

Off by default: nothing is written unless CTAI_BENCHMARK_LOG names a file.
Point it outside the project (for example ~/benchmarks/ctai.jsonl). Each
pipeline stage appends one JSON object; a logging failure never affects the
application. Read the log back with:

    python -m services.benchmark_log /path/to/ctai.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

ENV_VAR = "CTAI_BENCHMARK_LOG"
_lock = threading.Lock()
_names: Dict[str, str] = {}
_last_label = [""]


def enabled() -> bool:
    return bool(os.getenv(ENV_VAR, "").strip())


def name_file(temp_path: str, display_name: str) -> None:
    """Remember the uploaded file's real name for a temporary upload path."""
    if enabled() and display_name:
        _names[str(temp_path)] = display_name
        _last_label[0] = display_name


def last_label() -> str:
    """The most recent upload (benchmarks process one upload at a time)."""
    return _last_label[0]


def label_for(temp_path: str) -> str:
    return _names.get(str(temp_path)) or Path(str(temp_path)).name


def record(event: str, label: str, **fields: Any) -> None:
    """Append one event; silently does nothing when disabled or on I/O errors."""
    target = os.getenv(ENV_VAR, "").strip()
    if not target:
        return
    row = {"event": event, "label": label, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    try:
        with _lock, open(os.path.expanduser(target), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def summarize(path: str) -> str:
    """One row per file: the latest analysis, statements and summary events."""
    latest: "OrderedDict[str, Dict[str, Dict[str, Any]]]" = OrderedDict()
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            latest.setdefault(row["label"], {})[row["event"]] = row
    header = (
        "file | pages | chars | analysis s | statements s | summary s | total s | "
        "statements | warnings | ok"
    )
    lines = [header, "-" * len(header)]
    for label, events in latest.items():
        a, s, m = (events.get(k, {}) for k in ("analysis", "statements", "summary"))
        seconds = [e.get("seconds") for e in (a, s, m)]
        total = sum(x for x in seconds if isinstance(x, (int, float)))
        ok = all(e.get("success") for e in (a, s, m) if e)
        lines.append(" | ".join(str(x) for x in (
            label, a.get("pages_total", "-"), a.get("chars_without_filter", "-"),
            a.get("seconds", "-"), s.get("seconds", "-"), m.get("seconds", "-"),
            round(total, 2), s.get("statements", "-"), s.get("warnings", "-"),
            "yes" if ok and events else "no",
        )))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(summarize(sys.argv[1]))
