
"""
Thin wrapper that answers coordinator questions from pipeline output.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agents.coordinator_assistant_agent import answer_coordinator_query
from models.coordinator_models import CoordinatorAnswer
from models.protocol_models import ProtocolAnalysis


def answer_from_pipeline_result(
    query: str,
    pipeline_result: Mapping[str, Any],
) -> CoordinatorAnswer:
    """
    Run the deterministic coordinator assistant on analyze_protocol_pdf output.

    Expects keys: analysis, verification, and optional accuracy_warnings.
    Does not modify the Phase 3 pipeline.
    """
    analysis = pipeline_result["analysis"]
    if not isinstance(analysis, ProtocolAnalysis):
        raise TypeError(
            "pipeline_result['analysis'] must be a ProtocolAnalysis instance."
        )

    verification = list(pipeline_result.get("verification") or [])
    accuracy_warnings: Optional[list[str]] = list(
        pipeline_result.get("accuracy_warnings") or []
    ) or None

    return answer_coordinator_query(
        query,
        analysis,
        verification,
        accuracy_warnings=accuracy_warnings,
    )
