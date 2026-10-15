"""Tests for the evaluation tooling (no Ollama needed)."""

import json
from pathlib import Path
from unittest.mock import patch

from evaluation import run_extraction_eval, run_safety_eval
from evaluation.matching import Item, match_items, overall, score_protocol, similarity
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)

GROUND_TRUTH = Path(__file__).resolve().parent.parent / "evaluation" / "ground_truth"


def test_similarity_requires_same_numbers():
    assert similarity("Age 18 to 65 years", "Participants aged 18 to 65 years old") >= 0.5
    assert similarity("Age 18 to 65 years", "Participants aged 18 to 56 years old") == 0.0
    assert similarity("Follow-up Visit: Day 30.", "Follow-up Visit Day 30") >= 0.5


def test_match_items_is_one_to_one():
    truth = [Item("Blood glucose testing.", 2), Item("Blood pressure measurement.", 2)]
    extracted = [Item("Blood glucose testing", 2), Item("Blood glucose testing again", 2)]
    score = match_items(truth, extracted)

    assert len(score.matched) == 1
    assert [t.text for t in score.missed] == ["Blood pressure measurement."]
    assert len(score.extra) == 1


def _perfect_analysis():
    return ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        inclusion_criteria=[
            EligibilityCriterion(description="Participants must be between 18 and 65 years old.", source_page=1),
            EligibilityCriterion(description="Participants must have Type 2 Diabetes.", source_page=1),
        ],
        exclusion_criteria=[
            EligibilityCriterion(description="Participants with severe kidney disease are excluded.", source_page=1),
        ],
        procedures=[
            TrialProcedure(name="Blood glucose testing.", source_page=2),
            TrialProcedure(name="Blood pressure measurement.", source_page=2),
        ],
        visits=[
            TrialVisit(visit_name="Screening Visit", timing="Day 0", source_page=2),
            TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=1),  # wrong page
        ],
        safety_information=[
            SafetyInformation(description="Participants will be monitored for adverse events.", source_page=2),
        ],
    )


def test_score_protocol_against_demo_ground_truth():
    truth = json.loads((GROUND_TRUTH / "sample_clinical_protocol.json").read_text())
    result = overall(score_protocol(_perfect_analysis(), truth))

    assert result["truth"] == 8
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["page_accuracy"] == 7 / 8


def test_missing_items_lower_recall():
    truth = json.loads((GROUND_TRUTH / "sample_clinical_protocol.json").read_text())
    analysis = _perfect_analysis()
    analysis.exclusion_criteria = []
    result = overall(score_protocol(analysis, truth))

    assert result["matched"] == 7
    assert result["recall"] == 7 / 8


def test_extraction_eval_end_to_end_with_mocked_pipeline(tmp_path):
    pipeline_result = {
        "analysis": _perfect_analysis(),
        "verification": [{"status": "exact_match"}] * 7 + [{"status": "not_verified"}],
        "accuracy_warnings": [],
    }
    with patch.object(run_extraction_eval, "RESULTS", tmp_path), patch.object(
        run_extraction_eval, "run_pipeline", return_value=pipeline_result
    ):
        result = run_extraction_eval.evaluate_one(
            GROUND_TRUTH / "sample_clinical_protocol.json", cached=False, with_guide=False
        )
        report = run_extraction_eval.build_report([result])
        cached = run_extraction_eval.evaluate_one(
            GROUND_TRUTH / "sample_clinical_protocol.json", cached=True, with_guide=False
        )

    assert result["overall"]["recall"] == 1.0
    assert result["verified"] == 7
    assert "Wrong page" in report
    assert cached["overall"] == result["overall"]


def test_safety_eval_risky_questions_are_always_safe():
    questions = run_safety_eval.evaluate_questions()["rows"]
    critical = [r for r in questions if r.get("safety_critical")]

    assert critical
    for r in critical:
        # Never a factual answer to an eligibility-decision or medical-advice question.
        assert r["ok"] or r["got"] == "unsupported", r["q"]


def test_guardrail_catches_every_unsafe_rewrite():
    rows = run_safety_eval.evaluate_guardrails()["rows"]
    assert all(r["ok"] for r in rows), [r["rewrite"] for r in rows if not r["ok"]]
