"""
Safety & behaviour evaluation (no Ollama needed).

1. Patient questions: sends realistic questions through the same path the app
   uses (glossary lookup, then the Patient Assistant) and checks the intent.
2. Guardrail cases: checks whether the plain-language safety check accepts
   good rewrites and rejects bad ones (dropped numbers, eligibility
   decisions, medical advice, over-promising).

Run from the project root:
    python -m evaluation.run_safety_eval

Writes evaluation/results/safety_results.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from agents.patient_agent import answer_patient_query
from agents.plain_language_agent import check_rewrite, explain_term_query
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from evaluation.matching import pct

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "safety_results.md"


def demo_pipeline_result() -> tuple[ProtocolAnalysis, list[dict]]:
    """Demo protocol content with every statement verified on its page."""
    analysis = ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
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
            TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=2),
        ],
        safety_information=[
            SafetyInformation(description="Participants will be monitored for adverse events.", source_page=2),
        ],
    )
    verification = []
    for category, field in [
        ("inclusion_criteria", "description"),
        ("exclusion_criteria", "description"),
        ("procedures", "name"),
        ("visits", "visit_name"),
        ("safety_information", "description"),
    ]:
        for item in getattr(analysis, category):
            verification.append({
                "category": category,
                "statement": getattr(item, field),
                "source_page": item.source_page,
                "status": "exact_match",
            })
    return analysis, verification


def route_question(query: str, analysis, verification) -> str:
    """Same order as the app: glossary definitions first, then Patient Assistant."""
    if explain_term_query(query):
        return "term_definition"
    return answer_patient_query(query, analysis, verification).intent


def evaluate_questions() -> dict:
    data = json.loads((HERE / "patient_questions.json").read_text())
    analysis, verification = demo_pipeline_result()
    rows = []
    for case in data["questions"]:
        got = route_question(case["q"], analysis, verification)
        rows.append({**case, "got": got, "ok": got in case["expect"]})
    return {"rows": rows}


def evaluate_guardrails() -> dict:
    data = json.loads((HERE / "guardrail_cases.json").read_text())
    rows = []
    for case in data["cases"]:
        problems = check_rewrite(case["original"], case["rewrite"])
        passed = not problems
        rows.append({**case, "passed": passed, "ok": passed == case["should_pass"], "problems": problems})
    return {"rows": rows}


def _question_stats(rows: list) -> dict:
    critical = [r for r in rows if r.get("safety_critical")]
    return {
        "n": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "critical": len(critical),
        "refused": sum(r["ok"] for r in critical),
        # Safe = refused, or the generic "I can't answer that" fallback.
        # Unsafe = the assistant answered a decision/advice question with facts.
        "safe": sum(1 for r in critical if r["ok"] or r["got"] == "unsupported"),
    }


def _frac(a: int, b: int) -> str:
    return f"{a}/{b} ({pct(a / b)})" if b else "n/a"


def build_report(questions: dict, guardrails: dict) -> str:
    q_rows = questions["rows"]
    g_rows = guardrails["rows"]
    tuned = _question_stats([r for r in q_rows if not r.get("held_out")])
    held = _question_stats([r for r in q_rows if r.get("held_out")])

    by_group = defaultdict(lambda: {"tuned": [], "held": []})
    for r in q_rows:
        by_group[r["group"]]["held" if r.get("held_out") else "tuned"].append(r)

    bad = [r for r in g_rows if not r["should_pass"]]
    good = [r for r in g_rows if r["should_pass"]]
    caught = sum(1 for r in bad if not r["passed"])
    accepted = sum(1 for r in good if r["passed"])

    lines = [
        "# Safety & Behaviour Evaluation",
        "",
        "Generated by `python -m evaluation.run_safety_eval` (deterministic, no Ollama).",
        "",
        "- **Development set**: questions used while fixing the keyword rules.",
        "- **Held-out set**: new phrasings written after the fixes and never used for tuning.",
        "  This is the honest estimate of how the assistant handles questions it has not seen.",
        "",
        "## Headline numbers",
        "",
        "| Metric | Development set | Held-out set |",
        "|---|---|---|",
        f"| Risky questions **refused** (eligibility decisions + medical advice) | {_frac(tuned['refused'], tuned['critical'])} | {_frac(held['refused'], held['critical'])} |",
        f"| Risky questions answered **safely** (refused, or generic \"can't answer\" fallback) | **{_frac(tuned['safe'], tuned['critical'])}** | **{_frac(held['safe'], held['critical'])}** |",
        f"| All questions handled as expected | {_frac(tuned['ok'], tuned['n'])} | {_frac(held['ok'], held['n'])} |",
        "",
        "| Plain-language guardrail | Result |",
        "|---|---|",
        f"| Unsafe rewrites caught (dropped/changed numbers, eligibility decisions, advice, over-promising) | **{_frac(caught, len(bad))}** |",
        f"| Good rewrites accepted | {_frac(accepted, len(good))} |",
        "",
        "## Patient questions by group",
        "",
        "| Group | Development set | Held-out set |",
        "|---|---|---|",
    ]
    for group, sets in by_group.items():
        t, h = sets["tuned"], sets["held"]
        lines.append(f"| {group} | {_frac(sum(r['ok'] for r in t), len(t))} | {_frac(sum(r['ok'] for r in h), len(h))} |")

    lines += ["", "## Patient questions — every case", "", "| Set | Question | Expected | Got | OK |", "|---|---|---|---|---|"]
    for r in q_rows:
        lines.append(
            f"| {'held-out' if r.get('held_out') else 'dev'} | {r['q']} | {' / '.join(r['expect'])} "
            f"| {r['got']} | {'✅' if r['ok'] else '❌'} |"
        )

    lines += ["", "## Guardrail cases — every case", "", "| Original | Rewrite | Should pass | Passed | OK |", "|---|---|---|---|---|"]
    for r in g_rows:
        rewrite = r["rewrite"] or "(empty)"
        lines.append(
            f"| {r['original']} | {rewrite} | {'yes' if r['should_pass'] else 'no'} "
            f"| {'yes' if r['passed'] else 'no'} | {'✅' if r['ok'] else '❌'} |"
        )

    failures = [r for r in q_rows if not r["ok"]] + [r for r in g_rows if not r["ok"]]
    lines += ["", "## Failure cases", ""]
    if not failures:
        lines.append("None in this run.")
    for r in failures:
        if "q" in r:
            tag = "held-out" if r.get("held_out") else "dev"
            lines.append(f"- ({tag}) **\"{r['q']}\"** → `{r['got']}` (expected {' / '.join(r['expect'])})")
        else:
            verdict = "accepted" if r["passed"] else "rejected"
            lines.append(f"- Rewrite **\"{r['rewrite']}\"** was wrongly {verdict} ({r['why']})")
    return "\n".join(lines) + "\n"


def main() -> None:
    questions = evaluate_questions()
    guardrails = evaluate_guardrails()
    report = build_report(questions, guardrails)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(report)
    print(report)
    print(f"Saved to {RESULTS}")


if __name__ == "__main__":
    main()
