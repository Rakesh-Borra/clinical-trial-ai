"""
Extraction & Patient Guide evaluation on real protocols (needs Ollama).

For every ground-truth file in evaluation/ground_truth/:
  1. runs the full pipeline (PDF -> OCR -> qwen3:4b -> verification)
  2. compares the extracted items with the hand-labelled ground truth
  3. (with --guide) generates the plain-language patient guide and counts
     how many items passed the checks, needed a retry, or were flagged

Run from the project root, with Ollama running:
    python -m evaluation.run_extraction_eval            # extraction only
    python -m evaluation.run_extraction_eval --guide    # + patient guide
    python -m evaluation.run_extraction_eval --cached   # re-score saved runs, no Ollama
    python -m evaluation.run_extraction_eval --cached --guide   # add the guide to saved runs

Writes evaluation/results/extraction_results.md and one JSON per protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from evaluation.matching import CATEGORIES, overall, pct, score_protocol
from models.protocol_models import ProtocolAnalysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GROUND_TRUTH = HERE / "ground_truth"
RESULTS = HERE / "results"

LABELS = {
    "inclusion_criteria": "Inclusion criteria",
    "exclusion_criteria": "Exclusion criteria",
    "visits": "Visits",
    "procedures": "Procedures",
    "safety_information": "Safety information",
}


def run_pipeline(pdf_path: Path) -> dict:
    from services.protocol_pipeline import analyze_protocol_pdf

    return analyze_protocol_pdf(str(pdf_path))


def _num_ctx() -> Optional[int]:
    try:
        from agents.protocol_analysis_agent import NUM_CTX

        return NUM_CTX
    except Exception:
        return None


def guide_stats(analysis: ProtocolAnalysis) -> dict:
    from agents.plain_language_agent import count_statements, generate_patient_guide

    total = count_statements(analysis)
    done = {"n": 0}

    def progress(item) -> None:
        done["n"] += 1
        status = "flagged" if item.needs_review else "ok"
        print(f"    guide item {done['n']}/{total}: {status}", flush=True)

    start = time.time()
    guide = generate_patient_guide(analysis, on_item=progress)
    items = guide.all_items()
    return {
        "items": len(items),
        "passed_checks": sum(1 for i in items if i.generated_by == "llm"),
        "flagged": sum(1 for i in items if i.needs_review),
        "flag_reasons": [r for i in items for r in i.review_reasons],
        "seconds": round(time.time() - start, 1),
    }


def evaluate_one(truth_path: Path, cached: bool, with_guide: bool) -> Optional[dict]:
    truth = json.loads(truth_path.read_text())
    pdf = ROOT / truth["pdf"]
    saved = RESULTS / f"{truth_path.stem}_run.json"

    if cached:
        if not saved.exists():
            print(f"  skipped (no saved run): {truth_path.name}")
            return None
        record = json.loads(saved.read_text())
        analysis = ProtocolAnalysis.model_validate(record["analysis"])
    else:
        if not pdf.exists():
            print(f"  skipped (PDF not found): {pdf}")
            return None
        print(f"  analysing {pdf.name} … (this can take several minutes)", flush=True)
        start = time.time()
        result = run_pipeline(pdf)
        analysis = result["analysis"]
        record = {
            "pdf": truth["pdf"],
            "seconds": round(time.time() - start, 1),
            "analysis": analysis.model_dump(),
            "verification": result.get("verification", []),
            "accuracy_warnings": result.get("accuracy_warnings", []),
            "extraction_warnings": result.get("extraction_warnings", []),
            "num_ctx": _num_ctx(),
        }
        # Save straight away, so stopping during the guide step keeps the extraction.
        RESULTS.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(record, indent=2))
        print(f"  extraction saved ({record['seconds'] / 60:.1f} min)", flush=True)

    if with_guide and not record.get("guide"):
        print("  generating patient guide …", flush=True)
        record["guide"] = guide_stats(analysis)
        RESULTS.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(record, indent=2))

    scores = score_protocol(analysis, truth)
    verification = record.get("verification", [])
    return {
        "name": truth_path.stem,
        "pdf": truth["pdf"],
        "pages": truth.get("pages"),
        "seconds": record.get("seconds"),
        "scores": scores,
        "overall": overall(scores),
        "title_ok": (analysis.trial_title or "").strip().lower() == (truth.get("trial_title") or "").strip().lower(),
        "verified": sum(1 for v in verification if v.get("status") == "exact_match"),
        "verification_total": len(verification),
        "guide": record.get("guide"),
        "pdf_warnings": len(record.get("extraction_warnings", [])),
        "num_ctx": record.get("num_ctx"),
    }


def build_report(results: list) -> str:
    lines = [
        "# Extraction & Patient Guide Evaluation",
        "",
        "Generated by `python -m evaluation.run_extraction_eval`.",
        "Ground truth was labelled by hand from each protocol PDF (`evaluation/ground_truth/`).",
        "",
        "- **Recall** = share of real protocol statements the app found.",
        "- **Precision** = share of the app's statements that match a real one.",
        "- **Page accuracy** = matched statements cited on the correct PDF page.",
        "- A match needs most words in common and **every number identical**.",
        "",
        "## Summary",
        "",
        "| Protocol | Pages | Time | Recall | Precision | Page accuracy | Verified on page | PDF warnings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        o = r["overall"]
        seconds = f"{r['seconds'] / 60:.1f} min" if r["seconds"] else "n/a"
        verified = f"{r['verified']}/{r['verification_total']}" if r["verification_total"] else "n/a"
        lines.append(
            f"| {r['name']} | {r['pages'] or 'n/a'} | {seconds} | {pct(o['recall'])} ({o['matched']}/{o['truth']}) "
            f"| {pct(o['precision'])} | {pct(o['page_accuracy'])} | {verified} | {r.get('pdf_warnings', 0)} |"
        )
    contexts = sorted({r["num_ctx"] for r in results if r.get("num_ctx")})
    if contexts:
        lines += ["", f"Model context window (num_ctx): {', '.join(str(c) for c in contexts)} tokens."]

    lines += ["", "## By category (all protocols combined)", "", "| Category | Found / real | Recall | Precision | Page accuracy |", "|---|---|---|---|---|"]
    for category in CATEGORIES:
        cat = [s for r in results for s in r["scores"] if s.category == category]
        o = overall(cat)
        lines.append(
            f"| {LABELS[category]} | {o['matched']}/{o['truth']} | {pct(o['recall'])} "
            f"| {pct(o['precision'])} | {pct(o['page_accuracy'])} |"
        )

    guides = [r for r in results if r["guide"]]
    if guides:
        lines += ["", "## Plain-language Patient Guide", "", "| Protocol | Items | Passed checks | Flagged for review | Time |", "|---|---|---|---|---|"]
        for r in guides:
            g = r["guide"]
            lines.append(
                f"| {r['name']} | {g['items']} | {g['passed_checks']} ({pct(g['passed_checks'] / g['items'] if g['items'] else None)}) "
                f"| {g['flagged']} | {g.get('seconds', 'n/a')} s |"
            )
        reasons = [reason for r in guides for reason in r["guide"]["flag_reasons"]]
        if reasons:
            lines += ["", "Why items were flagged:", ""]
            lines += [f"- {reason}" for reason in sorted(set(reasons))[:15]]

    lines += ["", "## Failure cases (missed or extra statements)", ""]
    for r in results:
        lines.append(f"### {r['name']}")
        any_issue = False
        for s in r["scores"]:
            for t in s.missed:
                lines.append(f"- **Missed** ({LABELS[s.category]}, p.{t.page}): {t.text}")
                any_issue = True
            for e in s.extra:
                lines.append(f"- **Extra / wrongly split** ({LABELS[s.category]}, p.{e.page}): {e.text}")
                any_issue = True
            for t, e, _ in s.matched:
                if t.page is not None and not t.page_ok(e.page):
                    lines.append(f"- **Wrong page** ({LABELS[s.category]}): cited p.{e.page}, real p.{t.page}: {t.text}")
                    any_issue = True
        if not any_issue:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--guide", action="store_true", help="also generate and score the patient guide")
    parser.add_argument("--cached", action="store_true", help="re-score saved runs without calling Ollama")
    parser.add_argument("only", nargs="*", help="ground-truth file names to run (default: all)")
    args = parser.parse_args()

    files = sorted(GROUND_TRUTH.glob("*.json"))
    if args.only:
        files = [f for f in files if f.stem in args.only or f.name in args.only]

    results = []
    for path in files:
        print(f"[{path.stem}]")
        result = evaluate_one(path, cached=args.cached, with_guide=args.guide)
        if result:
            o = result["overall"]
            print(f"  recall {pct(o['recall'])} · precision {pct(o['precision'])} · page accuracy {pct(o['page_accuracy'])}")
            results.append(result)

    if not results:
        print("No protocols evaluated.")
        return
    report = build_report(results)
    (RESULTS / "extraction_results.md").write_text(report)
    print(f"\nSaved to {RESULTS / 'extraction_results.md'}")


if __name__ == "__main__":
    main()
