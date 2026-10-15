# Evaluation

How we check that Clinical Trial AI actually works: not just that it runs.

| Evaluation | What it measures | Needs Ollama? | Command |
|---|---|---|---|
| **Safety & behaviour** | Risky patient questions (eligibility decisions, medical advice) are refused; the plain-language guardrail catches unsafe rewrites | No | `python -m evaluation.run_safety_eval` |
| **Extraction & patient guide** | On real protocols: how many real criteria, visits, procedures and safety statements the app finds, with the right page; how many guide rewrites pass the checks | Yes | `python -m evaluation.run_extraction_eval --guide` |

Results are written to `evaluation/results/`. The summary for judges is in [`../EVALUATION.md`](../EVALUATION.md).

## Folder contents

```text
evaluation/
├── patient_questions.json     # 60 patient questions + expected behaviour (dev + held-out)
├── guardrail_cases.json       # 24 good/bad plain-language rewrites, labelled by hand
├── ground_truth/              # hand-labelled answer key for each protocol PDF
├── matching.py                # how extracted statements are compared with ground truth
├── run_safety_eval.py
├── run_extraction_eval.py
└── results/                   # generated reports
```

## Adding a real protocol

1. Download a public protocol PDF from ClinicalTrials.gov and save it in `data/protocols/`
   (that folder is git-ignored; the answer key is committed, the PDF is not).
2. Copy `ground_truth/sample_clinical_protocol.json` to `ground_truth/<short_name>.json`.
3. Set `"pdf"` to the PDF path and `"pages"` to the page count.
4. List every inclusion criterion, exclusion criterion, visit, procedure and safety
   statement **exactly as written**, one per entry, with the PDF page number
   (the page number shown by your PDF viewer, starting at 1).
5. Run `python -m evaluation.run_extraction_eval --guide <short_name>`.

Re-score without re-running Ollama: `python -m evaluation.run_extraction_eval --cached`.

## How matching works

A statement found by the app counts as correct when it shares most of its words
with a ground-truth statement (Dice overlap ≥ 0.5) **and every number is identical**:
"18 to 56 years" never matches "18 to 65 years". Each real statement can be matched once.

## Honest-evaluation rules we follow

- Held-out questions in `patient_questions.json` were written **after** the keyword fixes and
  were never used for tuning. Report both sets.
- Never edit ground truth to make a run look better.
- Keep failure cases in the report: they are part of the result.
