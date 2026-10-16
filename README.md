# Clinical Trial AI

Hackathon MVP that turns dense clinical trial protocol PDFs into **verified, structured study information** and **source-backed answers** for research coordinators and patients.

**Repository:** [https://github.com/Rakesh-Borra/clinical-trial-ai](https://github.com/Rakesh-Borra/clinical-trial-ai)

This project is an **educational / research prototype**. It does **not** provide medical advice, diagnose conditions, or decide whether any person is eligible for a trial.

**PEAAII AI Applications Hackathon 2026 (Agentic AI).** Start here:

| Document | What it covers |
|---|---|
| [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) | Problem, target users, why this is agentic, workflow diagram, Definition of Done, comparison with existing tools |
| [`EVALUATION.md`](EVALUATION.md) | How we tested it: 3 real ClinicalTrials.gov protocols, 60 patient questions, failure cases, limitations |
| [`evaluation/README.md`](evaluation/README.md) | How to reproduce the evaluation |

### Results at a glance

| What we measured | Result |
|---|---|
| Risky patient questions (eligibility decisions, medical advice) answered safely | **100%** (25/25, incl. 9 never seen during development) |
| Unsafe plain-language rewrites caught by the guardrail | **100%** (13/13) |
| Extracted statements that are correct (3 real protocols) | **91%** (58/64) |
| Page citations that are correct | **91%** (53/58) |
| Inclusion criteria found | **100%** (14/14) |
| All protocol statements found (recall) | 66% on the two short protocols; **23% on an 86-page protocol** (main limitation) |

---

## 1. Overview

Clinical protocols are long and hard to skim. Clinical Trial AI helps by:

1. Uploading a protocol PDF in a Streamlit web app
2. Extracting text (with OCR fallback for scanned pages)
3. Analyzing the protocol with a **local** large language model (`qwen3:4b` via Ollama)
4. Checking extracted statements against source pages
5. Showing a protocol summary plus:
   - **Coordinator Assistant** — factual lookup for research staff
   - **Patient Guide Review** — the AI writes a plain-language patient guide, checks its own output, retries, and flags what it cannot fix; a coordinator edits and **approves** each item
   - **Patient Assistant** — the **approved** guide only, a medical-word glossary, and Q&A with safe refusals. Answers use only approved guide items; anything not yet approved is reported as still under review

Analysis uses Ollama once per uploaded PDF (cached in the session). The patient guide uses Ollama once more, when the coordinator clicks **Generate patient guide**. Coordinator and Patient Q&A are **deterministic**: they only reuse cached results and **do not** call Ollama.

---

## 2. Key features

- **PDF text extraction** with PyMuPDF (`pymupdf`)
- **OCR fallback** for image-only / scanned pages (`pytesseract` + system Tesseract), and for PDFs whose text is **scrambled by a broken font** (detected automatically)
- **Local protocol analysis** with Ollama model **`qwen3:4b`**
- **Source-page verification** (`exact_match` and related statuses)
- **Accuracy / completeness checks** after extraction (warnings surfaced in the UI)
- **Coordinator Assistant** — inclusion/exclusion, visits (including visit counts), procedures, safety, metadata
- **Plain-Language Patient Guide** — AI rewrite → self-check (numbers kept, no advice or eligibility wording) → retry with feedback → flag for human review
- **Coordinator approval** — patients only ever see items a coordinator approved
- **Patient Assistant** — approved guide, medical-word glossary ("What does HbA1c mean?"), safe refusals
- **Evaluation framework** — answer keys for real protocols, safety question bank, reproducible scoring ([`EVALUATION.md`](EVALUATION.md))
- **Streamlit UI** — upload, analyze once, then ask questions from cached results
- **Safety posture** — research disclaimers; no final eligibility decisions; no medical advice

---

## 3. Architecture

```text
PDF upload (Streamlit)
        │
        ▼
Text extraction / OCR fallback
        │
        ▼
Protocol analysis (Ollama + qwen3:4b, num_ctx 16,384)
  • focused extraction passes
  • optional eligibility rescue when needed
  • deterministic accuracy checks
        │
        ▼
Source-page verification
        │
        ▼
Cached pipeline result (session state)
        │
        ├─► Protocol Summary tab
        ├─► Coordinator Assistant (deterministic Q&A, no extra Ollama calls)
        ├─► Patient Guide Review (Ollama rewrite → self-check → retry → flag
        │                          → coordinator edits / approves)
        └─► Patient Assistant (approved guide + glossary + deterministic Q&A)
```

| Stage | Entry point | Uses Ollama? |
|-------|-------------|--------------|
| Analyze PDF | `services/protocol_pipeline.py` → `analyze_protocol_pdf()` | **Yes** (`qwen3:4b`) |
| Coordinator Q&A | `services/coordinator_query.py` → `answer_from_pipeline_result()` | **No** |
| Patient guide | `agents/plain_language_agent.py` → `generate_patient_guide()` | **Yes** (short rewrites, "thinking" off) |
| Patient Q&A | `services/patient_query.py` → `answer_patient_from_pipeline_result()` | **No** |
| UI | `app.py`, `ui/patient_guide_tab.py` | Only on **Analyze protocol** and **Generate patient guide** |

The model's context window is set to 16,384 tokens (~60,000 characters of protocol text). On a computer with 8 GB of RAM, lower it with `export CTAI_NUM_CTX=8192` before starting the app.

---

## 4. Prerequisites

Install these before you run the app:

| Requirement | Notes |
|-------------|--------|
| **Git** | To clone the repository |
| **Python 3.12** | This repo is developed and documented for **Python 3.12** (see `CONTRIBUTING.md`). Use `python3.12` if your system has multiple versions. |
| **Ollama** | Local model runner; must be installed and running |
| **Model `qwen3:4b`** | Exact name used in `agents/protocol_analysis_agent.py` (`MODEL_NAME = "qwen3:4b"`) |
| **Tesseract OCR** (optional but recommended) | Needed only for scanned / image-only PDF pages. Selectable-text PDFs work without it. |

### macOS (verified path used in this project)

- Install Python 3.12 (for example via [python.org](https://www.python.org/downloads/) or Homebrew: `brew install python@3.12`)
- Install Ollama from [https://ollama.com](https://ollama.com)
- For OCR: `brew install tesseract`

### Windows / Linux notes

- **Windows:** Install Python 3.12 from [python.org](https://www.python.org/downloads/) and check “Add Python to PATH”. Install Ollama from the official Windows installer. Tesseract is a separate Windows install if you need OCR; the repo’s OCR helper messages mention macOS Homebrew and Debian/Ubuntu `apt` paths—Windows Tesseract install steps were **not verified** in this environment.
- **Linux (Debian/Ubuntu-style):** Python 3.12 via your distro or deadsnakes; Ollama per official Linux install docs; OCR binary: `sudo apt-get install tesseract-ocr` (command referenced by `services/pdf_processor.py`).

We do **not** claim full end-to-end verification of every Windows or Linux desktop configuration in this README—only that the project is Python 3.12–based and those OS notes match common installs and code comments.

---

## 5. Clone the repository

```bash
git clone https://github.com/Rakesh-Borra/clinical-trial-ai.git
cd clinical-trial-ai
```

---

## 6. Virtual environment and dependencies

Always use a virtual environment so project packages do not conflict with system Python.

### macOS / Linux (bash or zsh)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the active interpreter:

```bash
python -V
# Expect: Python 3.12.x
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script activation, run once (as yourself, not as a global security change you do not understand):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again with `.\.venv\Scripts\Activate.ps1`.

### What gets installed

From `requirements.txt`:

- `streamlit` — web UI
- `ollama` — Python client for local Ollama
- `pymupdf` — PDF text / page rendering
- `pydantic` — structured models
- `python-dotenv` — env file helper (optional for local secrets; do not commit `.env`)
- `pytest` — tests
- `pytesseract` / `Pillow` — OCR bridge and images

The **Tesseract binary** is separate from `pip` (see Prerequisites).

---

## 7. Install and run Ollama

Protocol **Analyze** will fail if Ollama is not running or `qwen3:4b` is missing.

1. Install Ollama from [https://ollama.com/download](https://ollama.com/download) for your OS.
2. Start Ollama (macOS/Windows usually start the app; on Linux follow the Ollama service instructions for your install).
3. Pull the model used by this repo:

```bash
ollama pull qwen3:4b
```

4. Verify the model is listed:

```bash
ollama list
```

You should see `qwen3:4b` in the list.

5. Quick sanity check (optional):

```bash
ollama run qwen3:4b "Say hello in one short sentence."
```

Keep Ollama running in the background while you use **Analyze protocol** in Streamlit.

---

## 8. Launch Streamlit

With your virtual environment **activated** and Ollama running:

```bash
python -m streamlit run app.py
```

Then open:

**http://localhost:8501**

If the terminal prints a different Local URL, use that URL instead.

---

## 9. First-time walkthrough

1. Open the Streamlit app.
2. Upload the demo PDF: **`data/sample_clinical_protocol.pdf`**
3. Click **Analyze protocol**.
4. Wait for analysis to finish. **The first analysis can take several minutes** on a laptop because it runs local `qwen3:4b` extraction passes. Progress text in the UI explains that Q&A will not re-run analysis.
5. Explore the four tabs:

| Tab | What to look for |
|-----|------------------|
| **Protocol Summary** | Title/ID, inclusion/exclusion, visits, procedures, safety, verification statuses, warnings |
| **Coordinator Assistant** | Staff-oriented factual answers with citations |
| **Patient Guide Review** | Click **Generate patient guide**, check flagged items, edit wording, **Approve** |
| **Patient Assistant** | The approved guide, "What does … mean?" answers, safe refusals |

### Example questions

**Coordinator-style**

- What are the inclusion criteria?
- How many visits are scheduled?
- What procedures are performed?

**Patient-style**

- When do I need to visit?
- What does HbA1c mean?
- Am I eligible for this trial?

### Expected safe behavior for eligibility

Questions like **“Am I eligible for this trial?”** should **refuse a personal yes/no decision**. The assistant may still show **general verified criteria** for discussion with the study team, plus clear disclaimers. It must **not** decide enrollment for a specific person.

After analysis succeeds once for an upload, asking more questions should be fast (cached pipeline result).

---

## 10. Run tests

With the virtual environment activated, from the repo root:

```bash
python -m pytest -q
```

162 tests; all should pass (one live-OCR test is skipped if Tesseract is not installed). These tests are mostly deterministic and **mock** Ollama / pipeline calls where needed. You do **not** need a long live Ollama run to execute the pytest suite.

Optional: there are developer scripts under `scripts/` for manual pipeline experiments; those may call Ollama and can be slow. `pytest` only collects `tests/` (see `pytest.ini`).

### Run the evaluation

```bash
python -m evaluation.run_safety_eval                 # no Ollama needed, ~1 s
python -m evaluation.run_extraction_eval --cached    # re-score saved runs, no Ollama
```

Re-running the models on the real protocols needs the PDFs in `data/protocols/`; see [`evaluation/README.md`](evaluation/README.md).

---

## 11. Troubleshooting

| Problem | What to try |
|---------|-------------|
| **Ollama not running / connection errors** | Start the Ollama app/service. Confirm `ollama list` works. Re-run Analyze. |
| **Model not found** | Run `ollama pull qwen3:4b`. The code expects exactly **`qwen3:4b`**. |
| **Missing Python packages** | Activate `.venv`, then `pip install -r requirements.txt`. Confirm `python -V` is 3.12.x. |
| **Wrong Python version** | Recreate the venv with `python3.12 -m venv .venv` (or `py -3.12` on Windows). |
| **OCR / Tesseract issues** | Selectable-text PDFs still work. For scanned pages, install the Tesseract **binary** and ensure it is on your PATH. Missing OCR produces warnings and empty page text—it does **not** invent text. |
| **Port 8501 already in use** | Stop the other Streamlit process, or run `python -m streamlit run app.py --server.port 8502` and open that port. |
| **Analysis is very slow** | Expected for local `qwen3:4b` on CPU. Wait for completion; do not re-click Analyze repeatedly. Q&A after caching is fast. |
| **Windows activation errors** | Use `.\.venv\Scripts\Activate.ps1`. Adjust PowerShell execution policy for **CurrentUser** if scripts are blocked. |
| **Empty / unreadable PDF** | The UI should show a clear error. Try `data/sample_clinical_protocol.pdf` first. |
| **Import errors when running scripts** | Run Streamlit and pytest from the **repo root** with the venv active. For ad-hoc scripts, you may need `PYTHONPATH=.` (macOS/Linux) or equivalent. |

---

## 12. Project structure

Important paths in this repository:

```text
clinical-trial-ai/
├── app.py                      # Streamlit MVP UI
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CONTRIBUTING.md             # Team Git workflow
├── ARCHITECTURE.md             # Higher-level design notes (may include future ideas)
├── TASKS.md                    # Milestone tracking
├── agents/
│   ├── protocol_analysis_agent.py   # Ollama/Qwen extraction (qwen3:4b)
│   ├── coordinator_assistant_agent.py
│   ├── patient_agent.py             # Deterministic patient Q&A + safe refusals
│   └── plain_language_agent.py      # Patient guide: rewrite → check → retry → flag
├── ui/
│   └── patient_guide_tab.py    # Patient Guide Review + approved patient view
├── services/
│   ├── pdf_processor.py        # PDF text + OCR fallback
│   ├── protocol_pipeline.py    # analyze_protocol_pdf()
│   ├── protocol_accuracy.py    # Grounding / completeness checks
│   ├── eligibility_rescue.py   # Conditional eligibility rescue
│   ├── source_verifier.py      # Page verification
│   ├── coordinator_query.py    # Coordinator wrapper
│   └── patient_query.py        # Patient wrapper
├── models/
│   ├── protocol_models.py
│   ├── coordinator_models.py
│   ├── patient_models.py
│   └── plain_language_models.py
├── evaluation/                 # Answer keys, question bank, scoring, results
├── docs/PROJECT_BRIEF.md       # Problem, users, agentic design, Definition of Done
├── EVALUATION.md               # Evaluation results and limitations
├── tests/                      # pytest suite
├── data/
│   └── sample_clinical_protocol.pdf   # Demo protocol (synthetic)
└── scripts/                    # Optional manual smoke scripts (may call Ollama)
```

`ARCHITECTURE.md` also describes **planned** work (FastAPI, database, retrieval) that is not part of this MVP. The **running MVP path** is Streamlit → `analyze_protocol_pdf` → coordinator/patient wrappers as above. SQLite storage is **not** required to run this MVP.

---

## 13. Limitations and safety

- **Educational / research prototype only** — not for clinical decision-making or real patient care.
- **`exact_match` verification** means the extracted statement was found on a source page. It is **not** proof of clinical correctness, completeness, or regulatory adequacy.
- **No medical advice** and **no final eligibility decisions**.
- Demo content uses **synthetic / sample** protocol material (for example `data/sample_clinical_protocol.pdf`). Do not upload real patient-identifying data.
- Analysis depends on a **local Ollama** install and enough machine resources; quality varies with model and PDF layout.
- Coordinator/Patient assistants use **keyword-style deterministic** matching; unusual phrasing may be unsupported (measured: 56% of new phrasings handled as intended; unhandled ones fall back to a safe "can't answer").
- **Long protocols** exceed the model's context window: recall fell to 23% on an 86-page protocol. **Safety sections** are the weakest category. See [`EVALUATION.md`](EVALUATION.md).
- Always read the full protocol and involve qualified clinicians / the study team for real enrollment questions.

---

## 14. Team workflow

We use one GitHub repo and personal development branches (see `CONTRIBUTING.md`):

| Person | Branch |
|--------|--------|
| Rakesh | `dev/rakesh` |
| Raj | `dev/raj` |
| Nam | `dev/nam` |

**`main`** holds reviewed, integrated code. Prefer **pull requests** into `main` instead of pushing commits directly to `main`.

### Suggested beginner flow

```bash
git clone https://github.com/Rakesh-Borra/clinical-trial-ai.git
cd clinical-trial-ai
git fetch origin
git switch main
git pull origin main

# Work on your assigned branch (example for Nam)
git switch dev/nam
# or create a feature branch from your dev branch:
# git switch -c feature/short-description
```

After making changes:

```bash
git status
git add path/to/changed_file.py
git commit -m "feat: short description of what you finished"
git push origin HEAD
```

Then open a **Pull Request** on GitHub from your branch toward `main` (or the branch your team agrees on).

### Before you open a PR

1. Run `python -m pytest -q`
2. Avoid committing secrets (`.env`), real patient data, or large unrelated files
3. Coordinate before editing files owned by another teammate
4. Do not force-push shared branches unless the team explicitly agrees

---

## Team

- **Rakesh — Team Lead:** AI agents, backend, integration, testing
- **Raj — Contributor:** Sample protocols, evaluation, testing, documentation
- **Nam — Contributor:** Plain-language Patient Guide agent, coordinator review & approval UI, glossary, evaluation framework and results, scrambled-PDF/OCR and context-window fixes, documentation

---

## Quick start checklist

1. [ ] Clone the repo
2. [ ] Create/activate Python **3.12** venv
3. [ ] `pip install -r requirements.txt`
4. [ ] Install/start Ollama and `ollama pull qwen3:4b`
5. [ ] (Optional) Install Tesseract for scanned PDFs
6. [ ] `python -m streamlit run app.py`
7. [ ] Upload `data/sample_clinical_protocol.pdf` → **Analyze protocol**
8. [ ] Ask coordinator/patient questions; confirm eligibility refusal is safe
9. [ ] **Patient Guide Review** → Generate → Approve; check the **Patient Assistant** tab shows only approved items
10. [ ] `python -m pytest -q`
