# Clinical Trial AI

An AI-assisted prototype that helps clinical trial coordinators turn dense protocol PDFs into **source-linked, coordinator-approved patient information**, and lets patients ask questions answered only from that approved, published content.

**Repository:** [https://github.com/Rakesh-Borra/clinical-trial-ai](https://github.com/Rakesh-Borra/clinical-trial-ai)

This project is an **educational / research prototype**. It does **not** provide medical advice, diagnose conditions, or decide whether any person is eligible for a trial.

**PEAAII AI Applications Hackathon 2026 (Agentic AI).** Start here:

| Document | What it covers |
|---|---|
| [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) | Problem, target users, why this is agentic, workflow diagram, Definition of Done, comparison with existing tools |
| [`EVALUATION.md`](EVALUATION.md) | How we tested it: 3 real ClinicalTrials.gov protocols, 60 patient questions, failure cases, limitations |
| [`evaluation/README.md`](evaluation/README.md) | How to reproduce the evaluation |

### Results at a glance

> Measured on an earlier version of the pipeline, before section classification, table extraction, caching and the conversational Patient Assistant were added. They have not been re-measured; see [`EVALUATION.md`](EVALUATION.md).

| What we measured | Result |
|---|---|
| Risky patient questions (eligibility decisions, medical advice) answered safely | **100%** (25/25, incl. 9 never seen during development) |
| Unsafe plain-language rewrites caught by the guardrail | **100%** (13/13) |
| Extracted statements that are correct (3 real protocols) | **91%** (58/64) |
| Page citations that are correct | **91%** (53/58) |
| Inclusion criteria found | **100%** (14/14) |
| All protocol statements found (recall) | 66% on the two short protocols; **23% on an 86-page protocol** (main limitation) |

---

## 1. Problem and proposed solution

### The problem

Clinical trial protocols are written for regulators, investigators and ethics boards. They are long and full of medical terminology, eligibility criteria, procedures, visit schedules, risks and participant responsibilities. **Patients and prospective participants** often struggle to understand them, and **clinical trial coordinators** must explain this material accurately and in plain language without changing its medical meaning.

Accuracy and traceability matter because a small wording change (a dropped number, an altered threshold, a softened risk) can change what a participant believes they are agreeing to. Every statement shown to a patient therefore needs to be traceable to a protocol page, and a **human coordinator must review it** before patients see it. AI-generated text can omit or distort information, so it is treated as a draft, never as a source of truth.

This prototype **does not replace** the official informed-consent document, the study team, or any clinician.

### The proposed solution

Clinical Trial AI is an AI-assisted workflow. Each capability below is implemented in this repository:

1. Accepts a clinical trial protocol PDF (Streamlit upload).
2. Extracts text (with OCR fallback), page numbers and ruled tables.
3. Classifies sections and filters for patient-relevant content, keeping the original pages intact for verification.
4. Extracts structured study details with a local LLM (`qwen3:4b` through Ollama) and generates plain-language, source-linked patient statements.
5. Verifies extracted and rewritten statements against protocol evidence.
6. Lets a **human coordinator** review, edit, approve, reject, publish and unpublish statements, persisted in SQLite.
7. Lets patients ask questions that are answered only from approved, currently published statements.

> This is an educational / research prototype. It is not clinically validated, not regulatory-compliant, and not production-ready. It does not provide medical advice, diagnose conditions, or decide whether anyone is eligible for a trial.

---

## 2. Three-agent architecture

| Agent | Module | What it does | Uses an LLM? |
|---|---|---|---|
| **1. Protocol Analysis Agent** | `agents/protocol_analysis_agent.py` | Extracts study metadata, eligibility, visits, procedures and safety information from section-aware chunks, with source page numbers | Yes (`qwen3:4b`) |
| **2. Verification Agent** | `services/source_verifier.py`, `services/protocol_accuracy.py`, and the rewrite checks in `agents/plain_language_agent.py` (`check_rewrite`, `reverify_item`) | Checks statements against source pages, grounds metadata, repairs page references, checks eligibility completeness, and checks that plain-language rewrites keep every number and contain no advice | No (deterministic rules) |
| **3. Patient Assistant Agent** | `agents/guide_patient_agent.py` (with `agents/patient_agent.py` for intent patterns, `services/chat_intents.py`, `services/chat_grounding.py`) | Answers patient questions from approved, published statements only | Optional (off by default) |

The **coordinator is a human reviewer** using the coordinator interface; there is no Coordinator AI agent (an earlier deterministic coordinator Q&A module was removed). The plain-language rewrite step (`agents/plain_language_agent.py`) is a helper used between analysis and review: it asks the model for a patient-friendly rewrite and runs the verification checks above.

Everything else (`services/`, `models/`, `ui/`) are helper modules that support these roles; they are not separate agents. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for details.

---

## 3. What is implemented

### PDF processing and analysis
- PDF text extraction with PyMuPDF, page numbers preserved; OCR fallback (`pytesseract` + Tesseract) for scanned or font-scrambled pages
- Ruled-table extraction (rows, columns and values kept) as an extra view, never replacing page text (`CTAI_EXTRACT_TABLES=0` disables it)
- Section classification (`services/section_classifier.py`): ICH M11-style and common protocol headings mapped to purpose, eligibility, visits, procedures, risks, safety, consent, withdrawal, responsibilities, administrative. Protocol amendments are not classified as a separate category.
- Section-aware, token-aware chunking (`services/section_chunker.py`)
- Conservative patient-relevance filtering at block level (`services/patient_relevance.py`): administrative-only sections are skipped, while safety wording found inside administrative sections is retained. All original pages remain available for verification.
- Category-based pass routing: an extraction pass is skipped only when section categories and keyword cues agree; a missing keyword alone never skips a pass
- SQLite cache of completed analyses (`services/analysis_cache.py`), keyed on document text and processing settings
- Completeness audit (`services/completeness_audit.py`) flagging missing or uncertain patient-relevant categories
- Performance instrumentation (`services/performance.py`): stage timings, chunk and model-call counts, token counts when the provider reports them
- Input and model-call limits (320,000 characters, 65 calls, 1,800 s analysis budget by default) with actionable error messages

### Human coordinator
- Review each statement with its source page, edit and save, approve, reject
- Edited statements must be reverified and then approved again before publishing
- Publication validation, patient-view preview, publish and unpublish
- Review state, drafts, publications and action history stored in SQLite (`data/ctai.db`)

### Patient Assistant
- Greeting, thanks and small talk handled without retrieval or model calls, including common misspellings
- Typo correction against the guide vocabulary, intent recognition, follow-up questions using previous answers, short clarification questions when ambiguous
- Retrieval limited to approved, currently published statements; citations show protocol pages
- Grounding validation: cited statements must still be approved and numbers must come from approved text
- Refusals for medical advice and personal eligibility decisions; "ask your study team" replies for topics the guide does not cover (costs, withdrawal, how procedures feel, contacting the team)
- Optional model-based rewording (`CTAI_CHAT_LLM=1`, off by default) that is rejected and replaced by the deterministic answer if it adds numbers, advice or URLs

### Evaluation
- Safety question bank and extraction answer keys under `evaluation/` (see [`EVALUATION.md`](EVALUATION.md))

---

## 4. Workflow

```text
Protocol PDF (Streamlit upload)
  → PDF extraction (text, OCR fallback, tables, page numbers)
  → Section classification + chunking + patient-relevance filtering
  → Protocol analysis (Agent 1, Ollama qwen3:4b; cached in SQLite)
  → Source verification and accuracy checks (Agent 2)
  → Plain-language statement generation + rewrite checks
  → Human coordinator review: edit, reverify, approve / reject
  → Publish (validated) → SQLite publication snapshot
  → Patient Assistant (Agent 3) answers from published statements only
```

| Stage | Entry point |
|-------|-------------|
| Analyze PDF | `services/protocol_pipeline.py` → `analyze_protocol_pdf()` |
| Patient statements | `agents/plain_language_agent.py` → `generate_patient_guide()` |
| Review / publish | `app.py`, `ui/patient_guide_tab.py`, `services/guide_review.py`, `services/review_store.py` |
| Patient chat | `ui/patient_view.py` → `agents/guide_patient_agent.py` → `answer_patient_from_guide()` |

The model context window is 16,384 tokens by default. On a computer with 8 GB of RAM, lower it with `export CTAI_NUM_CTX=8192`.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `CTAI_MODEL_PROVIDER` | `ollama` | `ollama` (supported) or `openrouter` (see below) |
| `CTAI_DB_PATH` | `data/ctai.db` | SQLite file for drafts, publications, history, analysis cache |
| `CTAI_NUM_CTX` | `16384` | Ollama context window |
| `CTAI_EXTRACTION_NUM_PREDICT` | `6144` | Output token cap for extraction |
| `CTAI_EXTRACTION_THINK` | `0` | Qwen3 thinking mode for extraction |
| `CTAI_PATIENT_RELEVANCE_FILTER` | `1` | Enable patient-relevance filtering |
| `CTAI_EXTRACT_TABLES` | `1` | Enable table extraction |
| `CTAI_PROTOCOL_CHUNK_TOKENS` / `CTAI_PROTOCOL_CHUNK_CHARS` | `2500` / `8000` | Chunk size limits |
| `CTAI_PROTOCOL_MAX_INPUT_CHARS` | `320000` | Maximum protocol text accepted |
| `CTAI_PROTOCOL_MAX_CALLS` | `65` | Maximum model calls per analysis |
| `CTAI_ANALYSIS_MAX_SECONDS` | `1800` | Analysis time budget |
| `CTAI_CHAT_LLM` | `0` | Optional model rewording of chat answers |

### OpenRouter (planned, not yet validated)

`services/model_provider.py` contains an opt-in OpenRouter code path (`CTAI_MODEL_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, `CTAI_OPENROUTER_MODEL`). It has **not** been validated end-to-end and the full integration is planned future work. **Ollama with `qwen3:4b` is the supported configuration.** If you try it, supply the key via the process environment only (the app does not load `.env`), and confirm you are authorised to send protocol text to a cloud service. Selecting OpenRouter does not change the input or call caps.

---

## 5. Prerequisites

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

## 6. Clone the repository

```bash
git clone https://github.com/Rakesh-Borra/clinical-trial-ai.git
cd clinical-trial-ai
```

---

## 7. Virtual environment and dependencies

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

## 8. Install and run Ollama

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

## 9. Launch Streamlit

With your virtual environment **activated** and Ollama running:

```bash
python -m streamlit run app.py
```

Then open:

**http://localhost:8501**

If the terminal prints a different Local URL, use that URL instead.

---

## 9b. Accounts and sign-in

Sign-in uses real accounts stored in the same SQLite file as the review data (`data/ctai.db`, or `CTAI_DB_PATH`). Passwords are hashed with Argon2id; sessions last 8 hours or until you log out; five failed sign-ins lock an account for 15 minutes.

- **Patients** register themselves on the welcome page (**Create account**, Patient card).
- **Coordinators** choose the Coordinator card and enter an administrator-issued, single-use invitation code (hashed in SQLite, expires in 72 h). Administrators create codes and accounts on the machine running the app (passwords are prompted for, never passed on the command line):

  ```bash
  python -m scripts.manage_users create-invite
  python -m scripts.manage_users create-coordinator --email you@example.org --name "Your Name"
  python -m scripts.manage_users promote --email existing@example.org --role patient   # patient -> coordinator
  python -m scripts.manage_users reset-password --email you@example.org --role coordinator
  ```

- **One email, two accounts:** the same email may hold one Patient and one Coordinator account (unique per email + role). Each has its own password, sessions and profile. Sign-in, password reset and admin commands use the selected role; a wrong role gives the same generic error as a wrong password.
- **Classroom demo mode (local only):** set `CTAI_DEMO_MODE=true` and the coordinator code `0000` is accepted. This is demo access, not production security; never enable it in a shared or deployed environment. Without it, `0000` is rejected.
- **Password recovery** emails a single-use reset code (valid 30 minutes). It needs SMTP settings in the environment: `CTAI_SMTP_HOST`, `CTAI_SMTP_FROM` (required), `CTAI_SMTP_PORT` (default 587), `CTAI_SMTP_USER`, `CTAI_SMTP_PASSWORD`, and `CTAI_APP_URL` (adds a clickable link). Without them the app says recovery is unavailable; an administrator can use `reset-password` instead. Keep these values in your environment or an untracked `.env`, never in Git.
- There is no "remember me": a session ends when the browser tab closes or you log out.

---

## 10. First-time walkthrough

1. Open the Streamlit app. You land on the **Welcome to Clinical Trial AI** page. Create the first coordinator account with the command above, then sign in.
3. **Step 1 · Upload**: upload any protocol PDF you choose (none is bundled), pick the document type, click **Analyze protocol**. **Analysis can take several minutes** with local `qwen3:4b`; a completed analysis of the same text and settings is reused from the SQLite cache.
4. Walk through the three steps:

| Step | What to look for |
|------|------------------|
| **1 · Upload** | Choose the document type; analyze once |
| **2 · Summary** | Title/ID, inclusion/exclusion, visits, procedures, safety, warnings, completeness notes, with page numbers |
| **3 · Review** | Generate patient statements; compare each with its source excerpt and page; edit and save, **reverify** edited text, approve or reject; **preview as a patient**; publish or unpublish |

   Review states are **Pending Review**, **Approved**, **Rejected**, **Needs Reverification** and **Published**. Publishing is blocked while any statement is pending, entirely rejected, unverified or edited-but-not-reverified.
5. Log out and sign in as a **patient** (register on the welcome page first): the published guide is on the left and the chat assistant on the right. Patients see only published content. Withdrawing (unpublishing) removes it from the Patient Assistant.

### Example questions

**Patient-style** (after a guide is published)

- When do I need to visit?
- What does HbA1c mean?
- Am I eligble? (misspellings are tolerated)
- What about the second one? (follow-up)
- Am I eligible for this trial? (safely refused as a personal decision)

### Expected safe behavior for eligibility

Questions like **“Am I eligible for this trial?”** should **refuse a personal yes/no decision**. The assistant may still show **general verified criteria** for discussion with the study team, plus clear disclaimers. It must **not** decide enrollment for a specific person.

Patient answers are deterministic by default and make no model call.

---

## 11. Run tests

With the virtual environment activated, from the repo root:

```bash
python -m pytest -q
```

The suite (492 tests at the last run) is deterministic and **mocks** Ollama / pipeline calls where needed. You do **not** need a long live Ollama run to execute the pytest suite.

Optional: there are developer scripts under `scripts/` for manual pipeline experiments; those may call Ollama and can be slow. `pytest` only collects `tests/` (see `pytest.ini`).

### Run the evaluation

```bash
python -m evaluation.run_safety_eval                 # no Ollama needed, ~1 s
python -m evaluation.run_extraction_eval --cached    # re-score saved runs, no Ollama
```

Re-running the models on the real protocols needs the protocol PDFs stored outside the repo (paths are set in the ground-truth JSON); see [`evaluation/README.md`](evaluation/README.md).

---

## 12. Troubleshooting

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
| **Empty / unreadable PDF** | The UI should show a clear error. Try a different text-based protocol PDF. |
| **Import errors when running scripts** | Run Streamlit and pytest from the **repo root** with the venv active. For ad-hoc scripts, you may need `PYTHONPATH=.` (macOS/Linux) or equivalent. |

---

## 13. Project structure

Important paths in this repository:

```text
clinical-trial-ai/
├── app.py                          # Streamlit app: coordinator workflow, publication, role routing
├── requirements.txt
├── agents/
│   ├── protocol_analysis_agent.py  # Agent 1: chunked extraction, pass routing (qwen3:4b)
│   ├── plain_language_agent.py     # Plain-language rewrite + checks, edit/reverify/approve/reject
│   ├── guide_patient_agent.py      # Agent 3: conversational answers from published guide
│   └── patient_agent.py            # Intent patterns, refusals, legacy deterministic Q&A
├── services/
│   ├── pdf_processor.py            # Text, OCR fallback, table extraction, page numbers
│   ├── section_classifier.py       # Heading/content section categories
│   ├── patient_relevance.py        # Block-level patient-relevance filter
│   ├── section_chunker.py / protocol_chunker.py  # Section- and token-aware chunks
│   ├── protocol_pipeline.py        # analyze_protocol_pdf() orchestration
│   ├── analysis_cache.py           # SQLite analysis cache
│   ├── analysis_jobs.py / guide_jobs.py  # Background job managers
│   ├── analysis_progress.py        # Progress reporting
│   ├── performance.py              # Timing and token instrumentation
│   ├── completeness_audit.py       # Missing/uncertain category audit
│   ├── source_verifier.py / protocol_accuracy.py  # Verification (Agent 2)
│   ├── eligibility_rescue.py       # Conditional eligibility re-extraction
│   ├── guide_review.py             # Review states, publish validation, patient-visible snapshot
│   ├── review_store.py             # SQLite drafts, publications, history
│   ├── chat_intents.py / chat_grounding.py  # Chat intents, typo handling, grounding checks
│   ├── model_provider.py           # Model-provider abstraction (Ollama; OpenRouter path unvalidated)
│   └── patient_query.py            # Wrapper over the legacy deterministic Q&A
├── models/                         # Pydantic models (protocol, patient, plain-language guide)
├── ui/                             # theme, landing/sign-in, patient view, review tab, chat store
├── evaluation/                     # Answer keys, question bank, scoring, results
├── tests/                          # pytest suite
├── data/                           # ctai.db is created locally (git-ignored)
├── docs/PROJECT_BRIEF.md, EVALUATION.md, ARCHITECTURE.md, CONTRIBUTING.md, TASKS.md
└── scripts/                        # Optional manual smoke scripts (may call Ollama)
```

---

## 14. Limitations and roadmap

### Known limitations
- **Educational / research prototype only.** Not clinically validated, not regulatory-compliant, not production-ready, and not for clinical decision-making or real patient care.
- `exact_match` verification means the statement text was found on a source page. It is **not** proof of clinical correctness or completeness.
- **No medical advice and no personal eligibility decisions.**
- Demo content is synthetic/sample material. Do not upload real patient-identifying data.
- Analysis depends on a local Ollama install and machine resources; local analysis of long protocols is slow, and recall dropped to 23% on an 86-page protocol in the earlier evaluation. Safety sections were the weakest category. Inputs over 320,000 characters or 65 model calls are rejected with an actionable message. No speed-up from the latest optimisations has been measured on a full protocol.
- Section/heading detection is heuristic; table extraction handles ruled tables only and its text appears alongside the page text in model input.
- Patient answers are mostly the approved statements themselves; unusual phrasing may get a safe "cannot answer" reply. The assistant is conservative about topics such as blood collection.
- Authentication is a lightweight, single-server design: no multi-factor sign-in, no "remember me", and the login rate limit is per email (not per IP).
- Evaluation figures in this README predate the latest pipeline changes.

### Planned improvements
- OpenRouter integration and validation
- Further performance optimisation, measured on full-length protocols
- Additional evaluation and reliability work

Always read the full protocol and involve qualified clinicians and the study team for real enrollment questions.

---

## 15. Team workflow

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
7. [ ] Upload a protocol PDF → **Analyze protocol**
8. [ ] **Review** → Generate → Approve; open **Preview as a patient** and check only approved items appear
9. [ ] `python -m pytest -q`

## Opt-in benchmarking

No benchmark PDFs are bundled. Keep them outside the repo.

- Ollama-free preview (extraction, relevance filter, chunk and call estimates):
  `PYTHONPATH=. python scripts/benchmark_preview.py "<pdf folder>" out.json`
- Timing log while using the normal UI: set `CTAI_BENCHMARK_LOG=<path>.benchmark.jsonl` before
  starting the app (unset = no logging, no behavior change). Summarize with
  `python -m services.benchmark_log <path>`.
