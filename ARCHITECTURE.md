# Clinical Trial AI — System Architecture

This document describes what is **implemented in the repository today**. Items that are not built are listed explicitly in [§11](#11-planned-and-not-built). The project is an educational / research prototype: it is not clinically validated, not regulatory-compliant and not production-ready.

## 1. Overview

Clinical Trial AI helps a **human clinical trial coordinator** turn a protocol PDF into reviewed, source-linked patient information, and lets patients ask questions answered only from content the coordinator approved and published. It is a single Streamlit application (`app.py`) with SQLite persistence and a local LLM (`qwen3:4b` through Ollama). There is no separate backend service.

## 2. The three agents

| Agent | Implementation | LLM? |
|---|---|---|
| **1. Protocol Analysis Agent** | `agents/protocol_analysis_agent.py` | Yes (Ollama `qwen3:4b`) |
| **2. Verification Agent** | `services/source_verifier.py`, `services/protocol_accuracy.py`, and the rewrite checks in `agents/plain_language_agent.py` | No (deterministic) |
| **3. Patient Assistant Agent** | `agents/guide_patient_agent.py`, supported by `agents/patient_agent.py`, `services/chat_intents.py`, `services/chat_grounding.py` | Optional, off by default |

The **coordinator is a human reviewer**, not an AI agent. An earlier deterministic "Coordinator Assistant" module was removed and is not planned.

**Agent 1** extracts study metadata, eligibility criteria, visits, procedures and safety information from section-aware chunks, recording a source page for each item. It may run a conditional eligibility re-extraction (`services/eligibility_rescue.py`).

**Agent 2** checks that extracted statements are present on their cited source pages (`verify_source_pages`), grounds title/protocol ID, repairs page references and checks eligibility completeness (`protocol_accuracy.py`). For generated patient text, `check_rewrite` and `reverify_item` confirm numbers are preserved and that no advice or eligibility-decision wording appears. A status of `exact_match` means the text was found on the page; it does not prove clinical correctness.

**Agent 3** answers patient questions using only approved, published statements (see §7).

**Helper, not an agent:** `agents/plain_language_agent.py` asks the model for a plain-language rewrite of each statement, runs the Agent 2 checks, retries once with feedback, and otherwise keeps the original wording flagged for review. It also provides the edit / reverify / approve / reject operations used by the review UI.

## 3. End-to-end workflow

```text
Protocol PDF (upload)
  → PDF extraction                       services/pdf_processor.py
  → Section classification + chunking    section_classifier / patient_relevance / section_chunker
  → Protocol analysis (Agent 1)          agents/protocol_analysis_agent.py   (cached in SQLite)
  → Source verification (Agent 2)        source_verifier / protocol_accuracy
  → Plain-language generation + checks   agents/plain_language_agent.py
  → Human coordinator review             ui/patient_guide_tab.py, services/guide_review.py
  → Validated publication                app.py → services/review_store.py (SQLite)
  → Patient Assistant (Agent 3)          ui/patient_view.py → agents/guide_patient_agent.py
```

Orchestration of the first four stages is in `services/protocol_pipeline.py` (`analyze_protocol_pdf()`), which returns the analysis, verification rows, warnings, relevance summary, performance metrics and completeness audit.

## 4. PDF processing and analysis

- **Extraction** (`pdf_processor.py`): PyMuPDF text per page with page numbers. Pages that are image-only, suspiciously incomplete or garbled by a font problem fall back to local OCR (`pytesseract` + the Tesseract binary), if available. Ruled tables are extracted via PyMuPDF `find_tables` and added to each page dict as `tables`; page `text` is never altered (`CTAI_EXTRACT_TABLES=0` disables).
- **Section classification** (`section_classifier.py`): heading patterns, including ICH M11-style headings, map to purpose, eligibility, visits, procedures, risks, safety, consent, withdrawal, responsibilities or administrative. Weak content cues are used when headings do not decide it. Protocol amendments are not a separate category.
- **Patient-relevance filtering** (`patient_relevance.py`): operates on blocks across **all** original pages. Administrative-only blocks are skipped; safety wording inside administrative sections is retained (the heading and the safety paragraphs); tables are kept verbatim; uncertain blocks are kept. The original PDF text remains the basis for verification and citations. No LLM is used for filtering. `CTAI_PATIENT_RELEVANCE_FILTER=0` disables it.
- **Chunking** (`section_chunker.py`, `protocol_chunker.py`): chunks follow sections and paragraphs under ~4–6k-character / token limits, split oversized sections and tables (with limited overlap and repeated table headers), record source-block ids, categories and page ranges, and refuse silent truncation via coverage checks. Intentional covered splits are informational warnings only; missing blocks, failed batches, and verification failures still block completeness/publication.
- **Analysis** (`protocol_analysis_agent.py`): two extraction passes (metadata/eligibility/safety, and visits/procedures) per batch, with bounded retries and optional per-pass cache reuse, then structured merge/de-duplication that keeps clinically distinct findings and flags conflicts. `route_passes` skips a pass only when the chunk's heading categories and keyword cues agree; otherwise both run. Qwen3 thinking is off by default and output tokens are capped (`CTAI_EXTRACTION_THINK`, `CTAI_EXTRACTION_NUM_PREDICT`). Failed batches mark analysis incomplete so publication stays blocked.
- **Limits**: overall patient-relevant text over `CTAI_PROTOCOL_MAX_INPUT_CHARS` (2,000,000), more than `CTAI_PROTOCOL_MAX_CALLS` (450) model calls, a single batch over `CTAI_PROTOCOL_CHUNK_CHARS`, or more than `CTAI_ANALYSIS_MAX_SECONDS` (1,800) produces an actionable error rather than running indefinitely or silently dropping content.
- **Caching** (`analysis_cache.py`): completed, failure-free analyses are stored in the `analysis_cache` SQLite table; successful chunk passes can be reused from `chunk_pass_cache` after a mid-run failure. Keys hash the exact text sent to the model plus provider, model, chunking, routing, think/token settings and an analysis version. Verification and accuracy checks always run afresh on the original PDF.
- **Completeness audit** (`completeness_audit.py`): reports per-category status and lists missing or uncertain patient-relevant categories; the UI shows a caption when attention is needed.
- **Performance instrumentation** (`performance.py`): stage timings, chunk and model-call counts, and prompt/output token counts when the provider reports them (Ollama does); otherwise they are left unset.
- **Background jobs** (`analysis_jobs.py`, `guide_jobs.py`, `analysis_progress.py`) run analysis and guide generation off the Streamlit thread with progress reporting.

## 5. Human coordinator review and publication

- Statements are shown with the original protocol excerpt/page and the generated text (`ui/patient_guide_tab.py`).
- Review states (`services/guide_review.py`): Pending Review, Approved, Rejected, Needs Reverification, and Published/Withdrawn at protocol level.
- Editing a statement saves the new text, marks it unverified, and requires **reverification** against the protocol and then explicit approval; the original citation and page are preserved.
- **Publication validation** (`decision_block_reason`): blocked while statements are pending or awaiting reverification, when everything is rejected, when an edited statement fails verification, or when a mandatory section ("Who may not be able to join", "Safety information") is entirely rejected. Rejected statements are excluded from the published snapshot (`build_published_snapshot`).
- Publishing writes a snapshot to SQLite; **Unpublish** marks it withdrawn so it is no longer available to patients. Action history is logged.
- A preview of the patient guide is available before publishing.

## 6. Data storage

SQLite at `data/ctai.db` (override with `CTAI_DB_PATH`; git-ignored). Created automatically by `services/review_store.py` and `services/analysis_cache.py`:

| Table | Purpose |
|---|---|
| `drafts` | Latest coordinator working copy per protocol |
| `publications` | Published/withdrawn guide snapshots (JSON) |
| `history` | Review and publication action log |
| `analysis_cache` | Cached model analyses |

Uploaded PDFs are written to a temporary file for processing and cleaned up. Streamlit session state holds the working analysis and guide objects; review decisions are persisted to SQLite and restored on restart. Pydantic models are in `models/`.

## 7. Patient Assistant

Flow in `agents/guide_patient_agent.py` (`answer_patient_from_guide`):

1. **Social intents** (greetings, thanks, goodbyes, acknowledgements, with typo tolerance) are answered without retrieval or a model call.
2. **Spelling and phrasing correction** against the vocabulary of the published guide (`chat_intents.py`).
3. **Context:** short follow-ups ("what about the second one", "how long is it") reuse the previous cited answer only when the follow-up's own words relate to the guide; ambiguous inputs get a short clarification question.
4. **Intent and retrieval:** answers are built from approved, non-rejected statements of the currently published guide only; items not approved are reported as under review. Out-of-scope topics (costs, withdrawal, how procedures feel, contacting the team) get an "ask your study team" reply with no citations.
5. **Safety:** requests for medical advice or a personal eligibility decision are refused (general approved criteria may be shown for discussion with the study team).
6. **Grounding validation** (`chat_grounding.py`): every citation must be a currently approved statement and every number must appear in approved text or cited page numbers; otherwise a safe reply is returned.
7. **Optional naturalization** (`CTAI_CHAT_LLM=1`, default off): the model may reword the deterministic answer; the result is discarded in favour of the deterministic text if it is long, adds numbers, includes advice/eligibility phrasing or URLs, or the call fails. Refusals and social replies are never rewritten.

The patient interface (`ui/patient_view.py`) loads the published guide (`patient_visible_guide` removes any rejected item defensively) and calls the assistant. Unpublishing removes it from the assistant's data.

## 8. Model provider

`services/model_provider.py` exposes `chat_completion` (schema-constrained JSON). Ollama is the supported provider, with model name `qwen3:4b` set in the agent modules. `LLM_PROVIDER=openrouter` switches every agent to OpenRouter (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, loaded from `.env`); it is never a silent fallback and has not yet been run live. Provider choice does not change the input/call limits.

## 9. Configuration

See the environment-variable table in [`README.md`](README.md#4-workflow). Other variables: `CTAI_MODEL_TIMEOUT_SECONDS`, `CTAI_LONG_PROTOCOL_THRESHOLD_CHARS`, `CTAI_PROTOCOL_RESCUE_MAX_CHARS`, `CTAI_OPENROUTER_MAX_OUTPUT_TOKENS`.

## 10. Safety and privacy

- Patients see only coordinator-approved, published content, with protocol page citations.
- Nothing is published automatically; publication requires validation and an explicit coordinator action.
- No medical advice, diagnosis, treatment recommendation or personal eligibility decision.
- Uncertain or unverified statements are flagged for the coordinator; the system errs toward omission rather than invention (see [`EVALUATION.md`](EVALUATION.md)).
- Sign-in is a demo role selector without authentication. Do not use real patient data.
- With Ollama, protocol text stays on the local machine.

## 11. Planned and not built

- FastAPI backend and API endpoints (previously proposed: `/trials`, `/analyze`, `/approve`, `/chat`)
- Real authentication and per-user roles
- Vector store / retrieval over the protocol
- Protocol version management
- Validated OpenRouter integration
- Further performance optimisation measured on full-length protocols and broader evaluation

## 12. Development principles

1. Preserve protocol source references.
2. Keep human review in front of anything patient-facing.
3. Write tests for important behaviour; mock model calls.
4. Keep secrets and private data out of Git.
5. Build incrementally.
