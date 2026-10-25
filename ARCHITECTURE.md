# Clinical Trial AI — System Architecture

Educational / research prototype. Not clinically validated or production-ready.

## 1. Overview

A Streamlit app (`app.py`) helps a **human coordinator** turn a protocol PDF into reviewed, source-linked patient information. Patients see only content that was approved and published. Persistence is SQLite (`data/ctai.db`, git-ignored). There is no separate backend API.

## 2. Three agents + verification layer

| Role | Implementation | LLM? |
|---|---|---|
| **Agent 1 — Protocol Analysis** | `agents/protocol_analysis_agent.py` | Yes |
| **Agent 2 — Plain-Language Patient Guide** | `agents/plain_language_agent.py` | Yes (rewrites); edit/approve helpers are deterministic |
| **Agent 3 — Patient Chat** | `agents/guide_patient_agent.py` (+ `agents/patient_agent.py` intent helpers, BM25 retrieval, chat grounding) | Optional (`CTAI_CHAT_LLM=1`, default off) |
| **Source verification (not an agent)** | `services/source_verifier.py`, `services/protocol_accuracy.py`, rewrite checks in Agent 2 | No |

The **coordinator is a human reviewer**, not an AI agent.

**Agent 1** extracts study metadata, eligibility, visits, procedures and safety from section-aware chunks, with source pages.  
**Agent 2** rewrites each fact into plain language, checks numbers/safety, and supplies edit / reverify / approve / reject for the review UI.  
**Agent 3** answers only from the **active published** guide.  
**Verification** checks excerpts against cited pages and blocks unsafe or unsupported rewrites before publication.

## 3. End-to-end workflow

```text
Protocol PDF (upload)
  → PDF extraction + section classification / chunking
  → Protocol analysis (Agent 1) — cached in SQLite
  → Source verification (safety layer)
  → Plain-language generation (Agent 2)
  → Human coordinator review
  → Publish (one current version per study)
  → Patient guide + Patient Chat (Agent 3)
```

Pipeline entry: `services/protocol_pipeline.py` (`analyze_protocol_pdf()`).

## 4. Model providers

`services/model_provider.py` (`chat_completion`) serves Agents 1–2 and optional chat rewording.

| Setting | Behavior |
|---|---|
| `LLM_PROVIDER=ollama` (default if unset) | Local Ollama; agents request `qwen3:4b` |
| `LLM_PROVIDER=openrouter` | OpenRouter cloud; needs `OPENROUTER_API_KEY`; default model `openai/gpt-4.1-mini` |

**No automatic fallback** between providers. If the selected provider fails, the error is shown. Patient Chat uses the same provider only when `CTAI_CHAT_LLM=1`.

## 5. Versions and publication

Each saved protocol is a **version** of a **study** (`services/version_store.py`). Publishing makes that version the study’s one current patient-visible snapshot; earlier published versions become superseded history. Patients see only current published guides. Coordinators keep all versions for review and V1/V2 comparison (`services/version_compare.py`).

## 6. Patient Chat (summary)

1. Social intents without retrieval.  
2. Spelling / phrasing correction against the published guide.  
3. Follow-up context without inventing facts.  
4. BM25 retrieval over approved items only.  
5. Refusals for medical advice and personal eligibility decisions.  
6. Grounding validation on citations and numbers.  
7. Optional naturalization discarded if unsafe or ungrounded.

## 7. Safety

- Nothing reaches patients without coordinator approval and publish.  
- No medical advice, diagnosis, or personal eligibility decisions.  
- Do not use real patient data.  
- Prefer omission over invention when evidence is weak.

## 8. Not built

- FastAPI backend and public API endpoints  
- Vector store over the raw protocol  
- Production deployment / monitoring  
- Safety-information rescue pass (see EVALUATION.md)
