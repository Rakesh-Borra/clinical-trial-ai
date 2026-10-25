# Clinical Trial AI — Team Development Board

Status for the CS697 / PEAAII educational prototype.

## Status Legend
- [x] Done in this MVP
- [ ] Not done (planned / future work)

## Protocol processing & AI agents
- [x] PDF extraction with page-level references
- [x] OCR fallback for scanned pages
- [x] Detection of scrambled PDF text (broken fonts) with OCR fallback
- [x] Protocol Analysis Agent (OpenRouter or Ollama via `services/model_provider.py`)
- [x] Eligibility rescue pass when criteria look incomplete
- [x] Source-page verification and accuracy checks (deterministic safety layer)
- [x] Section classification, section-aware chunking, patient-relevance filtering, table extraction
- [x] Analysis caching (SQLite), completeness audit and performance instrumentation
- [x] Plain-Language Patient Guide Agent: rewrite → self-check → retry → flag for review
- [x] Medical-word glossary ("What does HbA1c mean?")
- [x] Human coordinator review (edit, reverify, approve, reject, publish, unpublish) persisted in SQLite
- [x] Patient Chat (conversational, typo-tolerant, grounded in published statements; safe refusals)
- [x] Protocol amendment comparison (V1/V2) with source-backed verification
- [x] OpenRouter integration (manual `LLM_PROVIDER=openrouter`; no automatic Ollama fallback)
- [x] Active-version publication (patients see only the current published version per study)
- [x] Section-by-section processing for long protocols (implemented; not re-measured on full-length protocols)
- [ ] Safety-information rescue pass (weakest category, see EVALUATION.md)
- [ ] LLM intent classifier for unseen question phrasings
- [ ] Vector RAG over the raw protocol (patient chat already uses BM25 over the approved guide)

## User interface (Streamlit)
- [x] Protocol upload and analysis
- [x] Protocol summary with verification status and warnings
- [x] Patient guide review and approval (coordinator)
- [x] Patient view showing approved content only
- [x] Study/version list for coordinators (multiple trials and amendments)
- [x] Role-based login (coordinator vs patient)
- [ ] FastAPI backend / separate API service

## Evaluation & QA
- [x] Public protocols from ClinicalTrials.gov (3) with hand-checked answer keys
- [x] Patient question bank and guardrail cases under `evaluation/`
- [x] Extraction accuracy, page accuracy and failure-case report (EVALUATION.md)
- [x] Automated offline pytest suite

## Platform (future work)
- [ ] Backend API (FastAPI)
- [ ] Deployment and monitoring
- [ ] Broader re-measurement on full-length commercial protocols

## Collaboration Rules
1. Work primarily in your assigned folders.
2. Use your own development branch.
3. Do not commit secrets or real patient data.
4. Test changes before submitting a pull request.
5. Merge completed features into main after review.
6. Update this task board after meaningful milestones.

## Safety
This is a research prototype. AI outputs require
appropriate human review and must not independently
provide diagnosis, eligibility decisions, or medical advice.
