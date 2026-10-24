# Clinical Trial AI — Team Development Board

Status at hackathon submission (PEAAII AI Applications Hackathon, Oct 2026).

## Status Legend
- [x] Done in this MVP
- [ ] Not done (planned / future work)

## Protocol processing & AI agents
- [x] PDF extraction with page-level references
- [x] OCR fallback for scanned pages
- [x] Detection of scrambled PDF text (broken fonts) with OCR fallback
- [x] Protocol Analysis Agent (focused extraction passes, qwen3:4b via Ollama)
- [x] Eligibility rescue pass when criteria look incomplete
- [x] Source-page verification and accuracy checks
- [x] Section classification, section-aware chunking, patient-relevance filtering, table extraction
- [x] Analysis caching (SQLite), completeness audit and performance instrumentation
- [x] Human coordinator review (edit, reverify, approve, reject, publish, unpublish) persisted in SQLite
- [x] Patient Assistant (conversational, typo-tolerant, grounded in published statements; safe refusals for eligibility and medical advice)
- [x] Plain-Language Agent: rewrite → self-check → retry → flag for review
- [x] Medical-word glossary ("What does HbA1c mean?")
- [x] Section-by-section processing for long protocols (implemented; not re-measured on full-length protocols)
- [ ] Safety-information rescue pass (weakest category, see EVALUATION.md)
- [ ] LLM intent classifier for unseen question phrasings
- [ ] RAG / vector search
- [ ] Protocol version comparison
- [ ] OpenRouter integration and validation

## User interface (Streamlit)
- [x] Protocol upload and analysis
- [x] Protocol summary with verification status and warnings
- [x] Patient guide review and approval (coordinator)
- [x] Patient view showing approved content only
- [ ] Trial list / multiple trials
- [ ] Role-based login (coordinator vs patient)

## Evaluation & QA
- [x] Public protocols from ClinicalTrials.gov (3) with hand-checked answer keys (144 statements)
- [x] Patient question bank (60 questions, incl. 18 held-out)
- [x] Guardrail test cases (24 rewrites)
- [x] Extraction accuracy, page accuracy and failure-case report (EVALUATION.md)
- [x] Automated test suite (492 tests at last run)

## Platform (future work)
- [ ] Backend API (FastAPI)
- [ ] Database and protocol versioning
- [ ] Deployment and monitoring

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
