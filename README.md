
# Clinical Trial AI

An Agentic AI-powered clinical trial patient and coordinator assistant.

## Problem
Patients struggle to understand complex clinical trial information. Research coordinators spend time explaining procedures, answering repetitive questions, and managing protocol updates.

## Solution
A web application that converts trial protocols into understandable, source-backed information and provides a patient Q&A assistant, with coordinator review and approval.

## Three AI Agents
1. Protocol Analysis Agent — Extracts trial details from uploaded PDF protocols and identifies changes between versions.
2. Patient Assistant Agent — Generates patient-friendly summaries and answers questions using approved trial information.
3. Verification Agent — Checks generated information against protocol sources and flags uncertain claims for human review.

## Workflow
Upload Protocol → Analyze → Generate Summary → Verify → Coordinator Approval → Patient Portal and Q&A

## Technology
Python 3.12, Streamlit, Ollama, PyMuPDF, Pydantic, SQLite, pytest.

## Safety
Prototype for educational use only. No real patient data. AI output requires qualified human review and does not replace clinical advice or informed consent.

## Team
- **Rakesh — Team Lead:** AI agents, backend development, integration, and testing.
- **Raj — Contributor:** Sample clinical trial protocols, evaluation, testing, and documentation.
- **Nam — Contributor:** Streamlit user interface, patient portal, and usability testing.

## Project Status
Initial repository setup complete. Implementation in progress.
  