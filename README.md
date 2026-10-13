
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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OCR fallback (scanned PDFs)

Selectable-text PDFs work with PyMuPDF alone. Pages with no extractable
text are rendered and processed with local OCR (pytesseract + Tesseract).

1. Install Python OCR packages (included in `requirements.txt`):

```bash
pip install pytesseract Pillow
```

2. Install the Tesseract binary on your machine:

```bash
# macOS
brew install tesseract

# Debian/Ubuntu
sudo apt-get install tesseract-ocr
```

If Tesseract or the Python OCR packages are missing, scanned pages stay
empty and a clear warning is recorded. The extractor never invents text.

## Safety
Prototype for educational use only. No real patient data. AI output requires qualified human review and does not replace clinical advice or informed consent.

## Team
- **Rakesh — Team Lead:** AI agents, backend development, integration, and testing.
- **Raj — Contributor:** Sample clinical trial protocols, evaluation, testing, and documentation.
- **Nam — Contributor:** Streamlit user interface, patient portal, and usability testing.

## Project Status
Initial repository setup complete. Implementation in progress.
  