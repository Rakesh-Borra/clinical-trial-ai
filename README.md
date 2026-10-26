# Clinical Trial AI

**An AI-powered Clinical Trial Assistant that helps patients understand complex clinical trial protocols through simplified information and an interactive, source-grounded chatbot.**

Built as part of **CS697 — Industry AI Applications Lab** at the University of Massachusetts Boston.

## Problem Statement

Clinical trial protocols often contain 100+ pages of complex medical terminology, eligibility requirements, procedures, and safety information.

Patients may struggle to understand these documents, while clinical trial coordinators spend considerable time explaining study details and communicating protocol updates.

When protocols are amended, important changes can also be difficult to identify and communicate clearly.

## Our Solution

Clinical Trial AI transforms complex clinical trial PDFs into easy-to-understand, source-backed patient guides.

Our system uses three AI agents to analyze protocols, simplify medical information, and power an interactive **Patient Chat Assistant**.

Before any information becomes available to patients, a clinical trial coordinator must review and approve it.

Patients can then read their approved study guide and ask questions about their clinical trial in everyday language.

## Key Features

- **AI Patient Chat Assistant:** Patients ask questions about study visits, procedures, eligibility requirements, and safety information. Answers are grounded in the approved, published study guide.
- **Protocol Analysis:** Extracts important information from clinical trial PDFs with source-page references.
- **Plain-Language Guides:** Converts technical medical information into understandable explanations.
- **Protocol Amendment Detection:** Compares V1 and V2 protocols and highlights patient-relevant changes.
- **Source Verification:** Checks generated information against the original protocol.
- **Coordinator Portal:** Allows coordinators to review, approve, and publish patient information.
- **Patient Portal:** Provides access to the current approved study guide and AI Chat Assistant.

## Three-Agent Architecture

| Agent | Responsibility |
|---|---|
| Protocol Analysis Agent | Extracts structured information from clinical trial PDFs |
| Plain-Language Guide Agent | Simplifies medical terminology for patients |
| Patient Chat Agent | Answers patient questions using approved protocol information |

A separate source-verification layer checks supporting evidence before publication.

### How It Works

**Upload Protocol PDF → AI Analysis → Plain-Language Guide → Source Verification → Coordinator Approval → Patient Portal + AI Chat Assistant**

When a protocol changes, coordinators can compare versions, review amendments, and publish the updated information.

## Getting Started

**Requirements:** Python **3.12** preferred (3.9+ works). Dependencies are pinned for **Streamlit 1.50.0**.

**1. Clone the repository**

```bash
git clone https://github.com/Rakesh-Borra/clinical-trial-ai.git
cd clinical-trial-ai
```

**2. Install dependencies**

```bash
python3.12 -m venv .venv   # or: python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Configure the AI model and local demo mode**

```bash
cp .env.example .env
```

Edit `.env`:

- **OpenRouter:** `LLM_PROVIDER=openrouter` and set `OPENROUTER_API_KEY` (never commit this file).
- **Ollama:** `LLM_PROVIDER=ollama`, install [Ollama](https://ollama.com), then `ollama pull qwen3:4b`.
- **CS697 classroom demo registration:** set `CTAI_DEMO_MODE=true`. Coordinators can then register with invitation code **`0000`**. Leave this off for normal secure installs (use `python -m scripts.manage_users create-invite` instead).

Ollama is an alternative local provider, not an automatic fallback. Optional Patient Chat answer rewording: `CTAI_CHAT_LLM=1`.

**4. Run the application**

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Try the Demo

1. With `CTAI_DEMO_MODE=true`, create a **Coordinator** account using invitation code **`0000`**, then sign in.
2. Upload / publish using the synthetic Type 2 Diabetes protocols (`T2D-001 V1` and `V2`) to demonstrate amendment detection.
3. Sign in as a **Patient**, open the published guide, and ask the Chat Assistant questions.

The system identifies changes such as:

- Eligibility age expanded from 18–65 to 18–70.
- HbA1c blood testing added.
- An additional follow-up visit on Day 60.

After coordinator approval and publication, patients can ask the AI Chat Assistant questions such as:

- "When are my study visits?"
- "What tests will I have?"
- "Am I eligible for this clinical trial?"

The assistant provides source-grounded information and directs personal eligibility decisions to the study team.

## Safety and Limitations

Clinical Trial AI is an educational research prototype. It uses human approval, source verification, and approved-content restrictions to reduce unsupported medical claims.

It does not provide medical advice, diagnose conditions, prescribe treatment, or determine individual eligibility. Do not upload real patient data.

---

**Clinical Trial AI — Making clinical trial information easier to understand, one question at a time.**
