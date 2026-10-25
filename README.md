# Clinical Trial AI

Clinical Trial AI helps coordinators turn clinical trial protocol PDFs into plain-language patient guides linked to source pages. A human coordinator must review and publish every statement before patients can see it. Patients then read the guide and ask questions answered only from that approved content.

**Repository:** [https://github.com/Rakesh-Borra/clinical-trial-ai](https://github.com/Rakesh-Borra/clinical-trial-ai)

## Features

- PDF protocol analysis with page citations
- Three main agents: Protocol Analysis, Plain-Language Patient Guide, and Patient Chat
- V1/V2 amendment comparison with source checks
- Source-backed patient summaries
- Coordinator approval, publish, and unpublish
- Patient Chat grounded in the active published guide only

## Getting Started

```bash
git clone https://github.com/Rakesh-Borra/clinical-trial-ai.git
cd clinical-trial-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for your model provider (see below). Start the app:

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501**.

## AI Models

Model calls use `services/model_provider.py`. Set **`LLM_PROVIDER`** to:

- **`openrouter`** — cloud via OpenRouter (`OPENROUTER_API_KEY` required; default model `openai/gpt-4.1-mini`)
- **`ollama`** — local (default if unset; model `qwen3:4b`)

There is **no automatic fallback** between providers. Analysis, patient-guide generation, and optional Patient Chat rewording (`CTAI_CHAT_LLM=1`, off by default) share this setting. For Ollama, install from [ollama.com](https://ollama.com) and run `ollama pull qwen3:4b` before Analyze.

## Sample Protocols

Upload files from `Protocol samples/` in the coordinator Upload step.

- **T2D-001 V1/V2** — synthetic Type 2 Diabetes pair for amendment comparison
- Other `*-SYN-*` PDFs — synthetic teaching samples
- **`123_AsthmaDupilumab_NCT02414854.pdf`** — real research protocol (NCT02414854)

## Safety

Educational prototype only. Do not use real patient data. Coordinator approval is required before patients see content. This is not medical advice and cannot decide eligibility.
