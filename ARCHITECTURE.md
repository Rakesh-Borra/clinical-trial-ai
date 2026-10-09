
# Clinical Trial AI — System Architecture

## 1. Project Overview

Clinical Trial AI is a multi-agent platform that helps
research coordinators process clinical trial protocols
and provides patients with understandable,
source-grounded trial information.

The system is designed for modular development,
human oversight, security, and future scalability.

## 2. System Architecture

```text
                CLINICAL TRIAL AI
                        |
                FRONTEND APPLICATION
                        |
          +-------------+-------------+
          |                           |
    COORDINATOR PORTAL          PATIENT PORTAL
          |                           |
          +-------------+-------------+
                        |
                  BACKEND API
                    FastAPI
                        |
                AGENT ORCHESTRATOR
                        |
        +---------------+---------------+
        |               |               |
   PROTOCOL AGENT   PATIENT AGENT   VERIFICATION AGENT
        |               |               |
        +---------------+---------------+
                        |
              KNOWLEDGE / DATA LAYER
                        |
        +---------------+---------------+
        |               |               |
    PostgreSQL      Vector Store     PDF Storage
        |
   Trials, Users, Versions,
   Approvals, Audit Events
```

## 3. AI Agents

### Protocol Analysis Agent

Responsibilities:
- Read extracted protocol text.
- Identify trial objectives and requirements.
- Extract eligibility criteria.
- Extract visit schedules and procedures.
- Identify risks and important instructions.
- Produce structured outputs with source references.

### Patient Assistant Agent

Responsibilities:
- Generate plain-language trial explanations.
- Answer patient questions using approved sources.
- Explain trial visits and procedures.
- Indicate when information is unavailable.
- Refer medical or eligibility questions to a qualified
  clinical professional when appropriate.

### Verification Agent

Responsibilities:
- Compare generated statements with source passages.
- Identify unsupported or contradictory claims.
- Flag uncertain answers for review.
- Prevent unapproved information from being published.
- Record verification results for evaluation.

Verification is an additional safeguard, not a
guarantee of medical accuracy.

## 4. Backend Services

Planned services:

- PDF Processing Service
- Protocol Management Service
- Agent Orchestration Service
- Retrieval / RAG Service
- Authentication Service
- Approval and Publication Service
- Audit Logging Service

## 5. Data Storage

Initial development:
- SQLite for structured application data.
- Local file storage for public/synthetic PDFs.
- Local vector index for retrieval experiments.

Later development:
- PostgreSQL for application data.
- Managed document/object storage.
- Persistent vector retrieval.
- Database migrations and backup procedures.

## 6. Coordinator Workflow

1. Coordinator creates a trial.
2. Coordinator uploads a protocol.
3. PDF service extracts page-referenced text.
4. Protocol Agent analyzes the content.
5. Patient Agent prepares a plain-language guide.
6. Verification Agent checks source support.
7. Coordinator reviews and approves the guide.
8. Approved content becomes available to patients.

## 7. Patient Workflow

1. Patient opens an authorized trial portal.
2. Patient reads approved trial information.
3. Patient asks a question.
4. Retrieval service finds relevant approved content.
5. Patient Agent generates a grounded explanation.
6. Verification Agent checks the response.
7. The application displays the answer with references,
   or explains that the question needs human review.

## 8. Protocol Version Management

- Store protocol versions separately.
- Preserve original source documents.
- Compare changes between versions.
- Flag changes affecting patient-facing information.
- Require reapproval before publishing updated content.

## 9. Safety and Privacy

- Use public or synthetic data during development.
- Do not upload real patient information to AI services.
- Never independently diagnose or determine eligibility.
- Require qualified human review for patient materials.
- Enforce role-based access in later phases.
- Maintain audit logs for important actions.
- Treat AI-generated text as untrusted until validated.

## 10. Team Ownership

### Rakesh
AI agents, PDF processing, backend, retrieval,
database integration, and orchestration.

### Raj
Protocol research, test datasets, evaluation,
automated testing, and documentation.

### Nam
Coordinator dashboard, patient portal,
chatbot interface, usability, and accessibility.

## 11. Integration Contracts

Frontend and backend should exchange structured data
through documented interfaces.

Initial planned API endpoints:

POST /trials
POST /trials/{trial_id}/protocols
POST /trials/{trial_id}/analyze
GET  /trials/{trial_id}/summary
POST /trials/{trial_id}/approve
POST /trials/{trial_id}/chat

These are proposed endpoints, not yet implemented.

## 12. Development Principles

1. Build modules independently.
2. Define data contracts before integration.
3. Preserve protocol source references.
4. Write tests for important behavior.
5. Keep secrets and private data out of Git.
6. Review code before merging into main.
7. Build a working system incrementally.
