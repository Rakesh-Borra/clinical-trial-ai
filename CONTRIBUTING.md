
# Contributing to Clinical Trial AI

## Team Development Workflow

We use one GitHub repository and three development branches.

- Rakesh: dev/rakesh
- Raj: dev/raj
- Nam: dev/nam

The main branch contains reviewed, integrated code.

## 1. Getting Started

Clone the repository:

```bash
git clone https://github.com/Rakesh-Borra/clinical-trial-ai.git
cd clinical-trial-ai
```

Switch to your assigned branch:

```bash
git fetch origin
git switch dev/raj
```

Replace `dev/raj` with your assigned branch.

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Code Ownership

Rakesh:
- agents/
- services/
- backend/

Raj:
- tests/
- evaluation/
- docs/

Nam:
- frontend/
- UI components and pages

Coordinate before editing files owned by another teammate.

## 3. Using AI Coding Assistants

ChatGPT and Claude may help generate and review code.

Before accepting AI-generated changes:

1. Understand what the code does.
2. Check which files were modified.
3. Run the relevant tests.
4. Verify that existing functionality still works.
5. Never paste real patient data or secrets into AI tools.
6. Do not let AI tools overwrite teammates' work.

## 4. Committing Work

Work on your assigned branch.

```bash
git status
git add <specific-files>
git commit -m "feat: describe completed feature"
git push origin <your-branch>
```

Prefer specific file paths over `git add .` when
AI tools have made multiple changes.

## 5. Integration

Create a Pull Request when a feature or module is ready.

Before merging:
- Review changed files.
- Run tests.
- Confirm interface compatibility.
- Resolve conflicts carefully.
- Keep main working.

Do not force-push shared branches or commit directly to main.

## 6. Communication

Use TASKS.md to track major development milestones.

Before changing shared interfaces, notify teammates
and update ARCHITECTURE.md if needed.

## 7. Security and Safety

- Use only public or synthetic trial data.
- Never commit API keys, passwords, or patient records.
- Keep local secrets in ignored environment files.
- Treat AI outputs as unverified until reviewed.
- Do not use the prototype for clinical decisions.
