import pytest


def _offline_rewriter(original, section, feedback=None):
    return f"In simple words: {original}"


@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file; data/ctai.db is never touched."""
    monkeypatch.setenv("CTAI_DB_PATH", str(tmp_path / "ctai_test.db"))


@pytest.fixture
def publish_context(monkeypatch):
    """Session state + stored review data for calling publish_block_reason directly."""
    import streamlit as st
    from tests.publish_helpers import make_publishable

    monkeypatch.setattr(st, "session_state", {"coordinator_protocol_id": "protocol-eight"})
    make_publishable("protocol-eight")


@pytest.fixture(autouse=True)
def offline_guide_model(request, monkeypatch):
    """
    Opening the Review page now generates statements in the background.
    Keep every test off the real model; tests that need specific wording
    patch the rewriter themselves, and the model-provider tests are untouched.
    """
    if request.module.__name__.endswith("test_model_provider"):
        yield
        return
    from services import guide_jobs

    monkeypatch.setattr(
        "agents.plain_language_agent.ollama_rewriter", _offline_rewriter
    )

    def _offline_summary(prompt):
        raise RuntimeError("offline test: no model")

    monkeypatch.setattr(
        "services.summary_review.ollama_summary_writer", _offline_summary
    )
    guide_jobs.jobs.reset()
    yield
    guide_jobs.jobs.wait_idle(5)
    guide_jobs.jobs.reset()
