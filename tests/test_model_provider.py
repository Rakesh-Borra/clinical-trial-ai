import json
import os
from io import BytesIO
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest
from pydantic import BaseModel

from agents.protocol_analysis_agent import (
    MetadataEligibilitySafetyExtraction,
    _chat_structured,
)
from services.model_provider import (
    ModelProviderError,
    chat_completion,
)


class ExampleResponse(BaseModel):
    required_value: str
    optional_value: str | None = None


def test_ollama_provider_uses_configured_timeout_schema_and_model(monkeypatch):
    monkeypatch.delenv("CTAI_MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("CTAI_MODEL_TIMEOUT_SECONDS", "45")
    response = {"message": {"content": '{"required_value":"ok"}'}}
    client = Mock()
    client.chat.return_value = response

    with patch(
        "services.model_provider.ollama.Client",
        return_value=client,
    ) as client_factory:
        content = chat_completion(
            "prompt",
            ExampleResponse,
            model="qwen3:4b",
            options={"temperature": 0, "num_ctx": 16384},
        )

    client_factory.assert_called_once_with(timeout=45.0)
    client.chat.assert_called_once()
    call = client.chat.call_args.kwargs
    assert call["model"] == "qwen3:4b"
    assert call["messages"] == [{"role": "user", "content": "prompt"}]
    assert call["format"] == ExampleResponse.model_json_schema()
    assert call["options"] == {"temperature": 0, "num_ctx": 16384}
    assert json.loads(content) == {"required_value": "ok"}


def test_openrouter_sends_strict_schema_and_validates_json_response(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-log")
    monkeypatch.setenv("CTAI_OPENROUTER_MODEL", "example/model")
    monkeypatch.setenv("CTAI_MODEL_TIMEOUT_SECONDS", "31")
    response = BytesIO(json.dumps({
        "choices": [{
            "message": {
                "content": (
                    '{"required_value":"ok","optional_value":null}'
                )
            },
            "finish_reason": "stop",
        }]
    }).encode())

    with patch("services.model_provider.urlopen", return_value=response) as open_url:
        content = chat_completion(
            "clinical prompt",
            ExampleResponse,
            model="ignored-for-openrouter",
            options={"temperature": 0},
        )

    request = open_url.call_args.args[0]
    payload = json.loads(request.data)
    assert open_url.call_args.kwargs["timeout"] == 31.0
    assert payload["model"] == "example/model"
    assert payload["messages"] == [
        {"role": "user", "content": "clinical prompt"}
    ]
    assert payload["response_format"]["type"] == "json_schema"
    json_schema = payload["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    assert schema["required"] == ["required_value", "optional_value"]
    assert schema["additionalProperties"] is False
    assert payload["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "allow_fallbacks": False,
    }
    assert json.loads(content) == {
        "required_value": "ok",
        "optional_value": None,
    }


def test_openrouter_missing_key_fails_without_fallback(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with patch("services.model_provider.urlopen") as open_url, patch(
        "services.model_provider.ollama.Client"
    ) as ollama_client:
        with pytest.raises(ModelProviderError, match="OPENROUTER_API_KEY"):
            chat_completion("prompt", ExampleResponse, model="qwen3:4b")

    open_url.assert_not_called()
    ollama_client.assert_not_called()


def test_unknown_provider_and_invalid_timeout_fail_clearly(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "other")
    with pytest.raises(ModelProviderError, match="CTAI_MODEL_PROVIDER"):
        chat_completion("prompt", ExampleResponse, model="qwen3:4b")

    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("CTAI_MODEL_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ModelProviderError, match="positive number"):
        chat_completion("prompt", ExampleResponse, model="qwen3:4b")


def test_openrouter_http_error_is_bounded_and_does_not_leak_response(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-log")
    error = HTTPError(
        "https://openrouter.ai",
        429,
        "rate limited",
        hdrs=None,
        fp=BytesIO(b"private provider response"),
    )

    with patch("services.model_provider.urlopen", side_effect=error):
        with pytest.raises(ModelProviderError, match="HTTP 429") as exc_info:
            chat_completion("prompt", ExampleResponse, model="qwen3:4b")

    assert "test-key-do-not-log" not in str(exc_info.value)
    assert "private provider response" not in str(exc_info.value)


def test_truncated_response_is_rejected(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = BytesIO(json.dumps({
        "choices": [{
            "message": {"content": '{"required_value":"partial'},
            "finish_reason": "length",
        }]
    }).encode())

    with patch("services.model_provider.urlopen", return_value=response):
        with pytest.raises(ModelProviderError, match="truncated"):
            chat_completion("prompt", ExampleResponse, model="qwen3:4b")


def test_existing_protocol_json_and_pydantic_validation_remain_in_place(monkeypatch):
    monkeypatch.delenv("CTAI_MODEL_PROVIDER", raising=False)
    client = Mock()
    client.chat.return_value = {
        "message": {"content": "{not valid json"}
    }

    with patch(
        "services.model_provider.ollama.Client",
        return_value=client,
    ):
        with pytest.raises(json.JSONDecodeError):
            _chat_structured("prompt", MetadataEligibilitySafetyExtraction)


def test_openrouter_content_uses_existing_protocol_pydantic_schema(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = BytesIO(json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "trial_title": "Benchmark Study",
                    "protocol_id": None,
                    "inclusion_criteria": [],
                    "exclusion_criteria": [],
                    "safety_information": [],
                })
            },
            "finish_reason": "stop",
        }]
    }).encode())

    with patch("services.model_provider.urlopen", return_value=response):
        result = _chat_structured(
            "prompt",
            MetadataEligibilitySafetyExtraction,
        )

    assert result.trial_title == "Benchmark Study"
    assert result.inclusion_criteria == []


def test_plain_language_rewriter_uses_the_selected_provider():
    from agents.plain_language_agent import ollama_rewriter
    from models.plain_language_models import PlainLanguageRewrite

    response = '{"plain_text":"A simpler statement."}'
    with patch(
        "agents.plain_language_agent.chat_completion",
        return_value=response,
    ) as mock_completion:
        assert ollama_rewriter("Original statement.", "procedures") == (
            "A simpler statement."
        )

    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args
    assert "Original statement." in args[0]
    assert args[1] is PlainLanguageRewrite
    assert kwargs["model"] == "qwen3:4b"
    assert kwargs["options"] == {"temperature": 0}
    assert kwargs["think"] is False


# --- LLM_PROVIDER / OPENROUTER_* configuration (all calls mocked) -------------

from services import model_provider  # noqa: E402

_OK = json.dumps({"choices": [{"message": {"content": '{"required_value": "x"}'}, "finish_reason": "stop"}]}).encode()


def _fake_urlopen(captured):
    def opener(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        body = Mock()
        body.read.return_value = _OK
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=body)
        ctx.__exit__ = Mock(return_value=False)
        return ctx

    return opener


def test_llm_provider_selects_openrouter_and_wins_over_the_legacy_name(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert model_provider.provider_display_name() == "OpenRouter (cloud)"
    monkeypatch.delenv("LLM_PROVIDER")
    assert model_provider.provider_display_name() == "Ollama (local)"
    monkeypatch.delenv("CTAI_MODEL_PROVIDER")
    assert model_provider.provider_display_name() == "Ollama (local)"  # default


def test_invalid_llm_provider_fails_and_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with patch("services.model_provider.ollama.Client") as client, patch(
        "services.model_provider.urlopen"
    ) as opener:
        with pytest.raises(ModelProviderError, match="LLM_PROVIDER"):
            chat_completion("p", ExampleResponse, model="qwen3:4b")
    client.assert_not_called()
    opener.assert_not_called()


def test_openrouter_request_uses_env_base_url_model_and_bearer_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/api/v1/")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/model-x")
    captured = {}
    with patch("services.model_provider.urlopen", _fake_urlopen(captured)), patch(
        "services.model_provider.ollama.Client"
    ) as client:
        text = chat_completion("prompt", ExampleResponse, model="qwen3:4b")
    client.assert_not_called()
    request = captured["request"]
    assert request.full_url == "https://example.test/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-test-not-real"
    payload = json.loads(request.data)
    assert payload["model"] == "vendor/model-x"
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["provider"]["allow_fallbacks"] is False
    assert json.loads(text) == {"required_value": "x"}


def test_default_model_and_base_url_when_only_the_key_is_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    captured = {}
    with patch("services.model_provider.urlopen", _fake_urlopen(captured)):
        chat_completion("prompt", ExampleResponse, model="qwen3:4b")
    assert captured["request"].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert json.loads(captured["request"].data)["model"] == model_provider.DEFAULT_OPENROUTER_MODEL


def test_legacy_model_variable_still_works_and_new_one_wins(monkeypatch):
    monkeypatch.setenv("CTAI_OPENROUTER_MODEL", "legacy/model")
    assert model_provider._openrouter_model() == "legacy/model"
    monkeypatch.setenv("OPENROUTER_MODEL", "new/model")
    assert model_provider._openrouter_model() == "new/model"


def test_non_https_base_url_is_rejected_before_any_request(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://example.test/v1")
    with patch("services.model_provider.urlopen") as opener:
        with pytest.raises(ModelProviderError, match="https"):
            chat_completion("p", ExampleResponse, model="qwen3:4b")
    opener.assert_not_called()


def test_errors_never_contain_the_api_key(monkeypatch):
    secret = "sk-test-SECRET-123"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    failures = [
        HTTPError("https://x", 401, "bad key " + secret, {}, BytesIO(secret.encode())),
        TimeoutError("timed out " + secret),
    ]
    for failure in failures:
        with patch("services.model_provider.urlopen", side_effect=failure), patch(
            "services.model_provider.ollama.Client"
        ) as client:
            with pytest.raises(ModelProviderError) as caught:
                chat_completion("p", ExampleResponse, model="qwen3:4b")
        assert secret not in str(caught.value)
        client.assert_not_called()  # no silent fallback to Ollama


def test_openrouter_failure_is_not_retried_on_ollama_via_the_agent(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    with patch("services.model_provider.urlopen", side_effect=TimeoutError()), patch(
        "services.model_provider.ollama.Client"
    ) as client:
        with pytest.raises(ModelProviderError):
            chat_completion("p", ExampleResponse, model="qwen3:4b")
    client.assert_not_called()


def test_dotenv_is_loaded_without_overriding_the_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openrouter\nOPENROUTER_MODEL=from/dotenv\n")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(model_provider, "_dotenv_loaded", False)
    monkeypatch.setattr(model_provider, "_project_env_path", lambda: env_file)
    monkeypatch.setenv("OPENROUTER_MODEL", "from/real-env")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    try:
        model_provider.load_environment()
        assert os.environ["LLM_PROVIDER"] == "openrouter"  # taken from .env
        assert os.environ["OPENROUTER_MODEL"] == "from/real-env"  # real env wins
    finally:
        os.environ.pop("LLM_PROVIDER", None)


def test_dotenv_is_skipped_under_pytest(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openrouter\n")
    monkeypatch.setattr(model_provider, "_dotenv_loaded", False)
    monkeypatch.setattr(model_provider, "_project_env_path", lambda: env_file)
    model_provider.load_environment()  # PYTEST_CURRENT_TEST is set
    assert "LLM_PROVIDER" not in os.environ


def test_analysis_cache_key_follows_the_active_openrouter_model(monkeypatch):
    assert model_provider.active_model_id("qwen3:4b") == "qwen3:4b"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/a")
    first = model_provider.active_model_id("qwen3:4b")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/b")
    assert first == "vendor/a" and model_provider.active_model_id("qwen3:4b") == "vendor/b"


def test_env_is_git_ignored_and_the_example_holds_no_real_key():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=root)
    assert ignored.returncode == 0
    example = (root / ".env.example").read_text().splitlines()
    assert "OPENROUTER_API_KEY=" in example  # placeholder only
    assert not any(line.startswith("OPENROUTER_API_KEY=") and line != "OPENROUTER_API_KEY=" for line in example)
