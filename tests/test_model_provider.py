import json
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
