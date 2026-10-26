"""Backend-only chat completion providers for structured model calls."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional, Type
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ollama
from pydantic import BaseModel

from services import performance


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_URL = DEFAULT_OPENROUTER_BASE_URL + "/chat/completions"
# Strict JSON-schema output, 1M-token context and low cost; see the README.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192


class ModelProviderError(RuntimeError):
    """Raised when provider configuration or a model request fails."""


_dotenv_loaded = False


def _project_env_path():
    """Absolute path to the repository ``.env`` (stable regardless of cwd)."""
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / ".env"


def load_environment() -> None:
    """
    Load the project ``.env`` once, without overriding real environment variables.

    Uses the repository root (next to ``services/``), not the process cwd, so
    Streamlit always sees the same file. Skipped under pytest so a developer's
    ``.env`` can never change test results. Values are never printed or logged.
    """
    global _dotenv_loaded
    if _dotenv_loaded or "PYTEST_CURRENT_TEST" in os.environ:
        return
    _dotenv_loaded = True
    from dotenv import load_dotenv

    load_dotenv(_project_env_path(), override=False)


def _configured_provider() -> str:
    """LLM_PROVIDER, or the older CTAI_MODEL_PROVIDER; Ollama when neither is set."""
    load_environment()
    value = os.getenv("LLM_PROVIDER") or os.getenv("CTAI_MODEL_PROVIDER") or "ollama"
    return value.strip().lower()


def _provider_name() -> str:
    provider = _configured_provider()
    if provider not in {"ollama", "openrouter"}:
        raise ModelProviderError(
            "LLM_PROVIDER (or CTAI_MODEL_PROVIDER) must be either 'ollama' or 'openrouter'."
        )
    return provider


def _openrouter_model() -> str:
    load_environment()
    return (
        os.getenv("OPENROUTER_MODEL")
        or os.getenv("CTAI_OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    ).strip()


def active_model_id(ollama_model: str) -> str:
    """The model that will answer: part of the analysis-cache key."""
    if _configured_provider() == "openrouter":
        return _openrouter_model()
    return ollama_model


def _openrouter_endpoint() -> str:
    base = (os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).strip()
    if not base.lower().startswith("https://"):
        raise ModelProviderError("OPENROUTER_BASE_URL must start with https://.")
    return base.rstrip("/") + "/chat/completions"


def provider_display_name() -> str:
    """Return a safe UI label for the configured provider, without secrets."""
    provider = _configured_provider()
    if provider == "ollama":
        return "Ollama (local)"
    if provider == "openrouter":
        return "OpenRouter (cloud)"
    return "Unconfigured model provider"


def _timeout_seconds() -> float:
    raw_value = os.getenv(
        "CTAI_MODEL_TIMEOUT_SECONDS",
        str(DEFAULT_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_value)
    except ValueError as exc:
        raise ModelProviderError(
            "CTAI_MODEL_TIMEOUT_SECONDS must be a positive number."
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ModelProviderError(
            "CTAI_MODEL_TIMEOUT_SECONDS must be a positive number."
        )
    return timeout


def _max_output_tokens() -> int:
    raw_value = os.getenv(
        "CTAI_OPENROUTER_MAX_OUTPUT_TOKENS",
        str(DEFAULT_MAX_OUTPUT_TOKENS),
    )
    try:
        max_tokens = int(raw_value)
    except ValueError as exc:
        raise ModelProviderError(
            "CTAI_OPENROUTER_MAX_OUTPUT_TOKENS must be a positive integer."
        ) from exc
    if max_tokens <= 0:
        raise ModelProviderError(
            "CTAI_OPENROUTER_MAX_OUTPUT_TOKENS must be a positive integer."
        )
    return max_tokens


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema to the strict structured-output subset."""
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "default":
            continue
        if isinstance(value, dict):
            result[key] = _strict_schema(value)
        elif isinstance(value, list):
            result[key] = [
                _strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    if result.get("type") == "object":
        properties = result.get("properties", {})
        result["additionalProperties"] = False
        result["required"] = list(properties)
    return result


def _response_content(response: Any) -> tuple[str, Optional[str]]:
    """Extract response text and finish reason from provider response shapes."""
    if isinstance(response, dict):
        message = response.get("message") or {}
        if not message:
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return (
                    message.get("content"),
                    choices[0].get("finish_reason"),
                )
        return message.get("content"), response.get("done_reason")

    message = getattr(response, "message", None)
    if message is not None:
        return (
            getattr(message, "content", None),
            getattr(response, "done_reason", None),
        )

    choices = getattr(response, "choices", None)
    if choices:
        choice = choices[0]
        return (
            getattr(getattr(choice, "message", None), "content", None),
            getattr(choice, "finish_reason", None),
        )
    return None, None


def _validate_response(response: Any) -> str:
    content, finish_reason = _response_content(response)
    if finish_reason in {"length", "max_tokens"}:
        raise ModelProviderError(
            "The model response was truncated at the output-token limit."
        )
    if not isinstance(content, str) or not content.strip():
        raise ModelProviderError(
            "The model provider returned an empty or invalid response."
        )
    return content


def _ollama_completion(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
    options: Optional[dict[str, Any]],
    think: Optional[bool],
    timeout: float,
) -> str:
    try:
        client = ollama.Client(timeout=timeout)
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "options": options or {},
        }
        if think is not None:
            request["think"] = think
        try:
            response = client.chat(**request)
        except TypeError:
            if think is None:
                raise
            request.pop("think")
            response = client.chat(**request)
    except Exception as exc:
        raise ModelProviderError(
            f"Ollama request failed ({type(exc).__name__})."
        ) from exc
    performance.record_response(response)
    return _validate_response(response)


def _openrouter_completion(
    prompt: str,
    schema: dict[str, Any],
    *,
    timeout: float,
) -> str:
    load_environment()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ModelProviderError(
            "OpenRouter is selected, but OPENROUTER_API_KEY is not set."
        )

    model = _openrouter_model()
    if not model:
        raise ModelProviderError("OPENROUTER_MODEL cannot be empty.")
    endpoint = _openrouter_endpoint()

    try:
        max_tokens = _max_output_tokens()
    except ModelProviderError:
        raise

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_protocol_extraction",
                "strict": True,
                "schema": _strict_schema(schema),
            },
        },
        "provider": {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "allow_fallbacks": False,
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ModelProviderError(
            f"OpenRouter request failed (HTTP {exc.code})."
        ) from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise ModelProviderError(
            f"OpenRouter request failed ({type(exc).__name__})."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ModelProviderError(
            "OpenRouter returned an invalid JSON response envelope."
        ) from exc

    if not isinstance(response_data, dict):
        raise ModelProviderError(
            "OpenRouter returned an invalid JSON response envelope."
        )

    try:
        return _validate_response(response_data)
    except ModelProviderError as exc:
        raise ModelProviderError(f"OpenRouter: {exc}") from exc


def chat_completion(
    prompt: str,
    schema_model: Type[BaseModel],
    *,
    model: str,
    options: Optional[dict[str, Any]] = None,
    think: Optional[bool] = None,
) -> str:
    """Request schema-constrained JSON from the configured provider."""
    provider = _provider_name()
    timeout = _timeout_seconds()
    schema = schema_model.model_json_schema()

    if provider == "ollama":
        return _ollama_completion(
            prompt,
            schema,
            model=model,
            options=options,
            think=think,
            timeout=timeout,
        )
    return _openrouter_completion(prompt, schema, timeout=timeout)
