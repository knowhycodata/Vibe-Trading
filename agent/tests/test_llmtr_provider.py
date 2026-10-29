"""LLMTR provider: catalog entry, unchanged model ids, no reasoning object."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import src.providers.llm as llm_mod
from src.providers.capabilities import get_llm_credentials, get_provider_capabilities
from src.providers.llm import ChatOpenAIWithReasoning, build_llm
from src.swarm.models import public_model_metadata

_PROVIDERS_JSON = (
    Path(__file__).resolve().parents[1] / "src" / "providers" / "llm_providers.json"
)
_BASE_URL = "https://llmtr.com/v1"
_FREE_MODEL = "qwen/qwen3.8-27b-free"
_HOSTED_MODEL = "llmtr/qwen3-8-27b"


def _completion_body(model: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ---------------------------------------------------------------------------
# Registration surfaces
# ---------------------------------------------------------------------------


def test_catalog_lists_llmtr() -> None:
    catalog = {
        item["name"]: item
        for item in json.loads(_PROVIDERS_JSON.read_text(encoding="utf-8"))
    }
    entry = catalog["llmtr"]
    assert entry["label"] == "LLMTR"
    assert entry["api_key_env"] == "LLMTR_API_KEY"
    assert entry["base_url_env"] == "LLMTR_BASE_URL"
    assert entry["default_model"] == _FREE_MODEL
    assert entry["default_base_url"] == _BASE_URL
    assert entry["api_key_required"] is True
    assert entry["model_options"] == [_FREE_MODEL, _HOSTED_MODEL]


def test_capabilities_use_generic_openai_path() -> None:
    caps = get_provider_capabilities("llmtr", _HOSTED_MODEL)
    assert caps.name == "llmtr"
    assert caps.api_key_env == "LLMTR_API_KEY"
    assert caps.base_url_env == "LLMTR_BASE_URL"
    assert caps.capture_reasoning is False
    assert caps.send_reasoning_content is False
    assert caps.openrouter_reasoning_body is False
    assert caps.top_level_reasoning_effort is False


def test_qwen_prefixed_model_stays_on_llmtr() -> None:
    """The free catalog model starts with ``qwen/``; that is not the Qwen provider."""
    caps = get_provider_capabilities("llmtr", _FREE_MODEL)
    assert caps.name == "llmtr"
    assert caps.api_key_env == "LLMTR_API_KEY"


def test_llmtr_model_prefix_does_not_reroute_openai() -> None:
    """``llmtr/`` is part of the model id, not a provider inferred from the name."""
    caps = get_provider_capabilities("openai", _HOSTED_MODEL)
    assert caps.name == "openai"


def test_credentials_fall_back_to_catalog_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "LLMTR_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLMTR_API_KEY", "llmtr-test-key")
    creds = get_llm_credentials("llmtr", _HOSTED_MODEL)
    assert creds["provider"] == "llmtr"
    assert creds["api_key"] == "llmtr-test-key"
    assert creds["base_url"] == _BASE_URL
    assert creds["model"] == _HOSTED_MODEL


def test_swarm_public_provider_allowlist_includes_llmtr() -> None:
    from src.swarm.models import _PUBLIC_PROVIDERS

    assert "llmtr" in _PUBLIC_PROVIDERS
    assert public_model_metadata(_FREE_MODEL) == _FREE_MODEL
    assert public_model_metadata(_HOSTED_MODEL) == _HOSTED_MODEL


def test_cli_onboarding_surfaces_agree() -> None:
    import cli
    from cli.onboard import PROVIDERS

    onboard = next(p for p in PROVIDERS if p.key == "llmtr")
    legacy = next(item for item in cli._PROVIDER_CHOICES if item["provider"] == "llmtr")
    assert onboard.key_env == legacy["key_env"] == "LLMTR_API_KEY"
    assert onboard.base_env == legacy["base_env"] == "LLMTR_BASE_URL"
    assert onboard.base_url == legacy["base_url"] == _BASE_URL
    assert onboard.default_model == legacy["model"] == _FREE_MODEL
    assert onboard.key_prefix == legacy["key_prefix"] == "llmtr-"
    assert cli._provider_key_env("llmtr") == "LLMTR_API_KEY"
    assert cli._provider_base_env("llmtr") == "LLMTR_BASE_URL"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pin_dotenv_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep build_llm from reading a real ``.env``."""
    monkeypatch.setattr(llm_mod, "_dotenv_loaded", True)


def _capture_kwargs(env: dict[str, str]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    with patch.dict(os.environ, env, clear=True):
        with patch.object(llm_mod, "ChatOpenAIWithReasoning", _FakeChatOpenAI):
            build_llm()
    return captured


def test_build_llm_keeps_llmtr_model_prefix_and_skips_reasoning() -> None:
    kwargs = _capture_kwargs(
        {
            "LANGCHAIN_PROVIDER": "llmtr",
            "LLMTR_API_KEY": "llmtr-test-key",
            "LANGCHAIN_MODEL_NAME": _HOSTED_MODEL,
            "LANGCHAIN_REASONING_EFFORT": "high",
            "LANGCHAIN_USE_RESPONSES_API": "false",
        }
    )
    assert kwargs["model"] == _HOSTED_MODEL
    assert kwargs["base_url"] == _BASE_URL
    assert kwargs["api_key"] == "llmtr-test-key"
    assert kwargs["reasoning_effort"] is None
    assert kwargs["extra_body"] is None
    assert kwargs["reasoning"] is None


pytestmark_wire = pytest.mark.skipif(
    ChatOpenAIWithReasoning is None, reason="langchain-openai is not installed"
)


def _adapter(client: httpx.Client, *, model: str) -> ChatOpenAIWithReasoning:
    return ChatOpenAIWithReasoning(
        model=model,
        api_key="llmtr-test",
        base_url=_BASE_URL,
        temperature=0.0,
        http_client=client,
        vibe_provider="llmtr",
        vibe_api_key="llmtr-test",
    )


@pytestmark_wire
@pytest.mark.parametrize("model", [_FREE_MODEL, _HOSTED_MODEL])
def test_chat_request_sends_published_model_id(model: str) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion_body(model))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert _adapter(client, model=model).invoke("hello").content == "ok"

    assert seen[0]["model"] == model
    reasoning = seen[0].get("reasoning")
    assert not isinstance(reasoning, dict)
