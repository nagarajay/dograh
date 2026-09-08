"""Tests for Groq reasoning models being kept out of spoken content.

Groq's reasoning models (gpt-oss and friends) return their chain of thought in
the assistant message content by default. The pipeline hands that content
straight to TTS, so the bot ends up speaking its own reasoning and tool
narration. `reasoning_format: "hidden"` keeps the reasoning out of content.

The parameter has to reach Groq inside `extra_body`. `LLMSettings.extra` is
merged into the *top-level* kwargs of `AsyncCompletions.create()`, and that
signature is typed: a Groq-only field placed there raises TypeError before any
HTTP request is made, killing every generation in the call. The signature test
at the bottom of this module pins that down.
"""

import inspect

import pytest
from openai.resources.chat.completions import AsyncCompletions

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import (
    _is_groq_reasoning_model,
    create_llm_service_from_provider,
)

# `create` takes its request fields as keyword arguments plus the SDK's own
# escape hatches (extra_body, extra_headers, timeout, ...). Anything else in
# the top-level params dict is a TypeError at call time.
_SDK_CREATE_PARAMS = set(inspect.signature(AsyncCompletions.create).parameters)


def _create_kwargs(service):
    """The kwargs the service would hand to AsyncCompletions.create()."""
    return service.build_chat_completion_params(
        {"messages": [], "tools": None, "tool_choice": None}
    )


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "deepseek-r1-distill-llama-70b"],
)
def test_reasoning_models_are_detected(model: str):
    assert _is_groq_reasoning_model(model)


@pytest.mark.parametrize("model", ["llama-3.3-70b-versatile", "llama3-8b-8192", ""])
def test_non_reasoning_models_are_not_detected(model: str):
    assert not _is_groq_reasoning_model(model)


def test_groq_reasoning_model_hides_reasoning_from_content():
    service = create_llm_service_from_provider(
        ServiceProviders.GROQ.value,
        "openai/gpt-oss-120b",
        "test-key",
    )

    assert service._settings.extra == {"extra_body": {"reasoning_format": "hidden"}}
    assert "reasoning_format" not in service._settings.extra, (
        "reasoning_format must not sit at the top level: it is merged into the "
        "SDK's create() kwargs, which rejects unknown names"
    )


def test_groq_non_reasoning_model_sends_no_reasoning_format():
    """Groq rejects `reasoning_format` on models that do not reason."""
    service = create_llm_service_from_provider(
        ServiceProviders.GROQ.value,
        "llama-3.3-70b-versatile",
        "test-key",
    )

    assert service._settings.extra == {}
    assert "reasoning_format" not in _create_kwargs(service).get("extra_body", {})


def test_other_providers_are_untouched():
    service = create_llm_service_from_provider(
        ServiceProviders.OPENAI.value,
        "gpt-4o-mini",
        "test-key",
    )

    assert service._settings.extra == {}
    assert "reasoning_format" not in _create_kwargs(service).get("extra_body", {})


def test_reasoning_format_reaches_the_request_body():
    service = create_llm_service_from_provider(
        ServiceProviders.GROQ.value,
        "openai/gpt-oss-120b",
        "test-key",
    )

    params = _create_kwargs(service)

    assert params["extra_body"] == {"reasoning_format": "hidden"}
    assert "reasoning_format" not in params


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        (ServiceProviders.GROQ.value, "openai/gpt-oss-120b"),
        (ServiceProviders.GROQ.value, "llama-3.3-70b-versatile"),
        (ServiceProviders.OPENAI.value, "gpt-5-mini"),
        (ServiceProviders.OPENAI.value, "gpt-4o-mini"),
    ],
)
def test_create_kwargs_are_all_supported_by_the_sdk(provider: str, model: str):
    """Regression guard for the TypeError that killed every generation.

    A Groq- or provider-specific field left in the top-level params dict is not
    caught anywhere: `create()` raises `TypeError: AsyncCompletions.create() got
    an unexpected keyword argument ...` at call time, before any HTTP request,
    so the failure only shows up as a dead call.
    """
    service = create_llm_service_from_provider(provider, model, "test-key")

    unsupported = set(_create_kwargs(service)) - _SDK_CREATE_PARAMS

    assert not unsupported, (
        f"{provider}/{model} would pass kwargs the OpenAI SDK rejects: "
        f"{sorted(unsupported)}. Provider-specific fields belong in extra_body."
    )
