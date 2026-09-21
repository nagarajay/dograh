"""GPT-5 reasoning_effort must be a value the model accepts.

Every workflow turn sends function tools. On Chat Completions, gpt-5.6-luna
rejects "minimal" (HTTP 400, unsupported value) and "low" (HTTP 400, "Function
tools with reasoning_effort are not supported ... set reasoning_effort to
'none'"), so it must be sent "none". Other GPT-5 models keep "minimal".
"""

import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_llm_service_from_provider


def _extra(provider: str, model: str) -> dict:
    return create_llm_service_from_provider(provider, model, "test-key")._settings.extra


@pytest.mark.parametrize(
    "provider", [ServiceProviders.OPENAI.value, ServiceProviders.ATLASCLOUD.value]
)
def test_gpt_5_6_luna_gets_none(provider: str):
    extra = _extra(provider, "gpt-5.6-luna")

    assert extra == {"reasoning_effort": "none", "verbosity": "low"}
    assert extra["reasoning_effort"] not in ("minimal", "low")


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
def test_other_gpt_5_models_keep_minimal(model: str):
    assert _extra(ServiceProviders.OPENAI.value, model) == {
        "reasoning_effort": "minimal",
        "verbosity": "low",
    }


def test_non_gpt_5_model_gets_no_reasoning_effort():
    service = create_llm_service_from_provider(
        ServiceProviders.OPENAI.value, "gpt-4o-mini", "test-key"
    )

    assert "reasoning_effort" not in service._settings.extra
