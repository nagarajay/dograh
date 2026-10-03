"""Trace visibility must follow the span's destination, not one global flag.

Spans fan out to per-org Langfuse projects, so "public" cannot be a single
deployment-wide answer: an org that connected its own project decides for that
project, and everything bound for the env-configured project follows
``LANGFUSE_TRACES_PUBLIC``.
"""

import os
from unittest.mock import patch

import pytest
from pipecat.utils.run_context import set_current_org_id

from api.services.pipecat import tracing_config

ORG = "org-42"
CREDS = {
    "host": "https://langfuse.example.com",
    "public_key": "pk-lf-test",
    "secret_key": "sk-lf-test",
}


@pytest.fixture
def exporter():
    """Install a routing exporter with no default destination."""
    routing = tracing_config._OrgRoutingExporter(default_exporter=None)
    with patch.object(tracing_config, "_org_routing_exporter", routing):
        yield routing


@pytest.fixture(autouse=True)
def clear_org_context():
    """Keep the org context var from leaking between tests."""
    yield
    set_current_org_id(None)


def resolve(env_flag: str | None) -> bool:
    env = {} if env_flag is None else {"LANGFUSE_TRACES_PUBLIC": env_flag}
    with patch.dict(os.environ, env, clear=True):
        return tracing_config._resolve_trace_public()


def test_org_project_honours_its_own_opt_in(exporter):
    exporter.register_org(ORG, **CREDS, traces_public=True)
    set_current_org_id(ORG)

    assert resolve(None) is True


def test_org_project_ignores_the_env_flag(exporter):
    """The env flag belongs to Dograh's project; an org's own is unaffected."""
    exporter.register_org(ORG, **CREDS, traces_public=False)
    set_current_org_id(ORG)

    assert resolve("true") is False


def test_org_project_is_private_unless_asked(exporter):
    exporter.register_org(ORG, **CREDS)
    set_current_org_id(ORG)

    assert resolve("true") is False


def test_visibility_toggle_lands_without_a_credential_change(exporter):
    """Flipping the toggle alone must take effect.

    ``register_org`` short-circuits when the credentials are byte-identical, so
    a visibility-only save would otherwise be dropped.
    """
    exporter.register_org(ORG, **CREDS, traces_public=True)
    exporter.register_org(ORG, **CREDS, traces_public=False)
    set_current_org_id(ORG)

    assert resolve(None) is False


def test_org_without_own_project_follows_the_env_flag(exporter):
    """Unregistered orgs export to Dograh's project, so its policy applies."""
    set_current_org_id(ORG)

    assert resolve("true") is True
    assert resolve(None) is False


def test_unregistering_hands_the_org_back_to_the_env_flag(exporter):
    exporter.register_org(ORG, **CREDS, traces_public=False)
    exporter.unregister_org(ORG)
    set_current_org_id(ORG)

    assert resolve("true") is True


def test_spans_without_org_context_follow_the_env_flag(exporter):
    exporter.register_org(ORG, **CREDS, traces_public=False)

    assert resolve("true") is True
    assert resolve(None) is False


def test_visibility_agrees_with_routing(exporter):
    """The predicate must match the one ``export`` routes on.

    If they drift, a span can be marked public against one project's policy and
    then delivered to the other.
    """
    exporter.register_org(ORG, **CREDS, traces_public=True)

    for org_id, routed_to_own_project in ((ORG, True), ("other-org", False)):
        set_current_org_id(org_id)
        assert exporter.has_org(org_id) is routed_to_own_project
        assert resolve(None) is routed_to_own_project


# --- credential change detection ---------------------------------------------
class _BareExporter:
    """An exporter that exposes no private state, like current OpenTelemetry."""

    instances: list = []

    def __init__(self, endpoint, headers):
        self.shutdown_calls = 0
        type(self).instances.append(self)

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def bare_exporters():
    _BareExporter.instances = []
    with patch.object(tracing_config, "OTLPSpanExporter", _BareExporter):
        yield _BareExporter


def test_identical_registration_reuses_the_exporter_without_private_state(
    exporter, bare_exporters
):
    exporter.register_org(ORG, **CREDS, traces_public=True)
    exporter.register_org(ORG, **CREDS, traces_public=False)

    assert len(bare_exporters.instances) == 1
    assert bare_exporters.instances[0].shutdown_calls == 0


def test_rotated_secret_rebuilds_the_exporter_and_shuts_the_old_one_down(
    exporter, bare_exporters
):
    exporter.register_org(ORG, **CREDS)
    exporter.register_org(ORG, **{**CREDS, "secret_key": "sk-lf-rotated"})

    assert len(bare_exporters.instances) == 2
    assert bare_exporters.instances[0].shutdown_calls == 1
    assert exporter._org_exporters[str(ORG)] is bare_exporters.instances[1]
    # And the new value is now the baseline: re-saving it is a no-op.
    exporter.register_org(ORG, **{**CREDS, "secret_key": "sk-lf-rotated"})
    assert len(bare_exporters.instances) == 2


def test_rotated_public_key_or_host_rebuilds_the_exporter(exporter, bare_exporters):
    exporter.register_org(ORG, **CREDS)
    exporter.register_org(ORG, **{**CREDS, "public_key": "pk-lf-other"})
    exporter.register_org(
        ORG,
        **{
            **CREDS,
            "public_key": "pk-lf-other",
            "host": "https://eu.langfuse.example.com",
        },
    )

    assert len(bare_exporters.instances) == 3


def test_unregister_forgets_the_credential_so_reconnecting_rebuilds(
    exporter, bare_exporters
):
    exporter.register_org(ORG, **CREDS)
    exporter.unregister_org(ORG)
    assert str(ORG) not in exporter._org_auth_digests
    exporter.register_org(ORG, **CREDS)

    assert len(bare_exporters.instances) == 2


def test_the_secret_is_neither_stored_nor_logged(exporter, bare_exporters):
    import base64

    from loguru import logger

    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
    try:
        exporter.register_org(ORG, **CREDS)
        exporter.register_org(ORG, **{**CREDS, "secret_key": "sk-lf-rotated"})
        exporter.unregister_org(ORG)
    finally:
        logger.remove(sink)
    auth = base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    stored = repr(vars(exporter))
    for secret in ("sk-lf-test", "sk-lf-rotated", auth):
        assert secret not in stored
        assert all(secret not in m for m in messages)
