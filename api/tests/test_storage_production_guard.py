"""An explicitly enabled storage guard is enforced wherever storage is built.

MinIO is the local and OSS default: `ENABLE_AWS_S3` is unset in every
development environment, so a managed deployment that simply forgot the flag
would come up healthy and start writing greetings, opening audio and call
recordings into a container-local bucket that no backup covers. The loss is
silent, and is only discovered when the container is replaced.

`REQUIRE_EXTERNAL_S3_STORAGE` is the explicit opt-in. Unset, every environment
keeps its MinIO default. Set, MinIO is refused in `local` and `production` alike
-- an operator's explicit choice is not weakened by `ENVIRONMENT`. `test` never
builds a storage backend (it uses the null filesystem), so the guard is not
reached there.
"""

import pytest

from api.enums import Environment, StorageBackend
from api.services import storage as storage_module
from api.services.storage import (
    InsecureStorageConfigurationError,
    assert_durable_storage_configured,
)


def test_production_accepts_minio_when_strict_guard_disabled(monkeypatch):
    monkeypatch.setattr(storage_module, "ENVIRONMENT", Environment.PRODUCTION.value)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", False)

    assert_durable_storage_configured(StorageBackend.MINIO)


def test_strict_production_refuses_minio(monkeypatch):
    monkeypatch.setattr(storage_module, "ENVIRONMENT", Environment.PRODUCTION.value)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", True)

    with pytest.raises(InsecureStorageConfigurationError) as excinfo:
        assert_durable_storage_configured(StorageBackend.MINIO)

    # The message says what to set, because the person reading it is the person
    # who can fix it.
    assert "ENABLE_AWS_S3" in str(excinfo.value)
    assert "S3_BUCKET" in str(excinfo.value)


def test_production_refuses_s3_without_a_bucket(monkeypatch):
    monkeypatch.setattr(storage_module, "ENVIRONMENT", Environment.PRODUCTION.value)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", True)
    monkeypatch.setattr(storage_module, "S3_BUCKET", None)

    with pytest.raises(InsecureStorageConfigurationError):
        assert_durable_storage_configured(StorageBackend.S3)


def test_production_accepts_s3_with_a_bucket(monkeypatch):
    monkeypatch.setattr(storage_module, "ENVIRONMENT", Environment.PRODUCTION.value)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", True)
    monkeypatch.setattr(storage_module, "S3_BUCKET", "avsiq-media")

    assert_durable_storage_configured(StorageBackend.S3)


@pytest.mark.parametrize(
    "environment", [Environment.LOCAL.value, Environment.PRODUCTION.value]
)
def test_unset_guard_keeps_minio_in_every_environment(monkeypatch, environment):
    monkeypatch.setattr(storage_module, "ENVIRONMENT", environment)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", False)

    assert_durable_storage_configured(StorageBackend.MINIO)


@pytest.mark.parametrize(
    "environment", [Environment.LOCAL.value, Environment.PRODUCTION.value]
)
def test_enabled_guard_refuses_minio_in_every_environment(monkeypatch, environment):
    """Enabling the guard is explicit, so `local` does not get a silent pass."""
    monkeypatch.setattr(storage_module, "ENVIRONMENT", environment)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", True)

    with pytest.raises(InsecureStorageConfigurationError):
        assert_durable_storage_configured(StorageBackend.MINIO)


def _import_storage_in_fresh_process(**env):
    """Import api.services.storage as a deployment would, with real module-level wiring."""
    import os
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", "import api.services.storage"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_enabled_guard_stops_startup_in_local_and_production():
    for environment in ("local", "production"):
        result = _import_storage_in_fresh_process(
            ENVIRONMENT=environment,
            REQUIRE_EXTERNAL_S3_STORAGE="true",
            ENABLE_AWS_S3="false",
        )
        assert result.returncode != 0, environment
        assert "InsecureStorageConfigurationError" in result.stderr, environment


def test_enabled_guard_is_not_reached_in_the_test_environment():
    """ENVIRONMENT=test uses the null filesystem and never builds a backend."""
    result = _import_storage_in_fresh_process(
        ENVIRONMENT="test",
        REQUIRE_EXTERNAL_S3_STORAGE="true",
        ENABLE_AWS_S3="false",
    )
    assert result.returncode == 0, result.stderr[-400:]


def test_unset_guard_lets_local_start_on_minio():
    result = _import_storage_in_fresh_process(
        ENVIRONMENT="local",
        REQUIRE_EXTERNAL_S3_STORAGE="false",
        ENABLE_AWS_S3="false",
        MINIO_PUBLIC_ENDPOINT="http://localhost:9000",
    )
    assert result.returncode == 0, result.stderr[-400:]
