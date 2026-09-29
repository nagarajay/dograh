"""Managed production may require external object storage.

MinIO is the local and OSS default: `ENABLE_AWS_S3` is unset in every
development environment, so a production deployment that simply forgot the flag
would come up healthy and start writing greetings, opening audio and call
recordings into a container-local bucket that no backup covers. The loss is
silent, and is only discovered when the container is replaced.

The guard reads `ENVIRONMENT`, which is the deployment's own existing
declaration — the same value that already selects the null filesystem under
`test`. Nothing is inferred from a hostname, a URL or the absence of a debugger.
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
    "environment", [Environment.LOCAL.value, Environment.TEST.value]
)
def test_local_and_test_keep_minio(monkeypatch, environment):
    """MinIO stays the development default. The guard is production's alone."""
    monkeypatch.setattr(storage_module, "ENVIRONMENT", environment)
    monkeypatch.setattr(storage_module, "REQUIRE_EXTERNAL_S3_STORAGE", True)

    assert_durable_storage_configured(StorageBackend.MINIO)
