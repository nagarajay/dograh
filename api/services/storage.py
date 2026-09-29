from loguru import logger

from api.constants import (
    ENABLE_AWS_S3,
    ENVIRONMENT,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_PUBLIC_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    S3_ADDRESSING_STYLE,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SIGNATURE_VERSION,
    REQUIRE_EXTERNAL_S3_STORAGE,
)
from api.enums import Environment, StorageBackend

from .filesystem import BaseFileSystem, MinioFileSystem, NullFileSystem, S3FileSystem


def get_storage_for_backend(backend: str) -> BaseFileSystem:
    """Get storage instance for a specific backend enum.

    Maps StorageBackend enum codes to actual storage implementations:
    - Code 1 (S3): AWS S3 via S3FileSystem
    - Code 2 (MINIO): MinIO via MinioFileSystem
    """
    # Code 2: MinIO implementation (local/OSS deployments)
    if backend == StorageBackend.MINIO.value:
        if not MINIO_PUBLIC_ENDPOINT:
            raise ValueError(
                "MINIO_PUBLIC_ENDPOINT is required for MinIO storage. "
                "Set it to the full URL browsers use to reach MinIO, "
                "e.g. 'http://localhost:9000' for local dev or "
                "'https://your-server.example.com' for a remote deployment."
            )
        logger.info(
            f"Initializing {backend} storage at {MINIO_ENDPOINT} "
            f"(public: {MINIO_PUBLIC_ENDPOINT}) with bucket '{MINIO_BUCKET}'"
        )
        return MinioFileSystem(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            secure=MINIO_SECURE,
            public_endpoint=MINIO_PUBLIC_ENDPOINT,
        )

    # Code 1: AWS S3 implementation (cloud deployments)
    elif backend == StorageBackend.S3.value:
        if not S3_BUCKET:
            raise ValueError(
                "S3_BUCKET environment variable is required when using S3 storage"
            )
        bucket = S3_BUCKET
        region = S3_REGION
        logger.info(
            f"Initializing {backend} storage with bucket '{bucket}' in region '{region}'"
        )
        return S3FileSystem(
            bucket_name=bucket,
            region_name=region,
            endpoint_url=S3_ENDPOINT_URL,
            signature_version=S3_SIGNATURE_VERSION,
            addressing_style=S3_ADDRESSING_STYLE,
        )

    # Future backend implementations can be added here:
    # elif backend == StorageBackend.GCS:  # Code 3
    #     return GoogleCloudFileSystem(...)
    # elif backend == StorageBackend.AZURE:  # Code 4
    #     return AzureBlobFileSystem(...)

    else:
        raise ValueError(f"Unknown storage backend: {backend}")


class InsecureStorageConfigurationError(RuntimeError):
    """Raised when a production deployment has no durable object storage."""


def assert_durable_storage_configured(backend: StorageBackend) -> None:
    """Fail closed when an enabled guard would write call audio to dev-only storage.

    MinIO is the local and OSS default: `ENABLE_AWS_S3` is unset in every
    development environment, so a managed deployment that simply forgot the
    flag would come up healthy and start writing greetings, opening audio and
    call recordings into a container-local bucket that no backup and no
    lifecycle policy covers. The loss is silent and is only discovered when the
    container is replaced.

    The guard is an explicit opt-in: `REQUIRE_EXTERNAL_S3_STORAGE=true`. When it
    is set, S3 (or an S3-compatible endpoint, which is what `S3_ENDPOINT_URL`
    is for) with a bucket is required and startup is refused otherwise. An
    operator who enables it gets it in every environment that builds a storage
    backend, `local` included -- it is not silently weakened by `ENVIRONMENT`.
    `ENVIRONMENT=test` never builds one (it uses the null filesystem), so the
    guard is not reached there. With the flag unset, nothing changes.
    """
    if not REQUIRE_EXTERNAL_S3_STORAGE:
        return

    if backend is not StorageBackend.S3:
        raise InsecureStorageConfigurationError(
            "REQUIRE_EXTERNAL_S3_STORAGE is enabled, which requires durable "
            f"object storage. The configured backend is '{backend.value}', which "
            "is the local development default. Set ENABLE_AWS_S3=true and "
            "S3_BUCKET (with S3_ENDPOINT_URL for an S3-compatible provider). "
            "MinIO is dev-only: audio written to it is not durable and is lost "
            "when the container is replaced."
        )

    if not S3_BUCKET:
        raise InsecureStorageConfigurationError(
            "REQUIRE_EXTERNAL_S3_STORAGE is enabled, which requires S3_BUCKET "
            "to be set."
        )


def get_current_storage_backend() -> StorageBackend:
    """Get the current storage backend enum."""
    return StorageBackend.get_current_backend()


# Create a single storage instance at module load time.
# In the test environment we skip the real backend so import doesn't require
# MinIO/S3 to be reachable; tests that need storage must inject a real fs.
if ENVIRONMENT == Environment.TEST.value:
    logger.info("ENVIRONMENT=test — using NullFileSystem (no storage backend)")
    storage_fs: BaseFileSystem = NullFileSystem()
else:
    _backend = StorageBackend.get_current_backend()
    # Fail closed before the backend is built when the managed-deployment guard
    # is explicitly enabled.
    assert_durable_storage_configured(_backend)
    logger.info(
        f"Initializing storage backend: {_backend.name} (value: {_backend.value}, ENABLE_AWS_S3={ENABLE_AWS_S3})"
    )
    storage_fs = get_storage_for_backend(_backend.value)


# For backward compatibility, keep get_storage() function
def get_storage() -> BaseFileSystem:
    """Get the module-level storage instance.

    Deprecated: Use 'from api.services.storage import storage_fs' instead.
    """
    return storage_fs
