"""Shared constants for the Gemini-TTS sample worker and its recovery sweep."""

# Seconds a worker may own a ``running`` asset without renewing its lease. The
# provider call is bounded by 120 s; the worker renews well inside this window.
CLAIM_LEASE_SECONDS = 180
# How long a ``queued`` asset may exist before the sweep checks that a queue job
# still exists for it (covers the gap between commit and enqueue).
QUEUED_GRACE_SECONDS = 120
# Delay before a job that found every provider slot busy runs again. The job is
# rescheduled; it never waits inside an occupied worker slot.
SLOT_RETRY_DELAY_SECONDS = 3
# Each slot-busy deferral counts as an ARQ try; this bounds the total wait.
MAX_JOB_TRIES = 2000
JOB_TIMEOUT_SECONDS = 300
EXPIRED_LEASE_MESSAGE = (
    "worker_lost: the worker that claimed this sample stopped before finishing; "
    "no result was recorded. Retry the pack to generate a new version."
)


def sample_job_id(asset_id: int, epoch: int = 0) -> str:
    """ARQ job id. Epoch 0 keeps the original id; a re-enqueue gets a fresh one
    because ARQ retains finished-job keys and would refuse the old id."""
    base = f"gemini-tts-sample-asset-{asset_id}"
    return base if not epoch else f"{base}-e{epoch}"
