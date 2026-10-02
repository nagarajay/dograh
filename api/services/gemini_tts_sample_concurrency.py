"""Cross-process provider slots. Waiting never issues or retries a Google request."""

import asyncio
import os
from contextlib import asynccontextmanager
from uuid import uuid4

LEASE_SECONDS = 180
_RENEW = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


@asynccontextmanager
async def sample_provider_slot(redis):
    count = int(os.getenv("GEMINI_TTS_SAMPLE_CONCURRENCY", "1"))
    if not 1 <= count <= 10:
        raise ValueError("GEMINI_TTS_SAMPLE_CONCURRENCY must be between 1 and 10")
    token = uuid4().hex
    key = None
    while key is None:
        for index in range(count):
            candidate = f"gemini-tts-samples:provider-slot:{index}"
            if await redis.set(candidate, token, nx=True, ex=LEASE_SECONDS):
                key = candidate
                break
        if key is None:
            await asyncio.sleep(0.25)

    owner = asyncio.current_task()

    async def renew():
        try:
            while True:
                await asyncio.sleep(LEASE_SECONDS / 3)
                if not await redis.eval(_RENEW, 1, key, token, LEASE_SECONDS):
                    raise RuntimeError("Sample provider slot lease lost")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - any lease failure must cancel provider work
            # Fail closed: cancel the provider request if global ownership is lost.
            owner.cancel()

    heartbeat = asyncio.create_task(renew())
    try:
        yield
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        await redis.eval(_RELEASE, 1, key, token)
