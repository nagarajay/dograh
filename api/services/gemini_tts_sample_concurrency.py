"""Cross-process provider slots for sample generation.

Acquisition never waits: if every slot is taken the caller is told at once and
reschedules itself, so a busy provider never ties up an ARQ worker slot.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable
from uuid import uuid4

LEASE_SECONDS = 180
_RENEW = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


class ProviderSlotBusy(RuntimeError):
    """Every provider slot is in use right now."""


class _Slot:
    def __init__(self):
        self._checks: list[Callable[[], Awaitable[bool]]] = []

    def watch(self, check: Callable[[], Awaitable[bool]]) -> None:
        """Renew another lease on each heartbeat; losing it cancels the work."""
        self._checks.append(check)


def provider_slot_count() -> int:
    count = int(os.getenv("GEMINI_TTS_SAMPLE_CONCURRENCY", "1"))
    if not 1 <= count <= 10:
        raise ValueError("GEMINI_TTS_SAMPLE_CONCURRENCY must be between 1 and 10")
    return count


@asynccontextmanager
async def sample_provider_slot(redis):
    """Hold one global provider slot, or raise ``ProviderSlotBusy`` immediately."""
    count = provider_slot_count()
    token = uuid4().hex
    key = None
    for index in range(count):
        candidate = f"gemini-tts-samples:provider-slot:{index}"
        if await redis.set(candidate, token, nx=True, ex=LEASE_SECONDS):
            key = candidate
            break
    if key is None:
        raise ProviderSlotBusy("all Gemini-TTS sample provider slots are in use")

    owner = asyncio.current_task()
    slot = _Slot()

    async def renew():
        try:
            while True:
                await asyncio.sleep(LEASE_SECONDS / 3)
                if not await redis.eval(_RENEW, 1, key, token, LEASE_SECONDS):
                    raise RuntimeError("Sample provider slot lease lost")
                for check in slot._checks:
                    if not await check():
                        raise RuntimeError("Sample asset claim lost")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - any lease failure must cancel provider work
            # Fail closed: cancel the provider request if ownership is lost.
            owner.cancel()

    heartbeat = asyncio.create_task(renew())
    try:
        yield slot
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        await redis.eval(_RELEASE, 1, key, token)
