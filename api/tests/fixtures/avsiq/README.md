# AVSIQ consumer-contract fixtures

`workflow_slots.schema.json` is a JSON Schema transcription of the TypeScript
types AVSIQ compiles against: `VoicePlatformWorkflowSlots`,
`VoicePlatformSlotStatus`, `VoicePlatformSlotVersion` and
`VoicePlatformSlotEffective` in
`avsiq-voiceagent-saas/packages/voice-platform/src/types.ts` (verified against
commit `be21c40`, 2026-10-03). `readback_*.json` are example responses AVSIQ
must be able to parse.

When AVSIQ changes those types, update the schema here in the same change. The
tests in `test_workflow_slot_settings.py` (`test_readback_*`) validate real
API responses against the schema and the examples against both the schema and
the minimum fields AVSIQ reads, so a drift fails here before it reaches AVSIQ.
