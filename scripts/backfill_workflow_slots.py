"""Targeted, dry-run-first snapshot of workflows into per-slot settings.

Today a workflow with no slot rows follows the organization default at call time.
This tool freezes *the configuration a workflow uses right now* into published
slot v1 rows (credentials encrypted into the credential store), so later
organization-default changes stop reaching it. It never changes what the workflow
runs with at the moment of the snapshot.

    python -m scripts.backfill_workflow_slots --workflow-ids 12,13            # dry run
    python -m scripts.backfill_workflow_slots --workflow-ids 12,13 --apply    # writes

Workflow ids are mandatory; there is no "all workflows" mode. Output carries
credential status only, never secret values. Slots that already have history are
skipped. --apply needs PROVIDER_CREDENTIAL_ENCRYPTION_KEYS (fails closed).
Run inside the API image, with the environment of the database you intend to touch.
"""

import argparse
import asyncio
import json
import sys

from api.db import db_client
from api.services.configuration.ai_model_configuration import (
    get_effective_ai_model_configuration_for_workflow,
)
from api.services.configuration.effective_readback import _describe_service
from api.services.configuration.slot_settings import (
    SLOTS,
    SlotSettingsError,
    WorkflowSlotService,
    slot_source,
)


async def _one(workflow_id: int, organization_id: int | None, apply: bool) -> dict:
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None or workflow.released_definition is None:
        return {"workflow_id": workflow_id, "error": "not found or not published"}
    configs = workflow.released_definition.workflow_configurations
    before = await get_effective_ai_model_configuration_for_workflow(
        organization_id=workflow.organization_id,
        workflow_configurations=configs,
        workflow_id=workflow.id,
    )
    service = WorkflowSlotService(db_client)
    states = {s.slot: s for s in await db_client.list_slot_states(workflow.id)}
    plan = []
    for item in await service.plan_snapshot(effective=before):
        state = states.get(item["slot"])
        skipped = bool(state and state.last_version)
        plan.append(
            {
                "slot": item["slot"],
                "action": "skip (already has slot history)" if skipped else "snapshot",
                "source_before": slot_source(
                    item["slot"], state.published_version if state else None, configs
                ),
                "source_after": "workflow_slot",
                "credential_kind": item["kind"],
                "blocked": item["blocked"],
                "effective_before": _describe_service(
                    getattr(before, item["slot"]), item["slot"]
                ),
            }
        )
    result = {
        "workflow_id": workflow.id,
        "workflow_uuid": workflow.workflow_uuid,
        "organization_id": workflow.organization_id,
        "plan": plan,
    }
    if apply:
        try:
            result["seeded_slots"] = await service.seed_workflow_from_effective(
                organization_id=workflow.organization_id,
                workflow_id=workflow.id,
                effective=before,
                origin="backfill",
                created_by="backfill_workflow_slots",
            )
        except SlotSettingsError as exc:
            result["error"] = exc.message
            return result
        after = await get_effective_ai_model_configuration_for_workflow(
            organization_id=workflow.organization_id,
            workflow_configurations=configs,
            workflow_id=workflow.id,
        )
        result["effective_unchanged"] = all(
            _describe_service(getattr(before, s), s)
            == _describe_service(getattr(after, s), s)
            for s in SLOTS
        )
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workflow-ids", required=True)
    parser.add_argument("--organization-id", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    ids = [int(p) for p in args.workflow_ids.split(",") if p.strip()]
    results = [await _one(i, args.organization_id, args.apply) for i in ids]
    print(
        json.dumps({"applied": args.apply, "workflows": results}, indent=2, default=str)
    )
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
