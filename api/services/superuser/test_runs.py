"""Marking and recognising super-admin test runs.

A super admin verifying an agent drives the same runtime a customer does, so
the run is a real row in ``workflow_runs`` inside the customer's organization.
That makes it visible to every organization-scoped query, including the ones
that report usage back to the customer — which is wrong, because the customer
did not make the call.

The marker lives in ``WorkflowRunModel.extra`` rather than in a new column or a
new run mode: ``extra`` already exists, and the run must stay an ordinary run
of its mode so nothing in the runtime dispatch changes. The SQL predicate that
excludes marked runs lives in ``api/db/filters.py``, since query construction
belongs to the database layer.

What this does *not* cover: model calls still bill against the organization's
own managed-services key while the test runs. Dograh has no non-billable
credential path, so a test costs the customer credits even though it is
excluded from Dograh's own usage reporting.
"""

from typing import Any

SUPERADMIN_TEST_FLAG = "superadmin_test"
SUPERADMIN_TEST_INITIATOR = "superadmin_initiated_by_user_id"


def superadmin_test_run_extra(superuser_id: int) -> dict[str, Any]:
    """The ``extra`` payload stamped on a run started from the super-admin console."""
    return {
        SUPERADMIN_TEST_FLAG: True,
        SUPERADMIN_TEST_INITIATOR: superuser_id,
    }


def is_superadmin_test_run(extra: dict[str, Any] | None) -> bool:
    return bool(extra) and extra.get(SUPERADMIN_TEST_FLAG) is True


def superadmin_test_run_initiator(extra: dict[str, Any] | None) -> int | None:
    """The superuser id that started this test run, if it is one."""
    if not is_superadmin_test_run(extra):
        return None
    initiator = extra.get(SUPERADMIN_TEST_INITIATOR)
    return initiator if isinstance(initiator, int) else None
