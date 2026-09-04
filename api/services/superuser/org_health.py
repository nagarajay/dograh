"""Operational state of an organization, derived for super-admin review.

Every value returned here is *derived* — a boolean, a count, an enum, a
timestamp, a provider name. Organization configuration rows and telephony
configuration rows both store live credentials in their JSON columns
(``LANGFUSE_CREDENTIALS``, ``TELEPHONY_CONFIGURATION``,
``TelephonyConfigurationModel.credentials``), so nothing in this module may
pass a stored value through to a response. The allowlist is the function
bodies below: a new field is only added by writing code that computes it.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from api.db import db_client
from api.db.organization_configuration_client import LEASE_COMPLETED, LEASE_PENDING
from api.enums import OrganizationConfigurationKey
from api.services.organization_bootstrap import BOOTSTRAP_LEASE_STALE_AFTER

# Bootstrap states reported to the console.
BOOTSTRAP_NEVER = "never_started"
BOOTSTRAP_PENDING = "in_progress"
BOOTSTRAP_STALE = "stalled"
BOOTSTRAP_COMPLETED = "completed"


@dataclass
class TelephonyConfigurationState:
    id: int
    name: str
    provider: str
    is_default_outbound: bool
    inactive: bool
    inactive_since: Optional[datetime]
    inactive_reason: Optional[str]
    phone_number_count: int
    created_at: Optional[datetime]


@dataclass
class OrganizationOperationalState:
    bootstrap_state: str
    bootstrap_updated_at: Optional[datetime]
    model_configuration_present: bool
    model_configuration_updated_at: Optional[datetime]
    model_configuration_last_validated_at: Optional[datetime]
    langfuse_configured: bool
    active_api_key_count: int
    telephony_configurations: list[TelephonyConfigurationState] = field(
        default_factory=list
    )


def _bootstrap_state(value: object, updated_at: Optional[datetime]) -> str:
    if not isinstance(value, dict):
        return BOOTSTRAP_NEVER
    status = value.get("status")
    if status == LEASE_COMPLETED:
        return BOOTSTRAP_COMPLETED
    if status != LEASE_PENDING:
        return BOOTSTRAP_NEVER
    # A holder that died mid-provisioning leaves the lease pending forever;
    # past the takeover window that is a stall an operator should see, not
    # work still in flight.
    if updated_at is not None:
        age = datetime.now(updated_at.tzinfo) - updated_at
        if age > BOOTSTRAP_LEASE_STALE_AFTER:
            return BOOTSTRAP_STALE
    return BOOTSTRAP_PENDING


async def get_organization_operational_state(
    organization_id: int,
) -> OrganizationOperationalState:
    """Summarize whether an organization is actually provisioned and usable."""
    configurations = await db_client.get_all_configurations(organization_id)
    by_key = {config.key: config for config in configurations}

    bootstrap = by_key.get(OrganizationConfigurationKey.ORGANIZATION_BOOTSTRAP.value)
    model_config = by_key.get(OrganizationConfigurationKey.MODEL_CONFIGURATION_V2.value)
    langfuse = by_key.get(OrganizationConfigurationKey.LANGFUSE_CREDENTIALS.value)

    telephony_configurations = await db_client.list_telephony_configurations(
        organization_id
    )
    numbers_by_configuration: dict[int, int] = {}
    for configuration in telephony_configurations:
        numbers = await db_client.list_phone_numbers_for_config(configuration.id)
        numbers_by_configuration[configuration.id] = len(numbers)

    api_keys = await db_client.get_api_keys_by_organization(organization_id)

    return OrganizationOperationalState(
        bootstrap_state=_bootstrap_state(
            bootstrap.value if bootstrap else None,
            bootstrap.updated_at if bootstrap else None,
        ),
        bootstrap_updated_at=bootstrap.updated_at if bootstrap else None,
        model_configuration_present=bool(model_config and model_config.value),
        model_configuration_updated_at=(
            model_config.updated_at if model_config else None
        ),
        model_configuration_last_validated_at=(
            model_config.last_validated_at if model_config else None
        ),
        langfuse_configured=bool(langfuse and langfuse.value),
        active_api_key_count=sum(1 for key in api_keys if key.is_active),
        telephony_configurations=[
            TelephonyConfigurationState(
                id=configuration.id,
                name=configuration.name,
                provider=configuration.provider,
                is_default_outbound=bool(configuration.is_default_outbound),
                inactive=bool(configuration.inactive),
                inactive_since=configuration.inactive_since,
                inactive_reason=configuration.inactive_reason,
                phone_number_count=numbers_by_configuration.get(configuration.id, 0),
                created_at=configuration.created_at,
            )
            for configuration in telephony_configurations
        ],
    )
