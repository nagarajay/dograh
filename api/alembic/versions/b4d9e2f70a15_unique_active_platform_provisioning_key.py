"""enforce one active platform provisioning api key per organization

Revision ID: b4d9e2f70a15
Revises: a7c2f1e94b38
Create Date: 2026-09-05 12:00:00.000000

``POST /superuser/organizations/{id}/api-keys`` mints the credential AVSIQ
authenticates with. A lost response used to leave the caller with no key and the
organization with a live one nobody holds, and every retry added another. The
endpoint now replaces a reserved-name key rather than adding to it, and this
index is what makes "replaces" true under concurrency: two simultaneous retries
cannot both end up active, because the second insert violates the index and is
retried against the state the first one committed.

Deliberately partial, and narrow in three ways at once -- only the reserved
name, only active rows, only unarchived ones. Organizations already hold
multiple keys with duplicate names (``Default API Key`` is created for every
organization), so a broader index could not be created on existing data, and
archived keys must stay unique-free so a rotated key can keep its name forever.

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4d9e2f70a15"
down_revision: Union[str, None] = "a7c2f1e94b38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must equal api.constants.PLATFORM_PROVISIONING_API_KEY_NAME. Written out
# rather than imported: a migration is a historical record, and importing a
# constant would make this file's meaning change when that constant does.
_RESERVED_NAME = "platform-provisioning"


def upgrade() -> None:
    op.create_index(
        "uq_api_keys_active_platform_provisioning",
        "api_keys",
        ["organization_id"],
        unique=True,
        postgresql_where=sa.text(
            f"name = '{_RESERVED_NAME}' AND is_active AND archived_at IS NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_api_keys_active_platform_provisioning", table_name="api_keys"
    )
