"""merge upstream v1.47.0 migrations with organization-isolation migrations

The fork added a7c2f1e94b38 and b4d9e2f70a15 on top of an older upstream
revision; upstream v1.47.0 added its own chain ending in 3a7b91c5d402. The two
touch unrelated tables, so this revision only joins the heads. Databases at
either head upgrade through it; nothing is rewritten, so no stamped database
is invalidated.

Revision ID: e5c81a3f6b29
Revises: 3a7b91c5d402, b4d9e2f70a15
Create Date: 2026-09-29 00:00:00.000000
"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "e5c81a3f6b29"
down_revision: Union[str, Sequence[str], None] = (
    "3a7b91c5d402",
    "b4d9e2f70a15",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
