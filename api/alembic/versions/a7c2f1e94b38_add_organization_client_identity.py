"""add organization display name and external reference

Revision ID: a7c2f1e94b38
Revises: f3a1c47b9e02
Create Date: 2026-09-04 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c2f1e94b38"
down_revision: Union[str, None] = "f3a1c47b9e02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("display_name", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("external_reference", sa.String(length=128), nullable=True),
    )
    # Partial unique index: the reference identifies exactly one Dograh
    # organization when present, while organizations that predate the column
    # (or are created without one) stay valid rather than colliding on NULL.
    op.create_index(
        "uq_organizations_external_reference",
        "organizations",
        ["external_reference"],
        unique=True,
        postgresql_where=sa.text("external_reference IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_organizations_external_reference", table_name="organizations")
    op.drop_column("organizations", "external_reference")
    op.drop_column("organizations", "display_name")
