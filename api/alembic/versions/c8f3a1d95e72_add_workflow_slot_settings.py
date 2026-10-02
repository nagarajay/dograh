"""Add per-workflow model slot settings and encrypted provider credentials.

Additive only: three new tables, no change to existing tables or data. Existing
workflows have no rows and keep resolving their model configuration exactly as
before.

Revision ID: c8f3a1d95e72
Revises: e5c81a3f6b29
"""

import sqlalchemy as sa
from alembic import op

revision = "c8f3a1d95e72"
down_revision = "e5c81a3f6b29"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workflows",
        sa.Column("slot_template_status", sa.String(16), nullable=True),
    )
    op.create_table(
        "provider_credentials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "organization_id",
            sa.Integer(),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("credential_ref", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(128), nullable=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("ciphertext", sa.Text(), nullable=False),
        sa.Column("key_id", sa.String(16), nullable=False),
        sa.Column("source_ref", sa.String(255), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "organization_id",
            "credential_ref",
            "version",
            name="uq_provider_credentials_org_ref_version",
        ),
    )
    op.create_index("ix_provider_credentials_id", "provider_credentials", ["id"])
    op.create_index(
        "ix_provider_credentials_organization_id",
        "provider_credentials",
        ["organization_id"],
    )

    op.create_table(
        "workflow_slot_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workflow_id",
            sa.Integer(),
            sa.ForeignKey("workflows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("slot", sa.String(16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("credential_ref", sa.String(64), nullable=True),
        sa.Column("credential_version", sa.Integer(), nullable=True),
        sa.Column(
            "validation_status",
            sa.String(16),
            nullable=False,
            server_default="unvalidated",
        ),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("based_on_version", sa.Integer(), nullable=True),
        sa.Column("origin", sa.String(24), nullable=False, server_default="edit"),
        sa.Column("change_note", sa.String(255), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "workflow_id", "slot", "version", name="uq_workflow_slot_settings_version"
        ),
    )
    op.create_index("ix_workflow_slot_settings_id", "workflow_slot_settings", ["id"])
    op.create_index(
        "ix_workflow_slot_settings_workflow_slot",
        "workflow_slot_settings",
        ["workflow_id", "slot"],
    )

    op.create_table(
        "workflow_slot_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workflow_id",
            sa.Integer(),
            sa.ForeignKey("workflows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("slot", sa.String(16), nullable=False),
        sa.Column("published_version", sa.Integer(), nullable=True),
        sa.Column("draft_version", sa.Integer(), nullable=True),
        sa.Column("last_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workflow_id", "slot", name="uq_workflow_slot_state"),
    )
    op.create_index("ix_workflow_slot_state_id", "workflow_slot_state", ["id"])


def downgrade() -> None:
    op.drop_index("ix_workflow_slot_state_id", table_name="workflow_slot_state")
    op.drop_table("workflow_slot_state")
    op.drop_index(
        "ix_workflow_slot_settings_workflow_slot", table_name="workflow_slot_settings"
    )
    op.drop_index("ix_workflow_slot_settings_id", table_name="workflow_slot_settings")
    op.drop_table("workflow_slot_settings")
    op.drop_index(
        "ix_provider_credentials_organization_id", table_name="provider_credentials"
    )
    op.drop_index("ix_provider_credentials_id", table_name="provider_credentials")
    op.drop_table("provider_credentials")
    op.drop_column("workflows", "slot_template_status")
