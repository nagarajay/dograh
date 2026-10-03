"""Add claim ownership and leases to Gemini-TTS sample assets.

A worker claim now carries a random token and a lease. Work abandoned by a dead
worker is recognised by an expired lease, and a worker whose claim was replaced
can no longer finalize the asset. Additive: existing rows and audio are kept.
"""

import sqlalchemy as sa
from alembic import op

revision = "a9c3e5b7d1f2"
down_revision = "e2f1a9c4b7d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gemini_tts_sample_assets",
        sa.Column("claim_token", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "gemini_tts_sample_assets",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "gemini_tts_sample_assets",
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "gemini_tts_sample_assets",
        sa.Column("enqueue_epoch", sa.Integer(), nullable=False, server_default="0"),
    )
    # A job running under the previous release has no lease. Give it ten minutes
    # to finish before the recovery sweep treats it as abandoned.
    op.execute(
        sa.text(
            "UPDATE gemini_tts_sample_assets "
            "SET lease_expires_at = now() + interval '10 minutes' "
            "WHERE status = 'running'"
        )
    )


def downgrade() -> None:
    op.drop_column("gemini_tts_sample_assets", "enqueue_epoch")
    op.drop_column("gemini_tts_sample_assets", "queued_at")
    op.drop_column("gemini_tts_sample_assets", "lease_expires_at")
    op.drop_column("gemini_tts_sample_assets", "claim_token")
