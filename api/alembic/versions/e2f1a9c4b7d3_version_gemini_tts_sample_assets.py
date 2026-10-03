"""Version Gemini-TTS samples without overwriting prior audio."""

import sqlalchemy as sa
from alembic import op

revision = "e2f1a9c4b7d3"
down_revision = "d7e4f6a8b901"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gemini_tts_sample_assets",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "gemini_tts_sample_assets",
        sa.Column(
            "is_current", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.execute(
        sa.text(
            "UPDATE gemini_tts_sample_assets SET is_current = TRUE "
            "WHERE status = 'completed'"
        )
    )
    op.drop_constraint(
        "uq_gemini_tts_sample_asset_voice",
        "gemini_tts_sample_assets",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_gemini_tts_sample_asset_version",
        "gemini_tts_sample_assets",
        ["pack_id", "voice_id", "version"],
    )
    op.create_index(
        "uq_gemini_tts_sample_asset_current",
        "gemini_tts_sample_assets",
        ["pack_id", "voice_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )
    op.create_index(
        "uq_gemini_tts_sample_asset_active",
        "gemini_tts_sample_assets",
        ["pack_id", "voice_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    # Safe rollback requires deleting history, so leave this migration applied.
    raise RuntimeError("Gemini-TTS sample version history cannot be downgraded safely")
