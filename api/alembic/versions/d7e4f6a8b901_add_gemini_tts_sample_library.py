"""Add platform-owned Gemini-TTS sample packs and assets.

Additive and reversible. Existing workflows, provider credentials, and audio
artifacts are untouched.
"""

import sqlalchemy as sa
from alembic import op

revision = "d7e4f6a8b901"
down_revision = "c8f3a1d95e72"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gemini_tts_sample_packs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("catalog_revision", sa.String(length=64), nullable=False),
        sa.Column("location", sa.String(length=128), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("style_text", sa.Text(), nullable=False),
        sa.Column("sample_text", sa.Text(), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("request_fingerprint", name="uq_gemini_tts_sample_pack_fingerprint"),
    )
    op.create_index("ix_gemini_tts_sample_packs_id", "gemini_tts_sample_packs", ["id"])
    op.create_index(
        "ix_gemini_tts_sample_packs_model_status",
        "gemini_tts_sample_packs",
        ["model_id", "status"],
    )

    op.create_table(
        "gemini_tts_sample_assets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "pack_id",
            sa.Integer(),
            sa.ForeignKey("gemini_tts_sample_packs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("voice_id", sa.String(length=64), nullable=False),
        sa.Column("gender", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=True),
        sa.Column("playable_format", sa.String(length=16), nullable=True),
        sa.Column("mime_type", sa.String(length=64), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("generation_metadata", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("pack_id", "voice_id", name="uq_gemini_tts_sample_asset_voice"),
    )
    op.create_index("ix_gemini_tts_sample_assets_id", "gemini_tts_sample_assets", ["id"])
    op.create_index("ix_gemini_tts_sample_assets_pack_id", "gemini_tts_sample_assets", ["pack_id"])


def downgrade() -> None:
    op.drop_index("ix_gemini_tts_sample_assets_pack_id", table_name="gemini_tts_sample_assets")
    op.drop_index("ix_gemini_tts_sample_assets_id", table_name="gemini_tts_sample_assets")
    op.drop_table("gemini_tts_sample_assets")
    op.drop_index("ix_gemini_tts_sample_packs_model_status", table_name="gemini_tts_sample_packs")
    op.drop_index("ix_gemini_tts_sample_packs_id", table_name="gemini_tts_sample_packs")
    op.drop_table("gemini_tts_sample_packs")
