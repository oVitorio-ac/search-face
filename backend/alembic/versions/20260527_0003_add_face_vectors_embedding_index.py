"""add pgvector index for face_vectors embeddings

Revision ID: 20260527_0003
Revises: 20260525_0002
Create Date: 2026-05-27 00:00:03
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260527_0003"
down_revision = "20260525_0002"
branch_labels = None
depends_on = None


def _index_exists(connection, index_name: str) -> bool:
    query = sa.text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = :index_name
        )
        """
    )
    return bool(connection.execute(query, {"index_name": index_name}).scalar_one())


def upgrade() -> None:
    connection = op.get_bind()
    if _index_exists(connection, "ix_face_vectors_embedding_l2"):
        return

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_face_vectors_embedding_l2
        ON face_vectors
        USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_face_vectors_embedding_l2")
