"""create face_vectors table

Revision ID: 20260525_0001
Revises: 
Create Date: 2026-05-25 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260525_0001"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(connection, table_name: str) -> bool:
    query = sa.text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = :table_name
        )
        """
    )
    return bool(connection.execute(query, {"table_name": table_name}).scalar_one())


def upgrade() -> None:
    connection = op.get_bind()
    if _table_exists(connection, "face_vectors"):
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "face_vectors",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("face_index", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(128), nullable=False),
        sa.Column("vector_size", sa.Integer(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("file_path", "face_index", name="uq_face_vectors_file_face"),
    )


def downgrade() -> None:
    op.drop_table("face_vectors")
