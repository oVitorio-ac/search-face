"""migrate face_vectors to 512d embeddings

Revision ID: 20260525_0002
Revises: 20260525_0001
Create Date: 2026-05-25 00:00:02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260525_0002"
down_revision = "20260525_0001"
branch_labels = None
depends_on = None


def _embedding_type_name(connection) -> str | None:
    query = sa.text(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = current_schema()
          AND c.relname = 'face_vectors'
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """
    )
    return connection.execute(query).scalar_one_or_none()


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


def _constraint_exists(connection, table_name: str, constraint_name: str) -> bool:
    query = sa.text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.table_constraints
            WHERE table_schema = current_schema()
              AND table_name = :table_name
              AND constraint_name = :constraint_name
        )
        """
    )
    return bool(
        connection.execute(
            query,
            {"table_name": table_name, "constraint_name": constraint_name},
        ).scalar_one()
    )


def _create_face_vectors_table() -> None:
    op.create_table(
        "face_vectors",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("face_index", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("vector_size", sa.Integer(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("file_path", "face_index", name="uq_face_vectors_file_face"),
    )


def upgrade() -> None:
    connection = op.get_bind()
    embedding_type = _embedding_type_name(connection)

    if embedding_type == "vector(512)":
        return

    if embedding_type is not None:
        legacy_table_name = "face_vectors_legacy_128"
        if not _table_exists(connection, legacy_table_name):
            op.rename_table("face_vectors", legacy_table_name)
            if _constraint_exists(connection, legacy_table_name, "uq_face_vectors_file_face"):
                op.execute(
                    sa.text(
                        """
                        ALTER TABLE face_vectors_legacy_128
                        RENAME CONSTRAINT uq_face_vectors_file_face
                        TO uq_face_vectors_legacy_128_file_face
                        """
                    )
                )
        else:
            op.drop_table("face_vectors")

    _create_face_vectors_table()


def downgrade() -> None:
    connection = op.get_bind()
    if _table_exists(connection, "face_vectors"):
        op.drop_table("face_vectors")

    if _table_exists(connection, "face_vectors_legacy_128"):
        if _constraint_exists(connection, "face_vectors_legacy_128", "uq_face_vectors_legacy_128_file_face"):
            op.execute(
                sa.text(
                    """
                    ALTER TABLE face_vectors_legacy_128
                    RENAME CONSTRAINT uq_face_vectors_legacy_128_file_face
                    TO uq_face_vectors_file_face
                    """
                )
            )
        op.rename_table("face_vectors_legacy_128", "face_vectors")
