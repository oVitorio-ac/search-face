"""SQLAlchemy adapter for face vector persistence."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.persistence.models import FaceVector


class SQLAlchemyFaceVectorRepository:
    def __init__(self, session: Session):
        self.session = session

    def ensure_schema(self) -> None:
        # Specific DB setup kept as raw SQL because pgvector extension is DB-level.
        self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self.session.commit()

    def save_many(self, records: list[dict], upsert: bool = False) -> tuple[int, int]:
        if not records:
            return 0, 0

        if upsert:
            return self._upsert_many(records)

        rows = [self._to_row(record) for record in records]
        self.session.bulk_insert_mappings(FaceVector, rows)
        self.session.commit()
        return len(rows), 0

    def _upsert_many(self, records: list[dict]) -> tuple[int, int]:
        inserted = 0
        updated = 0

        for record in records:
            row = self._to_row(record)
            stmt = (
                insert(FaceVector)
                .values(**row)
                .on_conflict_do_update(
                    constraint="uq_face_vectors_file_face",
                    set_={
                        "name": row["name"],
                        "file_name": row["file_name"],
                        "embedding": row["embedding"],
                        "vector_size": row["vector_size"],
                        "model": row["model"],
                        "indexed_at": row["indexed_at"],
                    },
                )
                .returning(text("(xmax = 0) AS inserted"))
            )
            was_inserted = self.session.execute(stmt).scalar_one()
            if was_inserted:
                inserted += 1
            else:
                updated += 1

        self.session.commit()
        return inserted, updated

    def _to_row(self, record: dict) -> dict:
        return {
            "name": record["name"],
            "file_path": record["file_path"],
            "file_name": record["file_name"],
            "face_index": record["face_index"],
            "embedding": record["vector"],
            "vector_size": record["vector_size"],
            "model": record["model"],
            "indexed_at": record["indexed_at"],
        }
