"""Repository contracts for face vectors."""

from __future__ import annotations

from typing import Protocol


class FaceVectorRepository(Protocol):
    def ensure_schema(self) -> None:
        """Prepare required DB extensions/tables for indexing."""

    def save_many(self, records: list[dict], upsert: bool = False) -> tuple[int, int]:
        """Persist records and return (inserted, updated)."""
