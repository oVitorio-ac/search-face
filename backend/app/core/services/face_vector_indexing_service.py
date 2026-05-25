"""Business service for vector extraction + persistence."""

from __future__ import annotations

from pathlib import Path

from app.core.repositories.face_vector_repository import FaceVectorRepository
from app.core.services.vector_index_service import VectorIndexService


class FaceVectorIndexingService:
    def __init__(self, extractor: VectorIndexService, repository: FaceVectorRepository):
        self.extractor = extractor
        self.repository = repository

    def execute(self, folder: Path, model: str = "buffalo_l", recursive: bool = True, upsert: bool = False) -> dict:
        summary, records = self.extractor.index_folder(folder=folder, model=model, recursive=recursive)
        if summary["images_found"] == 0:
            return {**summary, "inserted": 0, "updated": 0}

        self.repository.ensure_schema()
        inserted, updated = self.repository.save_many(records, upsert=upsert)

        return {
            **summary,
            "inserted": inserted,
            "updated": updated,
        }
