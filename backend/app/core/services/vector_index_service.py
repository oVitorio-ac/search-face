"""Service to extract face vectors from image folders."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class VectorIndexService:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

    def __init__(self, face_engine):
        self.face_engine = face_engine

    def find_images(self, folder: Path, recursive: bool = True) -> list[Path]:
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        return [path.resolve() for path in iterator if path.suffix.lower() in self.SUPPORTED_EXTENSIONS]

    def index_folder(self, folder: Path, model: str = "buffalo_l", recursive: bool = True):
        images = self.find_images(folder, recursive=recursive)
        now = datetime.now(timezone.utc)
        resolved_model = self._resolve_model_name(model)

        summary = {
            "images_found": len(images),
            "faces_indexed": 0,
            "skipped_images": 0,
        }
        records: list[dict] = []

        if not images:
            return summary, records

        for image_path in images:
            encodings = self._safe_encodings(image_path, model, summary)
            if not encodings:
                continue
            for face_index, encoding in enumerate(encodings):
                summary["faces_indexed"] += 1
                records.append(self._doc(image_path, face_index, encoding.tolist(), resolved_model, now))
        return summary, records

    def _safe_encodings(self, image_path: Path, model: str, summary: dict):
        try:
            encodings = self.face_engine.encode_file(image_path, model=model)
        except Exception:
            summary["skipped_images"] += 1
            return []

        if not encodings:
            summary["skipped_images"] += 1
            return []

        return encodings

    def _doc(self, image_path: Path, face_index: int, vector: list[float], model: str, indexed_at: datetime) -> dict:
        return {
            "name": image_path.stem,
            "file_path": str(image_path),
            "file_name": image_path.name,
            "face_index": face_index,
            "vector": vector,
            "vector_size": len(vector),
            "model": model,
            "indexed_at": indexed_at,
        }

    def _resolve_model_name(self, model: str) -> str:
        resolver = getattr(self.face_engine, "canonical_model_name", None)
        if callable(resolver):
            return resolver(model)
        return model
