"""Persistence adapters and repository contracts."""

from .face_vector_repository import FaceVectorRepository
from .sqlalchemy_face_vector_repository import SQLAlchemyFaceVectorRepository

__all__ = ["FaceVectorRepository", "SQLAlchemyFaceVectorRepository"]
