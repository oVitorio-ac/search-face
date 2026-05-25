"""Backward-compatible imports for application layer."""

from app.core.application import InvalidImageError, InvalidInputError, NoFaceDetectedError, RecognizeFacesUseCase

__all__ = [
    "InvalidImageError",
    "InvalidInputError",
    "NoFaceDetectedError",
    "RecognizeFacesUseCase",
]
