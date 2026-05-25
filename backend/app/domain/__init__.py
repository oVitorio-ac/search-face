"""Public API for domain layer."""

from .matcher import FaceMatcher, MatchDecision
from .models import FaceBox, FaceMatchResult, KnownPerson

__all__ = [
    "FaceBox",
    "FaceMatchResult",
    "KnownPerson",
    "FaceMatcher",
    "MatchDecision",
]
