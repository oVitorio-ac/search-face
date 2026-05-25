"""Backward-compatible imports for domain layer."""

from app.core.domain import FaceBox, FaceMatchResult, FaceMatcher, KnownPerson, MatchDecision

__all__ = ["FaceBox", "FaceMatchResult", "FaceMatcher", "KnownPerson", "MatchDecision"]
