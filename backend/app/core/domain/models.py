"""Domain entities and value objects for face recognition."""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class FaceBox:
    top: int
    right: int
    bottom: int
    left: int


@dataclass(frozen=True)
class FaceMatchResult:
    polygon: FaceBox
    name: str
    distance: float
    possible_matches: List[Dict[str, Any]]


@dataclass(frozen=True)
class KnownPerson:
    name: str
    encoding: object
