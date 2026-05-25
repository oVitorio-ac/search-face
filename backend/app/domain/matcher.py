"""Domain matching rules."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.domain.models import KnownPerson


@dataclass(frozen=True)
class MatchDecision:
    name: str
    distance: float


class FaceMatcher:
    def all_matches(
        self,
        query_encoding: np.ndarray,
        known_people: Iterable[KnownPerson],
        threshold: float,
    ) -> list[MatchDecision]:
        candidates: list[MatchDecision] = []

        for person in known_people:
            distance = float(np.linalg.norm(query_encoding - person.encoding))
            if distance <= threshold:
                candidates.append(MatchDecision(name=person.name, distance=distance))

        candidates.sort(key=lambda item: item.distance)
        return candidates

    def best_match(
        self,
        query_encoding: np.ndarray,
        known_people: Iterable[KnownPerson],
        threshold: float,
    ) -> MatchDecision:
        best_name = "unknown"
        best_distance = float("inf")

        for person in known_people:
            distance = float(np.linalg.norm(query_encoding - person.encoding))
            if distance < best_distance:
                best_distance = distance
                best_name = person.name

        if best_distance == float("inf") or best_distance > threshold:
            return MatchDecision(name="unknown", distance=best_distance)

        return MatchDecision(name=best_name, distance=best_distance)
