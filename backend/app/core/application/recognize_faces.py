"""Use case for recognizing all faces from a query image."""

from dataclasses import asdict
from typing import Iterable, List

from app.core.application.exceptions import InvalidInputError, NoFaceDetectedError
from app.core.domain import FaceMatchResult, FaceMatcher, KnownPerson
from app.core.infrastructure import FaceEngine


class RecognizeFacesUseCase:
    def __init__(
        self,
        face_engine: FaceEngine,
        matcher: FaceMatcher,
        threshold: float = 0.6,
        model: str = "hog",
    ):
        self.face_engine = face_engine
        self.matcher = matcher
        self.threshold = threshold
        self.model = model

    def execute(
        self,
        query_image_bytes: bytes,
        known_items: Iterable[tuple[str, bytes]],
    ) -> dict:
        known_people = self._build_known_people(known_items)
        query_faces = self.face_engine.detect_and_encode(query_image_bytes, model=self.model)

        if not query_faces:
            raise NoFaceDetectedError("No face detected in query image")

        matches: List[FaceMatchResult] = []
        for box, query_encoding in query_faces:
            all_matches = self.matcher.all_matches(query_encoding, known_people, self.threshold)
            decision = self.matcher.best_match(query_encoding, known_people, self.threshold)
            matches.append(
                FaceMatchResult(
                    polygon=box,
                    name=decision.name,
                    distance=round(decision.distance, 5),
                    possible_matches=[
                        {"name": item.name, "distance": round(item.distance, 5)}
                        for item in all_matches
                    ],
                )
            )

        return {"faces": [self._to_dict(match) for match in matches]}

    def _build_known_people(self, known_items: Iterable[tuple[str, bytes]]) -> List[KnownPerson]:
        known_people: List[KnownPerson] = []

        for name, image_bytes in known_items:
            encoded = self.face_engine.detect_and_encode(image_bytes, model=self.model)
            if not encoded:
                continue
            for _, face_encoding in encoded:
                known_people.append(KnownPerson(name=name, encoding=face_encoding))

        if not known_people:
            raise InvalidInputError("No valid known faces were provided")

        return known_people

    def _to_dict(self, match: FaceMatchResult) -> dict:
        data = asdict(match)
        return {
            "polygon": data["polygon"],
            "name": data["name"],
            "distance": data["distance"],
            "possible_matches": data["possible_matches"],
        }
