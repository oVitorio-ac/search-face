import unittest

import numpy as np

from app.application.exceptions import InvalidInputError, NoFaceDetectedError
from app.application.recognize_faces import RecognizeFacesUseCase
from app.domain.matcher import FaceMatcher
from app.domain.models import FaceBox


class FakeFaceEngine:
    def __init__(self, query_faces, known_faces_map):
        self.query_faces = query_faces
        self.known_faces_map = known_faces_map

    def detect_and_encode(self, image_bytes, model="hog"):
        if image_bytes == b"query":
            return self.query_faces
        return self.known_faces_map.get(image_bytes, [])


class UseCaseTests(unittest.TestCase):
    def test_multiple_faces_with_mixed_results(self):
        query_faces = [
            (FaceBox(1, 2, 3, 4), np.array([0.0, 0.0])),
            (FaceBox(5, 6, 7, 8), np.array([3.0, 3.0])),
        ]
        known_faces_map = {
            b"known_a": [(FaceBox(0, 0, 0, 0), np.array([0.1, 0.1]))],
            b"known_b": [(FaceBox(0, 0, 0, 0), np.array([10.0, 10.0]))],
        }
        use_case = RecognizeFacesUseCase(
            face_engine=FakeFaceEngine(query_faces, known_faces_map),
            matcher=FaceMatcher(),
            threshold=0.6,
        )

        result = use_case.execute(b"query", [("alice", b"known_a"), ("bob", b"known_b")])

        self.assertEqual(len(result["faces"]), 2)
        self.assertEqual(result["faces"][0]["name"], "alice")
        self.assertEqual(result["faces"][1]["name"], "unknown")
        self.assertGreaterEqual(len(result["faces"][0]["possible_matches"]), 1)
        self.assertEqual(result["faces"][1]["possible_matches"], [])

    def test_raises_when_no_query_face(self):
        use_case = RecognizeFacesUseCase(
            face_engine=FakeFaceEngine([], {b"known": [(FaceBox(0, 1, 2, 3), np.array([0.0, 0.0]))]}),
            matcher=FaceMatcher(),
            threshold=0.6,
        )

        with self.assertRaises(NoFaceDetectedError):
            use_case.execute(b"query", [("alice", b"known")])

    def test_raises_when_no_valid_known_face(self):
        use_case = RecognizeFacesUseCase(
            face_engine=FakeFaceEngine([(FaceBox(1, 2, 3, 4), np.array([0.0, 0.0]))], {}),
            matcher=FaceMatcher(),
            threshold=0.6,
        )

        with self.assertRaises(InvalidInputError):
            use_case.execute(b"query", [("alice", b"invalid")])

    def test_uses_all_faces_from_known_images(self):
        query_faces = [(FaceBox(1, 2, 3, 4), np.array([0.49, 0.49]))]
        known_faces_map = {
            b"group": [
                (FaceBox(0, 0, 0, 0), np.array([2.0, 2.0])),
                (FaceBox(0, 0, 0, 0), np.array([0.5, 0.5])),
            ]
        }
        use_case = RecognizeFacesUseCase(
            face_engine=FakeFaceEngine(query_faces, known_faces_map),
            matcher=FaceMatcher(),
            threshold=0.6,
        )

        result = use_case.execute(b"query", [("alice", b"group")])

        self.assertEqual(result["faces"][0]["name"], "alice")

    def test_returns_all_possible_matches(self):
        query_faces = [(FaceBox(1, 2, 3, 4), np.array([0.0, 0.0]))]
        known_faces_map = {
            b"alice_img": [(FaceBox(0, 0, 0, 0), np.array([0.1, 0.1]))],
            b"bob_img": [(FaceBox(0, 0, 0, 0), np.array([0.2, 0.2]))],
            b"far_img": [(FaceBox(0, 0, 0, 0), np.array([5.0, 5.0]))],
        }
        use_case = RecognizeFacesUseCase(
            face_engine=FakeFaceEngine(query_faces, known_faces_map),
            matcher=FaceMatcher(),
            threshold=0.6,
        )

        result = use_case.execute(
            b"query",
            [("alice", b"alice_img"), ("bob", b"bob_img"), ("far", b"far_img")],
        )

        possible = result["faces"][0]["possible_matches"]
        self.assertEqual(len(possible), 2)
        self.assertEqual(possible[0]["name"], "alice")
        self.assertEqual(possible[1]["name"], "bob")


if __name__ == "__main__":
    unittest.main()
