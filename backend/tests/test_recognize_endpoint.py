import io
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app import create_app
from app.domain import FaceBox
from app.application.exceptions import NoFaceDetectedError


class RecognizeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()

    def test_happy_path(self):
        payload = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
            "known_images": [
                (io.BytesIO(b"k1"), "k1.jpg"),
                (io.BytesIO(b"k2"), "k2.jpg"),
            ],
            "known_names": ["alice", "bob"],
        }

        with patch("app.interface.http.recognize_controller.RecognizeFacesUseCase.execute") as mock_exec:
            mock_exec.return_value = {
                "faces": [
                    {
                        "polygon": {"top": 1, "right": 2, "bottom": 3, "left": 4},
                        "name": "alice",
                        "distance": 0.22,
                    }
                ]
            }
            response = self.app.post("/recognize", data=payload, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["faces"][0]["name"], "alice")

    def test_missing_fields(self):
        response = self.app.post("/recognize", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_mismatched_known_arrays(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
            "known_images": [(io.BytesIO(b"k1"), "k1.jpg")],
            "known_names": ["alice", "bob"],
        }
        response = self.app.post("/recognize", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_no_faces_query_returns_422(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
            "known_images": [(io.BytesIO(b"k1"), "k1.jpg")],
            "known_names": ["alice"],
        }

        with patch("app.interface.http.recognize_controller.RecognizeFacesUseCase.execute") as mock_exec:
            mock_exec.side_effect = NoFaceDetectedError("No face detected in query image")
            response = self.app.post("/recognize", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 422)

    def test_recognize_from_folder_happy_path(self):
        with TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "alice.jpg").write_bytes(b"k1")
            (folder / "bob.png").write_bytes(b"k2")

            data = {
                "query_image": (io.BytesIO(b"query"), "query.jpg"),
                "folder_path": str(folder),
            }
            with patch("app.interface.http.recognize_controller.RecognizeFacesUseCase.execute") as mock_exec:
                mock_exec.return_value = {
                    "faces": [
                        {
                            "polygon": {"top": 1, "right": 2, "bottom": 3, "left": 4},
                            "name": "alice",
                            "distance": 0.2,
                        }
                    ]
                }
                response = self.app.post("/recognize-from-folder", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["faces"][0]["name"], "alice")

    def test_recognize_from_folder_requires_valid_folder(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
            "folder_path": "/path/that/does/not/exist",
        }
        response = self.app.post("/recognize-from-folder", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_face_polygons_happy_path(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
        }
        with patch("app.interface.http.recognize_controller.FaceEngine.detect_and_encode") as mock_detect:
            mock_detect.return_value = [
                (FaceBox(1, 2, 3, 4), object()),
                (FaceBox(10, 20, 30, 40), object()),
            ]
            response = self.app.post("/face-polygons", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["polygons"]), 2)
        self.assertEqual(response.json["polygons"][0]["top"], 1)

    def test_face_polygons_without_face_returns_422(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
        }
        with patch("app.interface.http.recognize_controller.FaceEngine.detect_and_encode") as mock_detect:
            mock_detect.return_value = []
            response = self.app.post("/face-polygons", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 422)

    def test_face_quality_happy_path(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
        }
        with patch("app.interface.http.recognize_controller.FaceEngine.detect_with_landmarks") as mock_detect:
            mock_detect.return_value = [
                (
                    FaceBox(10, 170, 170, 10),
                    {
                        "left_eye": [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)],
                        "right_eye": [(4, 1), (5, 1), (6, 1), (4, 2), (5, 2), (6, 2)],
                        "nose_tip": [(3, 3), (4, 3), (5, 3)],
                    },
                )
            ]
            response = self.app.post("/face-quality", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["faces"][0]["quality"], "good")

    def test_face_quality_no_face_returns_422(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
        }
        with patch("app.interface.http.recognize_controller.FaceEngine.detect_with_landmarks") as mock_detect:
            mock_detect.return_value = []
            response = self.app.post("/face-quality", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 422)

    def test_search_from_db_happy_path(self):
        data = {
            "query_image": (io.BytesIO(b"query"), "query.jpg"),
            "top": "5",
            "threshold": "0.6",
            "model": "hog",
        }

        fake_rows = [
            SimpleNamespace(
                name="alice",
                file_path="/faces/alice.jpg",
                file_name="alice.jpg",
                face_index=0,
                model="hog",
                vector_size=128,
                distance=0.23,
            )
        ]

        class FakeResult:
            def all(self):
                return fake_rows

        class FakeSession:
            def execute(self, stmt):
                return FakeResult()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("app.api.http.recognize_controller.FaceEngine.detect_and_encode") as mock_detect:
            mock_detect.return_value = [(FaceBox(1, 2, 3, 4), np.array([0.0, 0.0], dtype=np.float32))]
            with patch("app.api.http.recognize_controller.create_session_factory") as mock_factory:
                mock_factory.return_value = lambda: FakeSession()
                response = self.app.post("/search-from-db", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["matches"][0]["name"], "alice")
        self.assertEqual(response.json["matches"][0]["distance"], 0.23)

    def test_search_from_db_requires_query_image(self):
        response = self.app.post("/search-from-db", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
