import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.services import VectorIndexService


class FakeFaceEngine:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_file(self, image_path: Path, model: str = "hog"):
        value = self.mapping.get(image_path.name, [])
        if isinstance(value, Exception):
            raise value
        return value


class VectorIndexServiceTests(unittest.TestCase):
    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            service = VectorIndexService(face_engine=FakeFaceEngine({}))
            summary, records = service.index_folder(folder=folder)

            self.assertEqual(summary["images_found"], 0)
            self.assertEqual(summary["faces_indexed"], 0)
            self.assertEqual(records, [])

    def test_invalid_or_face_less_images_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "bad.jpg").write_bytes(b"x")
            (folder / "no_face.png").write_bytes(b"x")

            engine = FakeFaceEngine({"bad.jpg": RuntimeError("invalid"), "no_face.png": []})
            service = VectorIndexService(face_engine=engine)
            summary, records = service.index_folder(folder=folder)

            self.assertEqual(summary["images_found"], 2)
            self.assertEqual(summary["faces_indexed"], 0)
            self.assertEqual(summary["skipped_images"], 2)
            self.assertEqual(records, [])

    def test_multiple_faces_are_inserted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "alice.jpg").write_bytes(b"x")

            engine = FakeFaceEngine({"alice.jpg": [np.array([0.1, 0.2]), np.array([0.3, 0.4])]})
            service = VectorIndexService(face_engine=engine)

            summary, records = service.index_folder(folder=folder)

            self.assertEqual(summary["faces_indexed"], 2)
            self.assertEqual(len(records), 2)

    def test_records_keep_face_index_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "alice.jpg").write_bytes(b"x")

            engine = FakeFaceEngine({"alice.jpg": [np.array([0.1, 0.2]), np.array([0.3, 0.4])]})
            service = VectorIndexService(face_engine=engine)
            summary, records = service.index_folder(folder=folder)

            self.assertEqual(summary["faces_indexed"], 2)
            self.assertEqual(records[0]["face_index"], 0)
            self.assertEqual(records[1]["face_index"], 1)
            self.assertEqual(records[0]["file_name"], "alice.jpg")


if __name__ == "__main__":
    unittest.main()
