"""Infrastructure adapter over face_recognition."""

from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import face_recognition
import numpy as np
from PIL import Image
import pillow_heif

from app.application.exceptions import InvalidImageError
from app.domain.models import FaceBox


class FaceEngine:
    def detect_and_encode(self, image_bytes: bytes, model: str = "hog") -> List[Tuple[FaceBox, np.ndarray]]:
        image_array = self._load_image_from_bytes(image_bytes)
        locations = face_recognition.face_locations(image_array, model=model)
        encodings = face_recognition.face_encodings(image_array, known_face_locations=locations)

        results: List[Tuple[FaceBox, np.ndarray]] = []
        for location, encoding in zip(locations, encodings):
            box = FaceBox(
                top=int(location[0]),
                right=int(location[1]),
                bottom=int(location[2]),
                left=int(location[3]),
            )
            results.append((box, encoding))

        return results

    def encode_file(self, image_path: Path, model: str = "hog") -> List[np.ndarray]:
        if not image_path.is_file():
            raise FileNotFoundError(f"File not found: {image_path}")
        image = face_recognition.load_image_file(str(image_path))
        return face_recognition.face_encodings(image, model=model)

    def detect_with_landmarks(self, image_bytes: bytes, model: str = "hog") -> List[Tuple[FaceBox, dict]]:
        image_array = self._load_image_from_bytes(image_bytes)
        locations = face_recognition.face_locations(image_array, model=model)
        landmarks = face_recognition.face_landmarks(image_array, face_locations=locations)

        results: List[Tuple[FaceBox, dict]] = []
        for location, landmark in zip(locations, landmarks):
            box = FaceBox(
                top=int(location[0]),
                right=int(location[1]),
                bottom=int(location[2]),
                left=int(location[3]),
            )
            results.append((box, landmark))

        return results

    def _load_image_from_bytes(self, image_bytes: bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                return np.array(img.convert("RGB"))
        except Exception as exc:
            raise InvalidImageError("Invalid or unsupported image") from exc
