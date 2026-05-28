"""Infrastructure adapter over dlib via face_recognition."""

from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import face_recognition
import numpy as np
import pillow_heif
from PIL import Image

from app.core.application.exceptions import InvalidImageError
from app.core.domain.models import FaceBox


pillow_heif.register_heif_opener()


class FaceEngine:
    def canonical_model_name(self, model: str | None) -> str:
        return self._resolve_detector_model(model)

    def detect_and_encode(self, image_bytes: bytes, model: str = "hog") -> List[Tuple[FaceBox, np.ndarray]]:
        image_array = self._load_image_from_bytes(image_bytes)
        locations = face_recognition.face_locations(image_array, model=self._resolve_detector_model(model))
        encodings = face_recognition.face_encodings(image_array, known_face_locations=locations)

        results: List[Tuple[FaceBox, np.ndarray]] = []
        for location, encoding in zip(locations, encodings):
            results.append((self._box_from_location(location), self._to_vector(encoding)))

        return results

    def encode_file(self, image_path: Path, model: str = "hog") -> List[np.ndarray]:
        if not image_path.is_file():
            raise FileNotFoundError(f"File not found: {image_path}")

        image = self._load_image_from_path(image_path)
        locations = face_recognition.face_locations(image, model=self._resolve_detector_model(model))
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)
        return [self._to_vector(encoding) for encoding in encodings]

    def detect_with_landmarks(self, image_bytes: bytes, model: str = "hog") -> List[Tuple[FaceBox, dict]]:
        image_array = self._load_image_from_bytes(image_bytes)
        locations = face_recognition.face_locations(image_array, model=self._resolve_detector_model(model))
        landmarks = face_recognition.face_landmarks(image_array, face_locations=locations)

        results: List[Tuple[FaceBox, dict]] = []
        for location, landmark in zip(locations, landmarks):
            results.append((self._box_from_location(location), self._landmarks_from_dlib(landmark)))

        return results

    def _load_image_from_bytes(self, image_bytes: bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                return np.array(img.convert("RGB"))
        except Exception as exc:
            raise InvalidImageError("Invalid or unsupported image") from exc

    def _load_image_from_path(self, image_path: Path) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                return np.array(img.convert("RGB"))
        except Exception as exc:
            raise InvalidImageError("Invalid or unsupported image") from exc

    def _resolve_detector_model(self, model: str | None) -> str:
        if model in {"", None}:
            return "hog"
        if model not in {"hog", "cnn"}:
            return "hog"
        return model

    def _box_from_location(self, location: tuple[int, int, int, int]) -> FaceBox:
        top, right, bottom, left = location
        return FaceBox(top=int(top), right=int(right), bottom=int(bottom), left=int(left))

    def _landmarks_from_dlib(self, landmarks: dict) -> dict:
        return {
            "left_eye": [tuple(point) for point in landmarks.get("left_eye", [])],
            "right_eye": [tuple(point) for point in landmarks.get("right_eye", [])],
            "nose_tip": [tuple(point) for point in landmarks.get("nose_tip", [])],
            "top_lip": [tuple(point) for point in landmarks.get("top_lip", [])],
            "bottom_lip": [tuple(point) for point in landmarks.get("bottom_lip", [])],
            "landmark_style": "dlib_68",
        }

    def _to_vector(self, encoding: np.ndarray) -> np.ndarray:
        return np.asarray(encoding, dtype=np.float32)
