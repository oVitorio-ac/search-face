"""Infrastructure adapter over InsightFace."""

import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import insightface
import numpy as np
from PIL import Image

from app.core.application.exceptions import InvalidImageError
from app.core.domain.models import FaceBox


class FaceEngine:
    def __init__(self):
        self._apps: Dict[str, insightface.app.FaceAnalysis] = {}

    def canonical_model_name(self, model: str | None) -> str:
        return self._resolve_model_name(model)

    def detect_and_encode(self, image_bytes: bytes, model: str = "buffalo_l") -> List[Tuple[FaceBox, np.ndarray]]:
        image_array = self._load_image_from_bytes(image_bytes)
        faces = self._get_app(model).get(image_array)

        results: List[Tuple[FaceBox, np.ndarray]] = []
        for face in faces:
            box = self._box_from_face(face.bbox)
            results.append((box, self._normalize_embedding(face.embedding)))

        return results

    def encode_file(self, image_path: Path, model: str = "buffalo_l") -> List[np.ndarray]:
        if not image_path.is_file():
            raise FileNotFoundError(f"File not found: {image_path}")
        image = self._load_image_from_path(image_path)
        faces = self._get_app(model).get(image)
        return [self._normalize_embedding(face.embedding) for face in faces]

    def detect_with_landmarks(self, image_bytes: bytes, model: str = "buffalo_l") -> List[Tuple[FaceBox, dict]]:
        image_array = self._load_image_from_bytes(image_bytes)
        faces = self._get_app(model).get(image_array)

        results: List[Tuple[FaceBox, dict]] = []
        for face in faces:
            box = self._box_from_face(face.bbox)
            results.append((box, self._landmarks_from_face(face)))

        return results

    def _load_image_from_bytes(self, image_bytes: bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                return self._pil_to_bgr(img)
        except Exception as exc:
            raise InvalidImageError("Invalid or unsupported image") from exc

    def _load_image_from_path(self, image_path: Path) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                return self._pil_to_bgr(img)
        except Exception as exc:
            raise InvalidImageError("Invalid or unsupported image") from exc

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        rgb_array = np.array(image.convert("RGB"))
        return rgb_array[:, :, ::-1]

    def _get_app(self, model: str) -> insightface.app.FaceAnalysis:
        model_name = self._resolve_model_name(model)
        app = self._apps.get(model_name)
        if app is not None:
            return app

        providers = [
            provider.strip()
            for provider in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",")
            if provider.strip()
        ]
        det_size = int(os.getenv("FACE_DET_SIZE", "640"))
        ctx_id = int(os.getenv("FACE_CTX_ID", "0"))

        app = insightface.app.FaceAnalysis(name=model_name, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self._apps[model_name] = app
        return app

    def _resolve_model_name(self, model: str | None) -> str:
        if model in {"hog", "cnn", "", None}:
            return os.getenv("FACE_MODEL", "buffalo_l")
        return model

    def _box_from_face(self, bbox: np.ndarray) -> FaceBox:
        left, top, right, bottom = bbox.astype(int).tolist()
        return FaceBox(top=top, right=right, bottom=bottom, left=left)

    def _landmarks_from_face(self, face) -> dict:
        keypoints = getattr(face, "kps", None)
        if keypoints is None or len(keypoints) < 5:
            return {}

        left_eye = tuple(int(value) for value in keypoints[0])
        right_eye = tuple(int(value) for value in keypoints[1])
        nose = tuple(int(value) for value in keypoints[2])
        mouth_left = tuple(int(value) for value in keypoints[3])
        mouth_right = tuple(int(value) for value in keypoints[4])

        return {
            "left_eye": [left_eye],
            "right_eye": [right_eye],
            "nose_tip": [nose],
            "top_lip": [mouth_left],
            "bottom_lip": [mouth_right],
            "landmark_style": "insightface_5pt",
        }

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
