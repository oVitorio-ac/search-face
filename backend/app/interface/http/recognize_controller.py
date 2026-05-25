"""HTTP controller for face recognition endpoints."""

from pathlib import Path

from flask import Blueprint, jsonify, request

from app.application import (
    InvalidImageError,
    InvalidInputError,
    NoFaceDetectedError,
    RecognizeFacesUseCase,
)
from app.domain import FaceMatcher
from app.infrastructure import FaceEngine

recognize_bp = Blueprint("recognize", __name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def _assess_face_quality(box, landmarks):
    width = box.right - box.left
    height = box.bottom - box.top
    reasons = []

    if min(width, height) < 60:
        reasons.append("small_face")

    left_eye = landmarks.get("left_eye", [])
    right_eye = landmarks.get("right_eye", [])
    nose_tip = landmarks.get("nose_tip", [])

    if not left_eye or not right_eye:
        reasons.append("missing_eye_landmarks")
    else:
        # Eye landmark imbalance is a simple heuristic for side-profile faces.
        if abs(len(left_eye) - len(right_eye)) >= 2:
            reasons.append("possible_side_profile")

    if len(nose_tip) < 3:
        reasons.append("partial_nose_landmarks")

    if "small_face" in reasons or "missing_eye_landmarks" in reasons:
        quality = "bad"
    elif "possible_side_profile" in reasons or "partial_nose_landmarks" in reasons:
        quality = "warning"
    else:
        quality = "good"

    return quality, reasons


@recognize_bp.post("/face-quality")
def get_face_quality():
    try:
        query_image = request.files.get("query_image")
        if query_image is None:
            raise InvalidInputError("Missing required field: query_image")

        engine = FaceEngine()
        faces = engine.detect_with_landmarks(query_image.read(), model="hog")
        if not faces:
            raise NoFaceDetectedError("No face detected in query image")

        payload_faces = []
        for box, landmarks in faces:
            quality, reasons = _assess_face_quality(box, landmarks)
            payload_faces.append(
                {
                    "polygon": {
                        "top": box.top,
                        "right": box.right,
                        "bottom": box.bottom,
                        "left": box.left,
                    },
                    "quality": quality,
                    "reasons": reasons,
                }
            )

        return jsonify({"faces": payload_faces}), 200

    except (InvalidInputError, InvalidImageError) as exc:
        return jsonify({"error": str(exc)}), 400
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        return jsonify({"error": "Internal server error"}), 500


@recognize_bp.post("/face-polygons")
def get_face_polygons():
    try:
        query_image = request.files.get("query_image")
        if query_image is None:
            raise InvalidInputError("Missing required field: query_image")

        engine = FaceEngine()
        faces = engine.detect_and_encode(query_image.read(), model="hog")
        if not faces:
            raise NoFaceDetectedError("No face detected in query image")

        polygons = [
            {
                "top": box.top,
                "right": box.right,
                "bottom": box.bottom,
                "left": box.left,
            }
            for box, _ in faces
        ]
        return jsonify({"polygons": polygons}), 200

    except (InvalidInputError, InvalidImageError) as exc:
        return jsonify({"error": str(exc)}), 400
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        return jsonify({"error": "Internal server error"}), 500


@recognize_bp.post("/recognize")
def recognize_faces():
    try:
        query_image = request.files.get("query_image")
        known_images = request.files.getlist("known_images")
        known_names = request.form.getlist("known_names")

        if query_image is None:
            raise InvalidInputError("Missing required field: query_image")

        if not known_images or not known_names:
            raise InvalidInputError("known_images and known_names are required")

        if len(known_images) != len(known_names):
            raise InvalidInputError("known_images and known_names must have the same length")

        known_items = []
        for image_file, name in zip(known_images, known_names):
            if not name.strip():
                raise InvalidInputError("known_names cannot contain empty values")
            known_items.append((name.strip(), image_file.read()))

        use_case = RecognizeFacesUseCase(face_engine=FaceEngine(), matcher=FaceMatcher(), threshold=0.6)
        payload = use_case.execute(query_image.read(), known_items)
        return jsonify(payload), 200

    except (InvalidInputError, InvalidImageError) as exc:
        return jsonify({"error": str(exc)}), 400
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        return jsonify({"error": "Internal server error"}), 500


@recognize_bp.post("/recognize-from-folder")
def recognize_faces_from_folder():
    try:
        query_image = request.files.get("query_image")
        folder_path = request.form.get("folder_path", "").strip()

        if query_image is None:
            raise InvalidInputError("Missing required field: query_image")
        if not folder_path:
            raise InvalidInputError("Missing required field: folder_path")

        folder = Path(folder_path)
        if not folder.is_dir():
            raise InvalidInputError("folder_path must be an existing directory")

        known_items = []
        for image_path in folder.rglob("*"):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                image_bytes = image_path.read_bytes()
            except OSError:
                continue
            known_items.append((image_path.stem, image_bytes))

        if not known_items:
            raise InvalidInputError("No supported image files found in folder_path")

        use_case = RecognizeFacesUseCase(face_engine=FaceEngine(), matcher=FaceMatcher(), threshold=0.6)
        payload = use_case.execute(query_image.read(), known_items)
        return jsonify(payload), 200

    except (InvalidInputError, InvalidImageError) as exc:
        return jsonify({"error": str(exc)}), 400
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        return jsonify({"error": "Internal server error"}), 500
