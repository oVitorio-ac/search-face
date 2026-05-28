"""HTTP controller for face recognition endpoints."""

from pathlib import Path

from flask import Blueprint, jsonify, request
from sqlalchemy import select

from app.core.application import (
    InvalidImageError,
    InvalidInputError,
    NoFaceDetectedError,
    RecognizeFacesUseCase,
)
from app.core.domain import FaceMatcher
from app.core.infrastructure import FaceEngine
from app.core.persistence import create_session_factory
from app.core.persistence.models import FaceVector

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
    landmark_style = landmarks.get("landmark_style")

    if not left_eye or not right_eye:
        reasons.append("missing_eye_landmarks")
    elif landmark_style == "dlib_68":
        # With dlib landmarks, eye shape is more stable and useful for a simple profile heuristic.
        if abs(len(left_eye) - len(right_eye)) >= 2:
            reasons.append("possible_side_profile")

    if not nose_tip:
        reasons.append("partial_nose_landmarks")
    elif landmark_style == "dlib_68" and len(nose_tip) < 3:
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
        faces = engine.detect_with_landmarks(query_image.read())
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
        faces = engine.detect_and_encode(query_image.read())
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
        threshold = float(request.form.get("threshold", 0.6))
        top_raw = request.form.get("top")
        model = request.form.get("model", "hog").strip().lower()

        if query_image is None:
            raise InvalidInputError("Missing required field: query_image")
        if not folder_path:
            raise InvalidInputError("Missing required field: folder_path")
        if model not in {"hog", "cnn"}:
            raise InvalidInputError("model must be 'hog' or 'cnn'")

        top = None
        if top_raw not in (None, ""):
            top = int(top_raw)
            if top <= 0:
                raise InvalidInputError("top must be greater than zero")

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

        use_case = RecognizeFacesUseCase(face_engine=FaceEngine(), matcher=FaceMatcher(), threshold=threshold, model=model)
        payload = use_case.execute(query_image.read(), known_items)
        if top is not None:
            payload["faces"] = payload["faces"][:top]
        return jsonify(payload), 200

    except (InvalidInputError, InvalidImageError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        return jsonify({"error": "Internal server error"}), 500


@recognize_bp.post("/search-from-db")
def search_faces_from_db():
    try:
        query_image = request.files.get("query_image")
        top_raw = request.form.get("top", "10")
        threshold_raw = request.form.get("threshold", "0.6")
        model = request.form.get("model", "hog").strip().lower()

        if query_image is None:
            raise InvalidInputError("Missing required field: query_image")
        if model not in {"hog", "cnn"}:
            raise InvalidInputError("model must be 'hog' or 'cnn'")

        try:
            top = int(top_raw)
        except ValueError as exc:
            raise InvalidInputError("top must be an integer") from exc
        if top <= 0:
            raise InvalidInputError("top must be greater than zero")

        try:
            threshold = float(threshold_raw)
        except ValueError as exc:
            raise InvalidInputError("threshold must be a number") from exc

        engine = FaceEngine()
        query_faces = engine.detect_and_encode(query_image.read(), model=model)
        if not query_faces:
            raise NoFaceDetectedError("No face detected in query image")

        query_vector = query_faces[0][1].tolist()
        session_factory = create_session_factory()

        with session_factory() as session:
            distance_expr = FaceVector.embedding.l2_distance(query_vector)
            stmt = (
                select(
                    FaceVector.name,
                    FaceVector.file_path,
                    FaceVector.file_name,
                    FaceVector.face_index,
                    FaceVector.model,
                    FaceVector.vector_size,
                    distance_expr.label("distance"),
                )
                .order_by(distance_expr)
                .limit(top)
            )
            rows = session.execute(stmt).all()

        matches = []
        for row in rows:
            if float(row.distance) > threshold:
                continue
            matches.append(
                {
                    "name": row.name,
                    "file_path": row.file_path,
                    "file_name": row.file_name,
                    "face_index": row.face_index,
                    "model": row.model,
                    "vector_size": row.vector_size,
                    "distance": round(float(row.distance), 6),
                }
            )

        return jsonify({"query": query_image.filename or "query_image", "matches": matches}), 200

    except (InvalidInputError, InvalidImageError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        return jsonify({"error": "Internal server error"}), 500
