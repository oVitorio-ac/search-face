#!/usr/bin/env python3
"""Utility to extract facial embeddings from a single image.

Usage:
    python -m app.get_vector --file /caminho/para/foto.jpg

The script loads the image, detects faces with ``face_recognition`` (default
model ``hog``) and prints a JSON object containing:

* ``num_faces`` – how many faces were detected.
* ``encodings`` – a list of 128‑dimensional vectors (list of floats).

If no face is found an empty list is returned.
"""

import argparse
import json
import sys
from pathlib import Path

import face_recognition
import pillow_heif

def extract_vectors(image_path: Path, model: str = "hog"):
    """Return the list of 128‑D face encodings for *image_path* using the selected model.

    ``face_recognition.face_encodings`` returns a list of ``np.ndarray`` objects.
    We convert each array to a plain Python list so the result can be serialized to JSON.
    """
    if not image_path.is_file():
        sys.exit(f"[ERROR] File not found: {image_path}")

    # Load the image (RGB) and compute encodings using the chosen model.
    image = face_recognition.load_image_file(str(image_path))
    encodings = face_recognition.face_encodings(image, model=model)
    # Convert ``np.ndarray`` -> list[float]
    encodings_as_lists = [enc.tolist() for enc in encodings]
    return encodings_as_lists


def main():
    parser = argparse.ArgumentParser(
        description="Extract facial embedding vectors from an image using face_recognition"
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the image (JPG, PNG, WebP, HEIC…) that contains a face.",
    )
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="Model to use for face detection (hog = fast, cnn = more accurate).",
    )
    args = parser.parse_args()

    vectors = extract_vectors(args.file, model=args.model)
    result = {
        "num_faces": len(vectors),
        "encodings": vectors,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
