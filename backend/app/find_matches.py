#!/usr/bin/env python3
"""Find facial matches in a folder of photos using a query image.

The script extracts the 128‑dimensional face embedding from the **query** image
(and, if several faces are present, uses the first one).  It then scans a target
folder, extracts embeddings from every image and computes the Euclidean
distance between the query vector and each face found.

Only matches whose distance is **<= threshold** are reported.  The threshold
defaults to the same value used by the main API (0.6).  You can also request the
top‑N closest matches irrespective of the threshold.

Usage:
    python -m app.find_matches \
        --query /path/to/query.jpg \
        --folder /path/to/gallery \
        [--threshold 0.6] [--top 5]

Output is printed as JSON:
{
  "query": "query.jpg",
  "matches": [
    {"filename": "photo1.jpg", "distance": 0.3123},
    {"filename": "photo2.png", "distance": 0.4157}
  ]
}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import face_recognition
import pillow_heif

def _load_embeddings(image_path: Path, model: str = "hog") -> List[np.ndarray]:
    """Return a list of 128‑D face embeddings for *image_path* using the selected model.

    The function uses ``face_recognition.face_encodings`` which returns a list of
    ``np.ndarray`` objects (dtype=float64, shape=(128,)).  An empty list means no
    face was detected.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")
    image = face_recognition.load_image_file(str(image_path))
    return face_recognition.face_encodings(image, model=model)

def _first_embedding(image_path: Path, model: str = "hog") -> np.ndarray:
    """Extract the *first* face embedding from *image_path* using the selected model.

    If no face is found the function aborts with a clear error message – the
    query image must contain at least one detectable face.
    """
    encodings = _load_embeddings(image_path, model=model)
    if not encodings:
        sys.exit(f"[ERROR] No face detected in query image: {image_path}")
    return encodings[0]

def _find_images(folder: Path) -> List[Path]:
    """Recursively collect image files (JPG, PNG, WebP) from *folder*.
    Returns a list of absolute ``Path`` objects.
    """
    supported = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    return [p.resolve() for p in folder.rglob("*") if p.suffix.lower() in supported]

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two 128‑D vectors."""
    return np.linalg.norm(a - b)

def main():
    parser = argparse.ArgumentParser(
        description="Search a folder of photos for faces that match a query image."
    )
    parser.add_argument(
        "--query",
        required=True,
        type=Path,
        help="Path to the query image (must contain at least one face).",
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Directory containing candidate photos to compare against.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Maximum Euclidean distance to consider a match (default: 0.6).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="If set, return only the N closest matches regardless of threshold.",
    )
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="Model to use for face detection (hog = fast, cnn = more accurate).",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"[ERROR] Provided folder does not exist or is not a directory: {args.folder}")

    # 1️⃣ Load query embedding (first face only) using selected model
    query_vec = _first_embedding(args.query, model=args.model)

    # 2️⃣ Scan target folder
    candidates = _find_images(args.folder)
    matches: List[Tuple[Path, float]] = []

    for img_path in candidates:
        try:
            encs = _load_embeddings(img_path, model=args.model)
        except Exception as exc:
            # Skip files we cannot read – they are not critical for the search
            print(f"[WARN] Skipping {img_path.name}: {exc}", file=sys.stderr)
            continue
        if not encs:
            # No faces in this image → ignore
            continue
        # Compute distance to **each** face found and keep the smallest one
        distances = [_distance(query_vec, e) for e in encs]
        best_dist = min(distances)
        if args.top is not None:
            matches.append((img_path, best_dist))
        else:
            if best_dist <= args.threshold:
                matches.append((img_path, best_dist))

    # 3️⃣ Sort results by distance (ascending)
    matches.sort(key=lambda x: x[1])
    if args.top is not None:
        matches = matches[: args.top]

    # 4️⃣ Build JSON output
    output = {
        "query": str(args.query.resolve()),
        "matches": [
            {"filename": str(p.name), "distance": round(d, 5)} for p, d in matches
        ],
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
