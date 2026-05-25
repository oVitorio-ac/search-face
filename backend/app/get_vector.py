#!/usr/bin/env python3
"""Utility to extract facial embeddings from a single image."""

import argparse
import json
import sys
from pathlib import Path

from app.core.infrastructure.face_engine import FaceEngine


def extract_vectors(image_path: Path, model: str = "buffalo_l"):
    if not image_path.is_file():
        sys.exit(f"[ERROR] File not found: {image_path}")

    engine = FaceEngine()
    encodings = engine.encode_file(image_path, model=model)
    return [enc.tolist() for enc in encodings]


def main():
    parser = argparse.ArgumentParser(
        description="Extract facial embedding vectors from an image using InsightFace"
    )
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--model", default="buffalo_l")
    args = parser.parse_args()

    vectors = extract_vectors(args.file, model=args.model)
    result = {"num_faces": len(vectors), "encodings": vectors}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
