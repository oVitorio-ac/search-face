#!/usr/bin/env python3
"""Find facial matches in a folder of photos using a query image."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from app.core.domain.matcher import FaceMatcher
from app.core.domain.models import KnownPerson
from app.core.infrastructure.face_engine import FaceEngine


def _first_embedding(image_path: Path, model: str = "buffalo_l"):
    engine = FaceEngine()
    encodings = engine.encode_file(image_path, model=model)
    if not encodings:
        sys.exit(f"[ERROR] No face detected in query image: {image_path}")
    return encodings[0]


def _find_images(folder: Path) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    return [p.resolve() for p in folder.rglob("*") if p.suffix.lower() in supported]


def main():
    parser = argparse.ArgumentParser(
        description="Search a folder of photos for faces that match a query image."
    )
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--model", default="buffalo_l")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"[ERROR] Provided folder does not exist or is not a directory: {args.folder}")

    engine = FaceEngine()
    matcher = FaceMatcher()

    query_vec = _first_embedding(args.query, model=args.model)

    candidates = _find_images(args.folder)
    matches: List[Tuple[Path, float]] = []

    for img_path in candidates:
        try:
            encs = engine.encode_file(img_path, model=args.model)
        except Exception:
            continue
        if not encs:
            continue

        known_people = [KnownPerson(name=img_path.name, encoding=enc) for enc in encs]
        decision = matcher.best_match(query_vec, known_people, threshold=args.threshold)

        if args.top is not None:
            matches.append((img_path, decision.distance))
        elif decision.name != "unknown":
            matches.append((img_path, decision.distance))

    matches.sort(key=lambda x: x[1])
    if args.top is not None:
        matches = matches[: args.top]

    output = {
        "query": str(args.query.resolve()),
        "matches": [{"filename": str(p.name), "distance": round(d, 5)} for p, d in matches],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
