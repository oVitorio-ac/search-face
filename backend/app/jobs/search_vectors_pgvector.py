#!/usr/bin/env python3
"""Search similar face vectors stored in PostgreSQL with pgvector."""

import argparse
import json
import os
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.infrastructure.face_engine import FaceEngine
from app.core.persistence import create_session_factory
from app.core.persistence.models import FaceVector


def main():
    parser = argparse.ArgumentParser(
        description="Search indexed face vectors for the closest matches to a query image."
    )
    parser.add_argument("--query", required=True, type=Path, help="Query image")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return")
    parser.add_argument("--threshold", type=float, default=0.6, help="Optional max distance")
    parser.add_argument(
        "--dsn",
        "--pg-dsn",
        dest="dsn",
        default=os.getenv("DATABASE_URL")
        or os.getenv("PG_DSN")
        or "postgresql+psycopg://postgres:postgres@localhost:5433/face_search",
        help="Database DSN. DATABASE_URL is preferred.",
    )
    parser.add_argument("--model", choices=["hog", "cnn"], default=os.getenv("FACE_MODEL", "hog"))
    args = parser.parse_args()

    if not args.query.is_file():
        sys.exit(f"[ERROR] Query image not found: {args.query}")
    if args.top <= 0:
        sys.exit("[ERROR] --top must be greater than zero")

    face_engine = FaceEngine()
    encodings = face_engine.encode_file(args.query, model=args.model)
    if not encodings:
        sys.exit("[ERROR] No face found in query image")

    query_vector = encodings[0].tolist()
    session_factory = create_session_factory(dsn=args.dsn)

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
            .limit(args.top)
        )

        rows = session.execute(stmt).all()

    matches = []
    for row in rows:
        if args.threshold is not None and float(row.distance) > args.threshold:
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

    print(json.dumps({"query": str(args.query.resolve()), "matches": matches}, indent=2))


if __name__ == "__main__":
    main()
