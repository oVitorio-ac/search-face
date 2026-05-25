#!/usr/bin/env python3
"""Index face vectors from a folder into PostgreSQL with pgvector."""

import argparse
import os
import sys
from pathlib import Path

from app.core.infrastructure.face_engine import FaceEngine
from app.core.persistence import create_session_factory
from app.core.repositories import SQLAlchemyFaceVectorRepository
from app.core.services import FaceVectorIndexingService, VectorIndexService


def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of images, extract face vectors, and store in pgvector."
    )
    parser.add_argument("--folder", required=True, type=Path, help="Folder containing images")
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv("PG_DSN", "postgresql+psycopg://postgres:postgres@localhost:5433/face_search"),
    )
    parser.add_argument(
        "--table",
        default=os.getenv("PG_TABLE", "face_vectors"),
        help="Deprecated: table is fixed by ORM entity and migrations",
    )
    parser.add_argument("--model", default=os.getenv("FACE_MODEL", "buffalo_l"))
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Upsert documents by file path + face index instead of always inserting",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        sys.exit(f"[ERROR] Folder does not exist or is not a directory: {folder}")
    if args.table != "face_vectors":
        print("[WARN] --table is deprecated and ignored. Using face_vectors.")

    session_factory = create_session_factory(dsn=args.pg_dsn)
    with session_factory() as session:
        repository = SQLAlchemyFaceVectorRepository(session=session)
        service = FaceVectorIndexingService(
            extractor=VectorIndexService(face_engine=FaceEngine()),
            repository=repository,
        )
        result = service.execute(
            folder=folder,
            model=args.model,
            recursive=not args.no_recursive,
            upsert=args.upsert,
        )

    if result["images_found"] == 0:
        print("[INFO] No supported images found.")
        return

    print(
        "[OK] "
        f"inserted={result['inserted']} "
        f"updated={result['updated']} "
        f"faces={result['faces_indexed']} "
        f"skipped={result['skipped_images']}"
    )


if __name__ == "__main__":
    main()
