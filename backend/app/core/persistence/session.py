"""Database engine and session factory."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.env_loader import load_env_file


_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_env_file(_ENV_PATH)


def create_engine_from_env(default_dsn: str | None = None):
    dsn = os.getenv("DATABASE_URL") or default_dsn
    if not dsn:
        raise ValueError("DATABASE_URL is required")
    return create_engine(dsn, future=True)


def create_session_factory(dsn: str | None = None) -> sessionmaker[Session]:
    engine = create_engine_from_env(default_dsn=dsn)
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
