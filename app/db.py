"""Postgres connections for the app and the scripts."""

import re

import psycopg

from app.config import get_database_url

CONNECT_TIMEOUT = 10  # seconds; fail fast instead of hanging when the database is down


def psycopg_url(url: str) -> str:
    """psycopg wants a plain libpq URL, without SQLAlchemy's "+psycopg" driver suffix."""
    return re.sub(r"^postgresql\+\w+://", "postgresql://", url)


def connect() -> psycopg.Connection:
    return psycopg.connect(psycopg_url(get_database_url()), connect_timeout=CONNECT_TIMEOUT)
