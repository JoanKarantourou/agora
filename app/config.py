"""Settings read from the environment, with a local .env file as a fallback."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Variables already set in the environment win over the .env file.
load_dotenv(ROOT / ".env")


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set; copy .env.example to .env or export it")
    return url
