"""create chunks table

Revision ID: 732a03eee5d1
Revises: 
Create Date: 2026-11-17 15:34:35.463641

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '732a03eee5d1'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # pgvector is needed from Phase 3 on. db/init creates it for the local Docker container, but
    # other databases (CI, a fresh server) only get it here.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE chunks (
            chunk_id    text PRIMARY KEY,
            source_id   text NOT NULL,
            author      text NOT NULL,
            work        text NOT NULL,
            translator  text,
            part        text NOT NULL,
            para_start  int NOT NULL,
            para_end    int NOT NULL,
            word_count  int NOT NULL,
            text        text NOT NULL,
            created_at  timestamptz DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE chunks")
    # The vector extension is left in place: the local db/init script also creates it, so this
    # migration does not own it.
