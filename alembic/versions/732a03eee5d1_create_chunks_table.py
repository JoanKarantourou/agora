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
