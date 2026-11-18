"""add embedding column to chunks

Revision ID: 0d252020a3bd
Revises: 732a03eee5d1
Create Date: 2026-11-18 10:28:08.980186

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0d252020a3bd'
down_revision: Union[str, Sequence[str], None] = '732a03eee5d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # The first migration already creates it; repeated here so this migration stands on its own.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # 768 = BAAI/bge-base-en-v1.5 output size. Nullable: rows are embedded after loading.
    op.execute("ALTER TABLE chunks ADD COLUMN embedding vector(768)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER TABLE chunks DROP COLUMN embedding")
