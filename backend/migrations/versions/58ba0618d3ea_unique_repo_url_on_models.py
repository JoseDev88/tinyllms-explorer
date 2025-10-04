"""unique repo_url on models

Revision ID: 58ba0618d3ea
Revises: 937d0f293c2d
Create Date: 2025-10-02 22:21:19.819717
"""

from alembic import op
import sqlalchemy as sa

# ---- REQUIRED HEADERS ----
revision = "58ba0618d3ea"
down_revision = "937d0f293c2d"
branch_labels = None
depends_on = None
# --------------------------

def upgrade() -> None:
    # Use a UNIQUE INDEX (works on SQLite)
    op.create_index(
        "ix_models_repo_url_unique",
        "models",
        ["repo_url"],
        unique=True,
    )

def downgrade() -> None:
    op.drop_index("ix_models_repo_url_unique", table_name="models")
