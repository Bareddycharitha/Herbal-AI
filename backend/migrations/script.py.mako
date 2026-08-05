"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma}
Create Date: ${create_date}

"""
from typing import Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades}


def downgrade() -> None:
    ${downgrades}
