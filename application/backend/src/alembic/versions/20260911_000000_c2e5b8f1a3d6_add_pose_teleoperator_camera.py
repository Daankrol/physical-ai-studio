"""add pose teleoperator camera to environment_robots

Adds a nullable ``tele_operator_camera_id`` foreign key to ``environment_robots`` so a robot's
teleoperator can be a pose estimator reading an environment camera, alongside the existing
``robot``/``none`` variants. ``tele_operator_type`` gains a third value, ``"pose"``, still fitting
the existing ``String(16)`` column.

NO ACTION (not CASCADE), matching ``tele_operator_robot_id``: a camera in use as a pose
teleoperator cannot be deleted out from under it (see 20260629_000000_a7c1e9f4b2d3). The service
layer's in-use guard is extended separately in the same change.

SQLite cannot ``ALTER TABLE ... ADD COLUMN ... REFERENCES`` in one step, so this uses Alembic's
batch mode (see ``render_as_batch=True`` in env.py), which rebuilds the table under the hood.

Revision ID: c2e5b8f1a3d6
Revises: 1510153d39a2
Create Date: 2026-09-11 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c2e5b8f1a3d6"
down_revision: str | Sequence[str] | None = "1510153d39a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable pose-teleoperator camera FK."""
    with op.batch_alter_table("environment_robots") as batch_op:
        batch_op.add_column(sa.Column("tele_operator_camera_id", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_environment_robots_tele_operator_camera_id",
            "project_cameras",
            ["tele_operator_camera_id"],
            ["id"],
            ondelete="NO ACTION",
        )


def downgrade() -> None:
    """Drop the pose-teleoperator camera FK.

    Any row with ``tele_operator_type == 'pose'`` loses its camera reference and reads back as
    "no teleoperator" — same defensive-degradation behaviour the repository already applies to
    an unrecognized ``tele_operator_type``.
    """
    with op.batch_alter_table("environment_robots") as batch_op:
        batch_op.drop_constraint("fk_environment_robots_tele_operator_camera_id", type_="foreignkey")
        batch_op.drop_column("tele_operator_camera_id")
