from uuid import UUID

from db.schema import EnvironmentCameraDB, EnvironmentRobotDB, ProjectEnvironmentDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.environment import (
    CameraEnvironmentConfiguration,
    Environment,
    RobotEnvironmentConfiguration,
    TeleoperatorNone,
    TeleoperatorPose,
    TeleoperatorRobot,
)


class ProjectEnvironmentMapper(IBaseMapper):
    """Mapper for Environment schema entity <-> DB entity conversions."""

    @staticmethod
    def build_robot_links(db_schema: Environment) -> list[EnvironmentRobotDB]:
        """Build standalone robot join rows (not attached to a parent) from the schema."""
        links = []
        for robot in db_schema.robots:
            tele_operator = robot.tele_operator
            links.append(
                EnvironmentRobotDB(
                    environment_id=str(db_schema.id),
                    robot_id=str(robot.robot_id),
                    tele_operator_type=tele_operator.type,
                    tele_operator_robot_id=(
                        str(tele_operator.robot_id) if isinstance(tele_operator, TeleoperatorRobot) else None
                    ),
                    tele_operator_camera_id=(
                        str(tele_operator.camera_id) if isinstance(tele_operator, TeleoperatorPose) else None
                    ),
                )
            )
        return links

    @staticmethod
    def build_camera_links(db_schema: Environment) -> list[EnvironmentCameraDB]:
        """Build standalone camera join rows (not attached to a parent) from the schema."""
        return [
            EnvironmentCameraDB(
                environment_id=str(db_schema.id),
                camera_id=str(camera.camera_id),
            )
            for camera in db_schema.cameras
        ]

    @staticmethod
    def to_schema(db_schema: Environment) -> ProjectEnvironmentDB:
        """Convert Environment schema to db model."""
        return ProjectEnvironmentDB(
            id=str(db_schema.id),
            project_id="",  # Will be set by repository
            name=db_schema.name,
            created_at=db_schema.created_at,
            updated_at=db_schema.updated_at,
            robot_links=ProjectEnvironmentMapper.build_robot_links(db_schema),
            camera_links=ProjectEnvironmentMapper.build_camera_links(db_schema),
        )

    @staticmethod
    def from_schema(model: ProjectEnvironmentDB) -> Environment:
        """Convert Environment db entity to schema."""
        robots = [
            RobotEnvironmentConfiguration(
                robot_id=UUID(link.robot_id),
                tele_operator=ProjectEnvironmentMapper._teleoperator_from_link(link),
            )
            for link in model.robot_links
        ]

        cameras = [CameraEnvironmentConfiguration(camera_id=UUID(link.camera_id)) for link in model.camera_links]

        return Environment(
            id=model.id,
            name=model.name,
            robots=robots,
            cameras=cameras,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _teleoperator_from_link(link: EnvironmentRobotDB) -> TeleoperatorRobot | TeleoperatorPose | TeleoperatorNone:
        if link.tele_operator_type == "robot" and link.tele_operator_robot_id is not None:
            return TeleoperatorRobot(robot_id=UUID(link.tele_operator_robot_id))
        if link.tele_operator_type == "pose" and link.tele_operator_camera_id is not None:
            return TeleoperatorPose(camera_id=UUID(link.tele_operator_camera_id))
        return TeleoperatorNone()
